import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        1D Spectral convolution layer.

        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        - modes1 (int): Number of Fourier modes to multiply (not directly used in this implementation)
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        # Complex-valued weights for the Fourier space transformation
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, dtype=torch.cfloat))

    def forward(self, x):
        """
        Forward pass of the spectral convolution.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, in_channels]

        Returns:
        - Tensor: Output tensor of shape [batch_size, out_channels]
        """
        # Compute Fourier coefficients
        x_ft = torch.fft.fft(x)
        # Initialize output Fourier coefficients
        out_ft = torch.zeros(x.shape[0], self.out_channels, dtype=torch.cfloat, device=x.device)
        # Multiply relevant Fourier modes
        out_ft = torch.einsum("bi,io->bo", x_ft, self.weights)
        # Return to physical space
        x = torch.fft.ifft(out_ft).real
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        """
        1D Fourier Neural Operator.

        Args:
        - modes (int): Number of Fourier modes to multiply
        - width (int): Number of channels in the convolutional layers
        """
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        # Initial fully connected layer to project input to 'width' dimensions
        self.fc0 = nn.Linear(3, self.width)

        # Four spectral convolution layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        # Four linear layers for the residual connections
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)

        # Final fully connected layers
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass of the FNO.

        Args:
        - x (Tensor): Input tensor of shape [batch_size, 3] representing (x, y, t) coordinates

        Returns:
        - Tensor: Output tensor of shape [batch_size] representing the predicted values
        """
        # Initial projection
        x = self.fc0(x)

        # Four layers of the integral operator
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Final fully connected layers
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return x.squeeze(-1)  # Remove the last dimension if it's singleton

# Define the function to approximate
def target_function(x, y, t):
    return torch.sin(2 * np.pi * x) * torch.cos(2 * np.pi * y) * torch.exp(-t)

# Generate data with specific intervals
def generate_data(nx, ny, nt, dx, dy, dt):
    x = torch.arange(0, nx * dx, dx)
    y = torch.arange(0, ny * dy, dy)
    t = torch.arange(0, nt * dt, dt)
    grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing='ij')
    inputs = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_t.flatten()], dim=1)
    outputs = target_function(grid_x, grid_y, grid_t).flatten()
    return inputs, outputs

# Set up the model and training parameters
modes = 16
width = 64
model = FNO1d(modes, width)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Generate training data
dx, dy, dt = 0.05, 0.05, 0.05
train_inputs, train_outputs = generate_data(20, 20, 20, dx, dy, dt)

# Create DataLoader for training
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_outputs in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
f(x, y, t) = perce(x, y, t)
# Generate test data with halved intervals
test_dx, test_dy, test_dt = dx/2, dy/2, dt/2
test_inputs, test_outputs = generate_data(39, 39, 39, test_dx, test_dy, test_dt)

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(test_inputs)
    test_loss = criterion(test_predictions, test_outputs)
    print(f"Test Loss: {test_loss:.4f}")

# Visualize results
plt.figure(figsize=(15, 5))

# True vs Predicted plot
plt.subplot(131)
plt.scatter(test_outputs.numpy(), test_predictions.numpy(), alpha=0.3)
plt.plot([test_outputs.min(), test_outputs.max()], [test_outputs.min(), test_outputs.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("FNO Predictions vs True Values")

# Error distribution plot
plt.subplot(132)
error = (test_predictions - test_outputs).abs()
plt.hist(error.numpy(), bins=50)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

# 2D slice plot
plt.subplot(133)
slice_idx = test_inputs[:, 2] == test_dt * 19  # Middle time slice
slice_inputs = test_inputs[slice_idx]
slice_outputs = test_outputs[slice_idx]
slice_predictions = test_predictions[slice_idx]

plt.scatter(slice_inputs[:, 0], slice_inputs[:, 1], c=slice_outputs, cmap='viridis', s=20, alpha=0.5)
plt.colorbar(label='True Values')
plt.scatter(slice_inputs[:, 0], slice_inputs[:, 1], c=slice_predictions, cmap='plasma', s=10, alpha=0.5)
plt.colorbar(label='Predicted Values')
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"2D Slice at t={test_dt*19:.2f}")

plt.tight_layout()
plt.show()

print(f"Mean Absolute Error: {error.mean():.4f}")
print(f"Max Absolute Error: {error.max():.4f}")