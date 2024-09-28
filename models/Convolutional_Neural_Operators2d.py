import torch
import torch.nn as nn
import torch.nn.functional as F

class CNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, width, depth, kernel_size=3):
        super(CNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.width = width
        self.depth = depth
        self.kernel_size = kernel_size

        # Input projection
        self.fc_in = nn.Linear(state_size + parameter_size, width - 2)  # -2 to leave room for grid channels

        # Convolutional layers with residual connections
        self.conv_layers = nn.ModuleList([
            ResidualConvBlock2D(width, width, kernel_size) for _ in range(depth)
        ])

        # Output projection
        self.fc_out = nn.Linear(width, state_size)

    def generate_grid(self, batch_size, device):
        x = torch.linspace(0, 1, self.nx, device=device)
        t = torch.linspace(0, 1, self.T_out, device=device)
        x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
        return x_grid.unsqueeze(0).repeat(batch_size, 1, 1), t_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, u0, P):
        batch_size = u0.shape[0]
        device = u0.device

        # Generate spatial-temporal grid
        x_grid, t_grid = self.generate_grid(batch_size, device)

        # Prepare input features
        P_expanded = P.unsqueeze(1).unsqueeze(2).expand(-1, self.nx, self.T_out, -1)
        u0_interp = F.interpolate(u0.permute(0, 3, 1, 2), size=(self.nx, self.T_out), mode='bilinear', align_corners=False)
        u0_interp = u0_interp.permute(0, 2, 3, 1)

        # Combine u0 and P
        inputs = torch.cat([u0_interp, P_expanded], dim=-1)

        # Apply input projection
        x = self.fc_in(inputs)

        # Add grid information as separate channels
        x = torch.cat([x, x_grid.unsqueeze(-1), t_grid.unsqueeze(-1)], dim=-1)

        # Apply convolutional layers with residual connections
        x = x.permute(0, 3, 1, 2)  # [batch, channels, nx, T_out]
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Apply output projection
        x = x.permute(0, 2, 3, 1)  # [batch, nx, T_out, channels]
        x = self.fc_out(x)

        return x

class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConvBlock2D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

# # Example usage:
# batch_size = 1
# nx = 40
# T_in, T_out = 5, 25
# state_size = 1
# parameter_size = 6
# width, depth = 32, 4
# kernel_size = 3
# model = CNO1DTime(nx, T_in, T_out, state_size, parameter_size, width, depth, kernel_size)
# u0 = torch.randn(batch_size, nx, T_in, state_size)
# P = torch.randn(batch_size, parameter_size)
# output = model(u0, P)  # Shape: [32, 64, 20, 5]

# print(output.shape)
# print()