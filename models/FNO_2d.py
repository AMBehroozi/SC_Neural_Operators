import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It performs 2D FFT, a linear transform, and Inverse 2D FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply along the spatial dimension
        self.modes2 = modes2  # Number of Fourier modes to multiply along the temporal dimension

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, t), (in_channel, out_channel, x, t) -> (batch, out_channel, x, t)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_t = x.shape[2], x.shape[3]

        # Compute Fourier coefficients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Prepare output array, initialize to zero
        out_ft = torch.zeros(batchsize, self.out_channels, size_x, size_t//2 + 1, dtype=torch.cfloat, device=x.device)

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size_x, size_t), dim=[2, 3])
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, T_in, T_out, state_size, parameter_size):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.T_in = T_in
        self.T_out = T_out

        self.state_size = state_size
        self.parameter_size = parameter_size
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(T_in + self.parameter_size + 2, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.state_size)

    def process_tensor(self, u):
        # Initial fully connected layer and reshape
        u = self.fc0(u)
        u = u.permute(0, 3, 1, 2)
        u = F.pad(u, [0, self.padding, 0, self.padding])

        # Process through convolution and weight layers
        layers = [(self.conv0, self.w0), (self.conv1, self.w1), 
                (self.conv2, self.w2), (self.conv3, self.w3)]
        for conv, weight in layers:
            u1 = conv(u)
            u2 = weight(u)
            u = u1 + u2
            if conv != self.conv3:  # Apply activation function except for the last layer
                u = F.gelu(u)

        # Final steps after the loop
        u = u[..., :-self.padding, :-self.padding]
        u = u.permute(0, 2, 3, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def forward(self, u, x, t, par):
        nbatch, s0 = u.shape[0], u.shape[1]
        u = u.reshape(nbatch, s0, 1, self.T_in).repeat([1, 1, self.T_out, 1])
        par_ex = par.reshape(nbatch, 1, 1, self.parameter_size).repeat([1, s0, self.T_out, 1])
        grid = self.get_grid(x, t, u.device)
        u = torch.cat((u, par_ex, grid), dim=-1)
        out = self.process_tensor(u)
        return out
    
    def get_grid(self, x, t, device):
        batchsize, size_x = x.shape[0], x.shape[1]
        gridx = x.reshape(batchsize, size_x, 1, 1).repeat([1, 1, self.T_out, 1])
        gridy = t.reshape(batchsize, 1, self.T_out, 1).repeat([1, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# # Usage
nbatch = 13
s0 = 64
T_in = 5
T_out = 25
parameter_size = 2
modes1, modes2,  width = 8, 8, 10
u_in = torch.rand(nbatch, s0, T_in)
parameters = torch.rand(nbatch, parameter_size)
x = torch.linspace(0, 1, s0).unsqueeze(0).repeat(nbatch, 1)
t = torch.linspace(0, 1, T_in + T_out)[..., T_in:].unsqueeze(0).repeat(nbatch, 1)
model = FNO2d(modes1, modes2,  width, T_in, T_out, parameter_size, 2)
u_out = model(u_in, x, t, parameters)
u_out.shape   # [nb, s0, T_out, state_size]
