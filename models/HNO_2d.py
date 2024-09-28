import torch
import torch.nn as nn
import torch.nn.functional as F


class HilbertSpectralConv2d(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(HilbertSpectralConv2d, self).__init__()

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (width * width))
        self.weights_real = nn.Parameter(self.scale * torch.randn(width, width, self.modes1, self.modes2, dtype=torch.float32))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(width, width, self.modes1, self.modes2, dtype=torch.float32))

    def compl_mul2d(self, input_real, input_imag, weights_real, weights_imag):
        return torch.einsum("bwxy,wwxy->bwxy", input_real, weights_real) - \
               torch.einsum("bwxy,wwxy->bwxy", input_imag, weights_imag), \
               torch.einsum("bwxy,wwxy->bwxy", input_real, weights_imag) + \
               torch.einsum("bwxy,wwxy->bwxy", input_imag, weights_real)

    def hilbert_2d(self, x):
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        nx, ny = x.shape[-2:]
        ny_half = ny // 2 + 1
        
        # Create Hilbert transform mask
        mask = torch.ones(nx, ny_half, dtype=torch.float32, device=x.device)
        mask[0, 0] = 1  # DC component
        mask[1:nx//2, 1:] = 2  # Positive frequencies
        mask[nx//2:, 1:] = 0  # Negative frequencies (for nx odd)
        if nx % 2 == 0:
            mask[nx//2, 1:] = 1  # Nyquist frequency (for nx even)

        # Apply mask
        x_ht_real = x_ft.real * mask
        x_ht_imag = x_ft.imag * mask

        return x_ht_real, x_ht_imag

    def forward(self, x):
        nbatch, width, nx, ny = x.shape

        # Apply Hilbert transform
        x_ht_real, x_ht_imag = self.hilbert_2d(x)

        # Initialize output Fourier coefficients
        out_ft_real = torch.zeros_like(x_ht_real)
        out_ft_imag = torch.zeros_like(x_ht_imag)

        # Apply spectral convolution to the first modes
        out_ft_real[:, :, :self.modes1, :self.modes2], out_ft_imag[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ht_real[:, :, :self.modes1, :self.modes2], 
            x_ht_imag[:, :, :self.modes1, :self.modes2],
            self.weights_real,
            self.weights_imag
        )

        # Perform inverse Fourier transform
        x_out = torch.fft.irfft2(torch.complex(out_ft_real, out_ft_imag), s=(nx, ny), dim=(-2, -1))

        return x_out


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
        self.padding = 20 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(T_in + self.parameter_size + 2, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = HilbertSpectralConv2d(self.width, self.modes1, self.modes2)
        self.conv1 = HilbertSpectralConv2d(self.width, self.modes1, self.modes2)
        self.conv2 = HilbertSpectralConv2d(self.width, self.modes1, self.modes2)
        self.conv3 = HilbertSpectralConv2d(self.width, self.modes1, self.modes2)
        
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
        u = torch.cat((u, par_ex, grid), dim=-1) #[nb, nx, T_out, T_in + 1:x +1:t + 2: parameters]
        out = self.process_tensor(u)
        return out
    
    def get_grid(self, x, t, device):
        batchsize, size_x = x.shape[0], x.shape[1]
        gridx = x.reshape(batchsize, size_x, 1, 1).repeat([1, 1, self.T_out, 1])
        gridy = t.reshape(batchsize, 1, self.T_out, 1).repeat([1, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# # # Usage
# nbatch = 16
# s0 = 64
# T_in = 5
# T_out = 25
# parameter_size = 1
# modes1, modes2,  width = 8, 8, 10
# u_in = torch.rand(nbatch, s0, T_in)
# parameters = torch.rand(nbatch, parameter_size)
# x = torch.linspace(0, 1, s0).unsqueeze(0).repeat(nbatch, 1)
# t = torch.linspace(0, 1, T_in + T_out)[..., T_in:].unsqueeze(0).repeat(nbatch, 1)
# model = FNO2d(modes1, modes2,  width, T_in, T_out, parameter_size, 1)
# u_out = model(u_in, x, t, parameters)
# u_out.shape   # [nb, s0, T_out, state_size]