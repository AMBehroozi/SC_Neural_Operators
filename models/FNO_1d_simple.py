import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It performs FFT, a linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, t), (in_channel, out_channel, t) -> (batch, out_channel, t)
        return torch.einsum("bit,iot->bot", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, T_in, T_out, state_size, parameters_size):
        super(FNO1d, self).__init__()

        self.modes = modes
        self.width = width
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameters_size = parameters_size
        self.padding = 2

        # Initial lifting layer
        self.fc0 = nn.Linear(T_in + 1, self.width)

        # Fourier layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        # W layers for local connections
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Projection layers
        self.fc1 = nn.Linear(self.width + self.parameters_size , 128)
        self.fc2 = nn.Linear(128, 1)


    def process_tensor(self, tensor, t, parameters):
        grid = self.get_grid(t, tensor.device)
        parameters_ex = parameters.unsqueeze(1).expand(-1, self.state_size * self.T_out, -1)
        # P_out_ex = P_out.repeat(1, self.state_size, 1)
        tensor = torch.cat((tensor, grid), dim=-1)
        tensor = self.fc0(tensor)
        tensor = tensor.permute(0, 2, 1)
        tensor = F.pad(tensor, [0, self.padding])

        for conv, w in zip([self.conv0, self.conv1, self.conv2, self.conv3],
                               [self.w0, self.w1, self.w2, self.w3]):
            t1 = conv(tensor)
            t2 = w(tensor)
            tensor = (t1 + t2)
            tensor = F.gelu(tensor)
        tensor = tensor[..., :-self.padding]
        tensor = tensor.permute(0, 2, 1)
        tensor = torch.cat((tensor, parameters_ex), dim=-1)
        tensor = self.fc1(tensor)
        tensor = F.gelu(tensor)
        tensor = self.fc2(tensor)
        return tensor

    def forward(self, u, t, parameters):
        # u: [nb, T_in, state_size]
        # P: [nbatch, T_in+T_out, forcing_size]
        # t: [nbatch, T_out]
        # parameters: [nbatch, parameters_size]


        nbatch, nx = u.shape[0], self.state_size
        output = u.unsqueeze(1).repeat(1, self.T_out, 1, 1)
        tensor_permuted = output.permute(0, 3, 1, 2)  # Now shape is [nbatch, nx, T_out, T_in]
        tensor_reshaped = tensor_permuted.reshape(nbatch, nx, self.T_out * self.T_in)  # Intermediate reshape
        tensor_final = tensor_reshaped.permute(0, 2, 1).reshape(nbatch, nx * self.T_out, self.T_in)
        combined = self.process_tensor(tensor_final, t, parameters)
        return combined.view(nbatch, self.T_out, nx) # shape: [nb, nt, nx]


    def get_grid(self, t_tensor, device):
        grid = t_tensor.repeat(1, self.state_size).unsqueeze(-1).float()
        return grid.to(device)


#         u: [nb, T_in, state_size]
#         P: [nbatch, T_in+T_out, forcing_size]
#         t: [nbatch, T_out]
#         parameters: [nbatch, parameters_size]



# T_in, modes, width, T_out, state_size, parameters_size = 10, 8, 20, 90, 1, 3
# model = FNO1d(modes, width, T_in, T_out, state_size, parameters_size)
# nbatch = 16
# U_in = torch.rand(nbatch, T_in, state_size)
# t_tensor_ = torch.linspace(0, 1, 100)[T_in:].unsqueeze(0).repeat(nbatch, 1).requires_grad_(True).to('cpu')
# parameters = torch.rand(nbatch, parameters_size)

# U_pred = model(U_in, t_tensor_ , parameters)


# print('')

