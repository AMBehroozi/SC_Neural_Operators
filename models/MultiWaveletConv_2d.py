import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT

class MultiWaveletConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelets=['db5', 'db6'], mode='symmetric'):
        super(MultiWaveletConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size
        self.wavelets = wavelets
        self.mode = mode

        self.dwts = nn.ModuleList([DWT(J=self.level, mode=self.mode, wave=wavelet) for wavelet in self.wavelets])
        self.idwts = nn.ModuleList([IDWT(mode=self.mode, wave=wavelet) for wavelet in self.wavelets])

        self.scale = (1 / (in_channels * out_channels * len(wavelets)))
        self.weights = nn.ParameterList([nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *size)) for _ in wavelets])

    def mul2d(self, input, weights):
        if input.dim() == 4:  # Low-frequency coefficients
            if input.shape[-2:] != weights.shape[-2:]:
                weights = F.interpolate(weights, size=input.shape[-2:], mode='bilinear', align_corners=False)
            return torch.einsum("bixy,ioxy->boxy", input, weights)
        elif input.dim() == 5:  # High-frequency coefficients
            if input.shape[-3:-1] != weights.shape[-2:]:
                weights = F.interpolate(weights, size=input.shape[-3:-1], mode='bilinear', align_corners=False)
            weights = weights.unsqueeze(-1).expand(-1, -1, -1, -1, input.size(-1))
            return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        else:
            raise ValueError(f"Unexpected input dimension: {input.dim()}")

    def forward(self, x):
        original_size = x.shape
        outputs = []

        for dwt, idwt, weight in zip(self.dwts, self.idwts, self.weights):
            x_low, x_high = dwt(x)

            out_low = self.mul2d(x_low, weight)
            out_high = [self.mul2d(x_h, weight) for x_h in x_high]

            out = idwt((out_low, out_high))
            
            # Crop the output to match the original input size
            if out.shape != original_size:
                out = out[:, :, :original_size[2], :original_size[3]]
            
            outputs.append(out)
        
        # Combine outputs from different wavelets
        return torch.mean(torch.stack(outputs), dim=0)

class MWNO2d(nn.Module):
    def __init__(self, levels, size, width, T_in, T_out, state_size, parameter_size):
        super(MWNO2d, self).__init__()

        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.levels = levels
        self.size = size
        self.width = width
        self.padding = 2  # Ensure this is an odd number

        self.fc0 = nn.Linear(T_in + self.parameter_size + 2, self.width)
        self.conv0 = MultiWaveletConv2d(self.width, self.width, self.levels, self.size)
        self.conv1 = MultiWaveletConv2d(self.width, self.width, self.levels, self.size)
        self.conv2 = MultiWaveletConv2d(self.width, self.width, self.levels, self.size)
        self.conv3 = MultiWaveletConv2d(self.width, self.width, self.levels, self.size)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.state_size)

    def process_tensor(self, u):
        u = self.fc0(u)
        u = u.permute(0, 3, 1, 2)
        
        pad = self.padding // 2
        u = F.pad(u, [pad, pad, pad, pad], mode='reflect')

        layers = [(self.conv0, self.w0), (self.conv1, self.w1), 
                  (self.conv2, self.w2), (self.conv3, self.w3)]
        for conv, weight in layers:
            u1 = conv(u)
            u2 = weight(u)
            u = u1 + u2
            if conv != self.conv3:
                u = F.gelu(u)

        u = u[..., pad:-pad, pad:-pad]
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

# # Usage remains the same as in the original code
# nbatch = 1
# s0 = 64
# T_in = 5
# T_out = 25
# parameter_size = 6
# state_size = 1
# levels, size, width = 3, [s0, s0], 20
# u_in = torch.rand(nbatch, s0, T_in)
# parameters = torch.rand(nbatch, parameter_size)
# x = torch.linspace(0, 1, s0).unsqueeze(0).repeat(nbatch, 1)
# t = torch.linspace(0, 1, T_in + T_out)[..., T_in:].unsqueeze(0).repeat(nbatch, 1)
# model = MWNO2d(levels, size, width, T_in, T_out, state_size, parameter_size)
# u_out = model(u_in, x, t, parameters)
# print(f"Final output shape: {u_out.shape}")
# print()