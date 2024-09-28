import torch
import torch.nn as nn
import torch.nn.functional as F

class MNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, width, depth):
        super(MNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.width = width
        self.depth = depth

        self.fc_in = nn.Linear(state_size + parameter_size + 2, width)  # +2 for x and t coordinates

        self.multipole_layers = nn.ModuleList([MultipoleLayer1DTime(width) for _ in range(depth)])

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

        # Combine all inputs
        inputs = torch.cat([
            u0_interp,
            P_expanded,
            x_grid.unsqueeze(-1),
            t_grid.unsqueeze(-1)
        ], dim=-1)

        # Apply network
        x = self.fc_in(inputs)

        for layer in self.multipole_layers:
            x = layer(x)

        x = self.fc_out(x)

        return x

class MultipoleLayer1DTime(nn.Module):
    def __init__(self, width):
        super(MultipoleLayer1DTime, self).__init__()
        self.local_conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.up_conv = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=2)
        self.down_conv = nn.ConvTranspose2d(width, width, kernel_size=4, padding=1, stride=2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [batch, width, nx, T_out]

        local = self.local_conv(x)

        # Upsampling (coarsening)
        up = self.up_conv(x)

        # Downsampling (refining)
        down = self.down_conv(up)

        # Ensure the downsampled tensor has the same size as the input
        if down.shape != x.shape:
            down = F.interpolate(down, size=x.shape[2:], mode='bilinear', align_corners=False)

        return (local + down).permute(0, 2, 3, 1)  # [batch, nx, T_out, width]

# # Example usage:
# nx = 64
# T_in, T_out = 10, 20
# state_size = 5
# parameter_size = 3
# width, depth = 32, 4
# model = MNO1DTime(nx, T_in, T_out, state_size, parameter_size, width, depth)
# u0 = torch.randn(32, nx, T_in, state_size)
# P = torch.randn(32, parameter_size)
# output = model(u0, P)  # Shape: [32, 64, 20, 5]