import torch
import torch.nn as nn

class DNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, branch_layers, trunk_layers):
        super(DNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size  # State size
        self.parameter_size = parameter_size  # Number of parameters

        # Branch net
        branch_layers = [nx * T_in * state_size + parameter_size] + branch_layers
        self.branch_net = self._build_mlp(branch_layers)

        # Trunk net
        trunk_layers = [3] + trunk_layers  # 3 for x, t, s coordinates
        self.trunk_net = self._build_mlp(trunk_layers)

        # Ensure the output sizes of branch and trunk nets match
        assert branch_layers[-1] == trunk_layers[-1], "Branch and trunk output dimensions must match"

        self.output_dim = branch_layers[-1]

    def _build_mlp(self, layers):
        mlp = nn.ModuleList()
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                mlp.append(nn.ReLU())
        return nn.Sequential(*mlp)

    def _prepare_grid(self, device):
        x = torch.linspace(0, 1, self.nx, device=device)
        t = torch.linspace(0, 1, self.T_out, device=device)
        s = torch.linspace(0, 1, self.state_size, device=device)
        grid_x, grid_t, grid_s = torch.meshgrid(x, t, s, indexing='ij')
        grid = torch.stack([grid_x, grid_t, grid_s], dim=-1).reshape(-1, 3)
        return grid

    def forward(self, u0, P):
        batch_size = u0.shape[0]

        # Branch net
        u0_flat = u0.reshape(batch_size, -1)
        branch_input = torch.cat([u0_flat, P], dim=1)
        branch_output = self.branch_net(branch_input)

        # Trunk net
        trunk_input = self._prepare_grid(u0.device)
        trunk_output = self.trunk_net(trunk_input)

        # Combine branch and trunk outputs
        output = torch.einsum('bi,ni->bn', branch_output, trunk_output)

        # Reshape to match desired output shape
        return output.reshape(batch_size, self.nx, self.T_out, self.state_size)

# # Example usage:
# batch_size = 1
# nx = 20  # Spatial dimension
# T_in = 5  # Number of input time steps
# T_out = 25  # Number of output time steps
# state_size = 1  # State size
# parameter_size = 6  # Number of PDE parameters
# branch_layers = [256, 256, 256]
# trunk_layers = [64, 128, 256]
# model = DNO1DTime(nx, T_in, T_out, state_size, parameter_size, branch_layers, trunk_layers)
# u0 = torch.randn(batch_size, nx, T_in, state_size)  # Batch of 32 initial conditions
# P = torch.randn(batch_size, parameter_size)  # Batch of 32 parameter sets
# u_pred = model(u0, P)  # Shape: [32, 64, 20, 5]
# print(u_pred.shape)

# (torch.Size([1, 20, 5, 1]), torch.Size([1, 6]))