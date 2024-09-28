import torch
import torch.nn as nn
import numpy as np

class PNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, poly_degree, width, depth):
        super(PNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.poly_degree = poly_degree
        self.width = width
        self.depth = depth

        self.fc_in = nn.Linear(state_size + parameter_size, width)

        self.poly_layers = nn.ModuleList([PolynomialLayer(width, poly_degree) for _ in range(depth)])

        self.fc_out = nn.Linear(width, state_size)

    def generate_grid(self, batch_size, device):
        x = torch.linspace(-1, 1, self.nx, device=device)
        t = torch.linspace(-1, 1, self.T_out, device=device)

        x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
        return x_grid.unsqueeze(0).repeat(batch_size, 1, 1), t_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, u0, P):
        batch_size = u0.shape[0]
        device = u0.device

        # Generate spatial-temporal grid
        x_grid, t_grid = self.generate_grid(batch_size, device)

        # Prepare input features
        P_expanded = P.unsqueeze(1).unsqueeze(2).expand(-1, self.nx, self.T_out, -1)
        u0_interp = nn.functional.interpolate(u0.permute(0, 3, 1, 2), size=(self.nx, self.T_out), mode='bilinear', align_corners=False)
        u0_interp = u0_interp.permute(0, 2, 3, 1)

        # Combine u0 and P
        inputs = torch.cat([u0_interp, P_expanded], dim=-1)

        # Apply initial linear layer
        x = self.fc_in(inputs)

        # Apply polynomial layers
        for layer in self.poly_layers:
            x = layer(x, x_grid, t_grid)

        # Apply final linear layer
        x = self.fc_out(x)

        return x

class PolynomialLayer(nn.Module):
    def __init__(self, width, poly_degree):
        super(PolynomialLayer, self).__init__()
        self.width = width
        self.poly_degree = poly_degree
        self.weights_x = nn.Parameter(torch.randn(width, poly_degree + 1))
        self.weights_t = nn.Parameter(torch.randn(width, poly_degree + 1))
        self.bias = nn.Parameter(torch.zeros(width))
        self.activation = nn.Tanh()

    def forward(self, x, x_grid, t_grid):
        # Generate Legendre polynomial basis for x and t
        P_x = self.legendre_basis(x_grid, self.poly_degree)
        P_t = self.legendre_basis(t_grid, self.poly_degree)

        # Apply polynomial transformation
        out_x = torch.einsum('bijd,md->bijm', P_x, self.weights_x)
        out_t = torch.einsum('bijd,md->bijm', P_t, self.weights_t)

        out = x * (out_x + out_t) + self.bias

        return self.activation(out)

    def legendre_basis(self, x, degree):
        P = torch.ones_like(x)
        P_list = [P]

        if degree > 0:
            P = x.clone()
            P_list.append(P)

        for n in range(2, degree + 1):
            P = ((2 * n - 1) * x * P_list[-1] - (n - 1) * P_list[-2]) / n
            P_list.append(P)

        return torch.stack(P_list, dim=-1)

# # Example usage:
# batch_size = 1
# nx = 20
# T_in, T_out = 5, 25
# state_size = 1
# parameter_size = 6
# poly_degree = 5
# width, depth = 32, 4
# model = PNO1DTime(nx, T_in, T_out, state_size, parameter_size, poly_degree, width, depth)
# u0 = torch.randn(batch_size, nx, T_in, state_size)
# P = torch.randn(batch_size, parameter_size)
# output = model(u0, P)  # Shape: [32, 64, 20, 5]

# print(output.shape)