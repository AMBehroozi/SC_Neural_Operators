import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

class MGNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, width, depth):
        super(MGNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.width = width
        self.depth = depth

        self.fc_in = nn.Linear(state_size + parameter_size + 2, width)  # +2 for x and t coordinates

        self.multipole_layers = nn.ModuleList([MultipoleGraphLayer(width) for _ in range(depth)])

        self.fc_out = nn.Linear(width, state_size)

    def generate_grid(self, batch_size, device):
        x = torch.linspace(0, 1, self.nx, device=device)
        t = torch.linspace(0, 1, self.T_out, device=device)

        x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
        return x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)

    def create_graph_structure(self, batch_size, device):
        num_nodes = self.nx * self.T_out
        edge_index = []

        # Spatial connections
        for t in range(self.T_out):
            for i in range(self.nx - 1):
                edge_index.append([t * self.nx + i, t * self.nx + i + 1])
                edge_index.append([t * self.nx + i + 1, t * self.nx + i])

        # Temporal connections
        for x in range(self.nx):
            for t in range(self.T_out - 1):
                edge_index.append([t * self.nx + x, (t + 1) * self.nx + x])
                edge_index.append([(t + 1) * self.nx + x, t * self.nx + x])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

        # Repeat for each batch
        edge_index = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        edge_index += torch.arange(0, batch_size * num_nodes, num_nodes, device=device).unsqueeze(1).unsqueeze(2)
        edge_index = edge_index.view(2, -1)

        return edge_index

    def forward(self, u0, P):
        batch_size = u0.shape[0]
        device = u0.device

        # Generate spatial-temporal grid
        x_grid, t_grid = self.generate_grid(batch_size, device)

        # Create graph structure
        edge_index = self.create_graph_structure(batch_size, device)

        # Prepare input features
        P_expanded = P.unsqueeze(1).repeat(1, self.nx * self.T_out, 1)
        u0_interp = nn.functional.interpolate(u0.permute(0, 3, 1, 2), size=(self.nx, self.T_out), mode='bilinear', align_corners=False)
        u0_interp = u0_interp.permute(0, 2, 3, 1).reshape(batch_size, -1, self.state_size)

        # Combine all inputs
        x = torch.cat([
            u0_interp,
            P_expanded,
            x_grid.repeat(batch_size, 1, 1),
            t_grid.repeat(batch_size, 1, 1)
        ], dim=-1)

        # Apply network
        x = self.fc_in(x)
        x = x.view(-1, self.width)

        for layer in self.multipole_layers:
            x = layer(x, edge_index)

        x = self.fc_out(x)

        return x.view(batch_size, self.nx, self.T_out, self.state_size)

class MultipoleGraphLayer(nn.Module):
    def __init__(self, width):
        super(MultipoleGraphLayer, self).__init__()
        self.conv = gnn.GCNConv(width, width)
        self.norm = nn.LayerNorm(width)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x_res = x
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x + x_res)
        return x

# # Example usage:
# nx = 64
# T_in, T_out = 10, 20
# state_size = 5
# parameter_size = 3
# width, depth = 32, 4
# model = MGNO1DTime(nx, T_in, T_out, state_size, parameter_size, width, depth)
# u0 = torch.randn(32, nx, T_in, state_size)
# P = torch.randn(32, parameter_size)
# output = model(u0, P)  # Shape: [32, 64, 20, 5]

# print(output.shape)