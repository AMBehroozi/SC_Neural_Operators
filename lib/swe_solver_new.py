# Batch new
import torch
import numpy as np


class SWE_Solver:
    def __init__(self,
                 device,  # Device to use (e.g., 'cuda' or 'cpu')
                 S=64,
                 T=50,
                 L_x=1.0,  # Length of domain in x-direction
                 L_y=1.0,  # Length of domain in y-direction
                 g=9.81,  # Acceleration of gravity [m/s^2]
                 H=0.1,  # Depth of fluid [m]
                 rho_0=1024.0,  # Density of fluid [kg/m^3]
                 N_x=200,  # Number of grid points in x-direction
                 N_y=200,  # Number of grid points in y-direction
                 amp=0.05,
                 sigma=0.1,
                 max_time_step=2000,  # Total number of time steps in simulation
                 ):
        self.device = torch.device(device)

        # Assigning the parameters
        self.S = S
        self.T = T
        self.L_x = torch.tensor(L_x, device=self.device)
        self.L_y = torch.tensor(L_y, device=self.device)
        self.g = torch.tensor(g, device=self.device)
        self.H = torch.tensor(H, device=self.device)
        self.rho_0 = torch.tensor(rho_0, device=self.device)
        self.N_x = N_x
        self.N_y = N_y
        self.amp = amp
        self.sigma = sigma
        self.max_time_step = max_time_step

        # Computational parameters
        self.dx = self.L_x / (self.N_x - 1)
        self.dy = self.L_y / (self.N_y - 1)
        self.dt = 0.1 * min(self.dx, self.dy) / torch.sqrt(self.g * self.H)
        T_end = self.dt * self.max_time_step
        print(f'dt={self.dt}   T_end={T_end}')
        self.x = torch.linspace(-self.L_x / 2, self.L_x / 2, self.N_x, device=self.device)
        self.y = torch.linspace(-self.L_y / 2, self.L_y / 2, self.N_y, device=self.device)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.X = self.X.t()
        self.Y = self.Y.t()
        # Determine the indices for spatial downsampling
        self.sample_indices = torch.linspace(0, self.N_x - 1, self.S, dtype=torch.long, device=self.device)

    def run_batch(self, w_batch):
        n_batch = w_batch.shape[0]
        alfa_batch = w_batch[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, self.N_x, self.N_y)
        beta_batch = w_batch[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, self.N_x, self.N_y)

        # Expand X, Y to match batch size
        X_batch = self.X.expand(n_batch, -1, -1)
        Y_batch = self.Y.expand(n_batch, -1, -1)

        # Initialize state tensors for the batch
        u_n = torch.zeros(n_batch, self.N_x, self.N_y, device=self.device)
        v_n = torch.zeros(n_batch, self.N_x, self.N_y, device=self.device)
        eta_n = self.amp * torch.exp(-((X_batch - alfa_batch * self.L_x) ** 2 / (2 * self.sigma ** 2) +
                                       (Y_batch - beta_batch * self.L_y) ** 2 / (2 * self.sigma ** 2)))

        eta_list = []
        u_list = []
        v_list = []
        # Time step interval for storing results
        save_interval = self.max_time_step // self.T

        for time_step in range(1, self.max_time_step + 1):
            u_np1 = u_n - self.g * self.dt / self.dx * (torch.roll(eta_n, -1, dims=1) - eta_n)
            v_np1 = v_n - self.g * self.dt / self.dy * (torch.roll(eta_n, -1, dims=2) - eta_n)

            # Apply boundary conditions
            v_np1[:, :, -1] = 0.0
            u_np1[:, -1, :] = 0.0

            # Upwind scheme for eta
            h_e = torch.where(u_np1 > 0, eta_n, torch.roll(eta_n, -1, dims=1)) + self.H
            h_w = torch.where(u_np1 > 0, torch.roll(eta_n, 1, dims=1), eta_n) + self.H
            h_n = torch.where(v_np1 > 0, eta_n, torch.roll(eta_n, -1, dims=2)) + self.H
            h_s = torch.where(v_np1 > 0, torch.roll(eta_n, 1, dims=2), eta_n) + self.H

            uhwe = u_np1 * h_e - torch.roll(u_np1, 1, dims=1) * h_w
            vhns = v_np1 * h_n - torch.roll(v_np1, 1, dims=2) * h_s

            eta_np1 = eta_n - self.dt * (uhwe / self.dx + vhns / self.dy)

            # Update arrays for the next iteration
            u_n, v_n, eta_n = u_np1, v_np1, eta_np1
            if time_step % save_interval == 0:
                eta_downsampled = eta_n[:, self.sample_indices][:, :, self.sample_indices]
                u_downsampled = u_n[:, self.sample_indices][:, :, self.sample_indices]
                v_downsampled = v_n[:, self.sample_indices][:, :, self.sample_indices]

                eta_list.append(eta_downsampled)
                u_list.append(u_downsampled)
                v_list.append(v_downsampled)

        # Convert lists to tensors of shape [batch, S, S, T]
        eta_tensor = torch.stack(eta_list, dim=-1)
        u_tensor = torch.stack(u_list, dim=-1)
        v_tensor = torch.stack(v_list, dim=-1)

        return eta_tensor, u_tensor, v_tensor

# function for calculation of PDE loss
def pde_loss(h, u, v, X, Y, T, g, S, T_out):
    """
    Compute the left-hand side terms of the 2D shallow water equations (SWE)
    and compute derivatives using automatic differentiation (autograd).

    Args:
    h (Tensor): Free surface height 'h' at time 'T'.
    u (Tensor): Horizontal velocity 'u' at time 'T'.
    v (Tensor): Vertical velocity 'v' at time 'T'.
    X (Tensor): Spatial grid in the x direction.
    Y (Tensor): Spatial grid in the y direction.
    T (float): Current time.
    g (float): Acceleration due to gravity.
    S (int): Target size in the X and Y dimensions.
    T_out (int): Target size in the T dimension.

    Returns:
    Tuple of Tensors: Left-hand side terms for continuity, u-momentum, and v-momentum equations.
    """
    # Use autograd to compute derivatives
    h_t = torch.autograd.grad(h.reshape(S * S * T_out, 1), T, torch.ones_like(h.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    u_t = torch.autograd.grad(u.reshape(S * S * T_out, 1), T, torch.ones_like(u.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    v_t = torch.autograd.grad(v.reshape(S * S * T_out, 1), T, torch.ones_like(v.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    h_x = torch.autograd.grad(h.reshape(S * S * T_out, 1), X, torch.ones_like(h.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    u_x = torch.autograd.grad(u.reshape(S * S * T_out, 1), X, torch.ones_like(u.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    v_x = torch.autograd.grad(v.reshape(S * S * T_out, 1), X, torch.ones_like(v.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    h_y = torch.autograd.grad(h.reshape(S * S * T_out, 1), Y, torch.ones_like(h.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    u_y = torch.autograd.grad(u.reshape(S * S * T_out, 1), Y, torch.ones_like(u.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)
    v_y = torch.autograd.grad(v.reshape(S * S * T_out, 1), Y, torch.ones_like(v.reshape(S * S * T_out, 1)),
                              create_graph=True)[0].reshape(1, S, S, T_out)

    # Compute the left-hand side term for the continuity equation
    continuity_lhs = h_t + (h * (u_x + v_y)) + (u * h_x + v * h_y)

    # Compute the left-hand side terms for the momentum equations
    u_momentum_lhs = u_t + (u * u_x + v * u_y) + (g * h_x)
    v_momentum_lhs = v_t + (u * v_x + v * v_y) + (g * h_y)

    # Detach and delete tensors to free up GPU memory
    del h_t, u_t, v_t, h_x, u_x, v_x, h_y, u_y, v_y
    return (1/3) * torch.sqrt(continuity_lhs.mean() ** 2 + u_momentum_lhs.mean() ** 2 + v_momentum_lhs.mean() ** 2)


# # function for calculation of Boundary condition loss (u, v = 0 ) @ boundaries
def bc_loss(h, u, v):
    """
    Calculate the sum of the means of the spatial boundaries of three 3D tensors (h, u, v).

    Args:
    h (Tensor): Input tensor representing free surface height 'h' of shape [n_batch, S, S, T].
    u (Tensor): Input tensor representing horizontal velocity 'u' of shape [n_batch, S, S, T].
    v (Tensor): Input tensor representing vertical velocity 'v' of shape [n_batch, S, S, T].

    Returns:
    Tensor: Sum of the means of the spatial boundaries for all three tensors.
    """

    # Extract the spatial boundaries and calculate the mean for each tensor
    top_mean_h = torch.mean(h[:, :, 0, :])
    bottom_mean_h = torch.mean(h[:, :, -1, :])
    left_mean_h = torch.mean(h[:, 0, :, :])
    right_mean_h = torch.mean(h[:, -1, :, :])

    top_mean_u = torch.mean(u[:, :, 0, :])
    bottom_mean_u = torch.mean(u[:, :, -1, :])
    left_mean_u = torch.mean(u[:, 0, :, :])
    right_mean_u = torch.mean(u[:, -1, :, :])

    top_mean_v = torch.mean(v[:, :, 0, :])
    bottom_mean_v = torch.mean(v[:, :, -1, :])
    left_mean_v = torch.mean(v[:, 0, :, :])
    right_mean_v = torch.mean(v[:, -1, :, :])

    # Calculate the sum of all means
    total_sum_of_means = (1/9) * torch.sqrt(top_mean_h ** 2 + bottom_mean_h ** 2 + left_mean_h ** 2 + right_mean_h ** 2 + \
                                    top_mean_u ** 2 + bottom_mean_u ** 2 + left_mean_u ** 2 + right_mean_u ** 2 + \
                                    top_mean_v ** 2 + bottom_mean_v ** 2 + left_mean_v ** 2 + right_mean_v ** 2)

    # Detach and delete tensors to free up GPU memory
    del top_mean_h, bottom_mean_h, left_mean_h, right_mean_h
    del top_mean_u, bottom_mean_u, left_mean_u, right_mean_u
    del top_mean_v, bottom_mean_v, left_mean_v, right_mean_v

    return total_sum_of_means
