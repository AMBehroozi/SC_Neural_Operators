# %%
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys

# %%
from torch.utils.data import TensorDataset, ConcatDataset
import sourcedefender
sys.path.append("../")
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../../")

from lib.DerivativeComputer import batchJacobian_AD

# %%
def skewed_gaussian_fixed_range(x, mu, sigma, a, b, device):
    """
    Calculate a batch of Gaussian-like functions with fixed range [a, b] using PyTorch tensors.
    
    Parameters:
    - x: Tensor of x values, shape (nx,), placed on the specified device
    - mu: Tensor of means, shape (nb, 1), placed on the specified device
    - sigma: Standard deviation of the Gaussians, scalar
    - a: Lower bound of the output range, scalar
    - b: Upper bound of the output range, scalar
    - device: The device (CPU or GPU) on which the computation is to be performed
    
    Returns:
    - scaled_gaussians: Tensor of scaled Gaussian functions, shape (nb, nx)
    """
    # Ensure inputs are on the correct device
    x = x.to(device)
    mu = mu.to(device)

    # Calculate the Gaussian function for each mean in mu
    # x.shape -> (nx,), mu.shape -> (nb, 1)
    # Using unsqueeze to broadcast x over mu
    gaussian = torch.exp(-((x.unsqueeze(0) - mu)**2) / (2 * sigma**2))
    
    # Directly scale the Gaussian to the range [a, b]
    scaled_gaussian = gaussian * (b - a) + a
    
    return scaled_gaussian

class WaveODE(nn.Module):
    def __init__(self, dx, nx):
        super(WaveODE, self).__init__()
        self.dx = dx
        self.nx = nx

    def forward(self, t, y, params, initial_params):
        c, alpha, beta, gamma, omega = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4], params[:, 4:5]
        mu = initial_params[:, 0:1]
        u = y[..., :self.nx]  # extract u for each batch
        v = y[..., self.nx:]  # extract v for each batch
        u_xx = torch.zeros_like(u)
        u_xx[..., 1:-1] = (u[..., :-2] - 2 * u[..., 1:-1] + u[..., 2:]) / self.dx**2
        dvdt = c * u_xx + alpha * v + beta * u + gamma * torch.sin(omega * u)
        
        # Add a small term that depends on the initial condition parameter
        dvdt = dvdt + 1e-6 * torch.sin(mu * u)
        
        return torch.cat([v, dvdt], dim=-1)  # concatenate du/dt and dv/dt along feature dimension

def solve_wave_equation(model, params, initial_conditions, initial_params, t, device):
    params = params.to(device)
    initial_conditions = initial_conditions.to(device)
    initial_params = initial_params.to(device)
    nx = int(initial_conditions.shape[1] / 2)
    
    def func(t, y):
        return model(t, y, params, initial_params)

    solution = odeint(func, initial_conditions, t, method='rk4')
    solution = solution.permute(1, 0, 2)  # Shape: [batch_size, timesteps, nx * 2]
    u_sol = solution[..., :model.nx].permute(0, 2, 1)  # Extract u and adjust shape
    return u_sol  # shape [nb, nx, nt]

def solution(model, parameters, t, step_save, device):
    L = 1.0
    dx = 0.05
    nx = int(L / dx)
    x = torch.linspace(0, L, nx, device=device)
    
    # Separate the initial condition parameters
    initial_params = parameters[:, -1:].clone()
    ode_params = parameters[:, :-1].clone()
    
    mu = initial_params
    u0 = skewed_gaussian_fixed_range(x, mu=mu, sigma=0.2, a=0, b=1.0, device=device)
    v0 = torch.zeros_like(u0)
    initial_conditions = torch.cat([u0, v0], dim=-1)
    initial_conditions.requires_grad_(True)
    
    u_solutions = solve_wave_equation(model, ode_params, initial_conditions, initial_params, t, device)
    jump = int(u_solutions.shape[2] / step_save)
    u_solutions = u_solutions[:, :, ::jump]
    
    nx = parameters.shape[-1]
    nb, s0, nt = u_solutions.shape[0], u_solutions.shape[1], u_solutions.shape[2]
    du_dP1 = batchJacobian_AD(u_solutions.reshape(nb, s0 * nt), parameters, graphed=False)

    return u_solutions.detach(), du_dP1.reshape(nb, s0, nt, nx).detach()

def create_and_save_dataset(t_end, steps_solve, step_save, device, alfa, dataset_segment_size, dx, nx):
    t_tensor = torch.linspace(0, t_end, steps_solve, device=device)
    model = WaveODE(dx, nx).to(device)
    # Creating dataset
    temp_list_u, temp_list_grads, temp_list_parameters = [], [], []
    outer_loop = tqdm(range(len(alfa) // dataset_segment_size), desc="Progress", position=0)
    for i in outer_loop:
        
        batch_alfa = alfa[i * dataset_segment_size:(i + 1) * dataset_segment_size].to(device)

        text = (f'Preparing {i}-th dataset segment...')
        batch_solutions, du_d_P = solution(model, batch_alfa, t_tensor, step_save, device)
        temp_list_u.append(batch_solutions.unsqueeze(-1))
        temp_list_grads.append(du_d_P.unsqueeze(-1))
        temp_list_parameters.append(batch_alfa)

        # Delete individual tensors and Clear GPU cache
        del batch_alfa, batch_solutions, du_d_P
        torch.cuda.empty_cache()
        outer_loop.set_description(text)
        
    parameters_tensor = torch.cat(temp_list_parameters, dim=0)
    u_tensor = torch.cat(temp_list_u, dim=0)
    grads_tensor = torch.cat(temp_list_grads, dim=0)
    main_dataset = TensorDataset(parameters_tensor.detach(), u_tensor, grads_tensor)
    
    del u_tensor, grads_tensor, parameters_tensor
    del temp_list_u, temp_list_grads, temp_list_parameters

    return main_dataset


def compute_fdm_derivatives(u, x, t):
    """
    Compute spatial and temporal derivatives for tensor u using Finite Difference Method (FDM),
    with torch.roll to handle wrapping and boundaries with three-point schemes.
    
    Args:
    - u (torch.Tensor): Tensor of shape [nbatch, nx, nt], representing the wave amplitude.
    - x (torch.Tensor): Spatial positions, assumed to be uniformly spaced.
    - t (torch.Tensor): Temporal positions, assumed to be uniformly spaced.
    
    Returns:
    - u_x (torch.Tensor): Approximated spatial derivative of u.
    - u_t (torch.Tensor): Approximated first temporal derivative of u.
    - u_tt (torch.Tensor): Approximated second temporal derivative of u.
    """
    # Assuming uniform spacing
    dx = x[0, 1] - x[0, 0]
    dt = t[0, 1] - t[0, 0]

    # Spatial derivatives
    u_x = torch.zeros_like(u)
    u_x[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dx)
    u_x[:, 0, :] = (-3 * u[:, 0, :] + 4 * u[:, 1, :] - u[:, 2, :]) / (2 * dx)
    u_x[:, -1, :] = (3 * u[:, -1, :] - 4 * u[:, -2, :] + u[:, -3, :]) / (2 * dx)

    # Temporal derivatives using roll
    rolled_right_t = torch.roll(u, shifts=-1, dims=2)
    rolled_left_t = torch.roll(u, shifts=1, dims=2)
    u_t = (rolled_right_t - rolled_left_t) / (2 * dt)
    # Apply boundary corrections
    u_t[:, :, 0] = (-3 * u[:, :, 0] + 4 * u[:, :, 1] - u[:, :, 2]) / (2 * dt)
    u_t[:, :, -1] = (3 * u[:, :, -1] - 4 * u[:, :, -2] + u[:, :, -3]) / (2 * dt)

    # Second temporal derivatives
    u_tt = (rolled_right_t - 2 * u + rolled_left_t) / (dt ** 2)
    # Boundary corrections
    u_tt[:, :, 0] = (2 * u[:, :, 0] - 5 * u[:, :, 1] + 4 * u[:, :, 2] - u[:, :, 3]) / (dt ** 2)
    u_tt[:, :, -1] = (2 * u[:, :, -1] - 5 * u[:, :, -2] + 4 * u[:, :, -3] - u[:, :, -4]) / (dt ** 2)

    return u_x, u_t, u_tt


def compute_residual(u, params, t, x):
    """
    Compute the residual of the extended parametric wave equation.

    Args:
    - u (torch.Tensor): Tensor of the wave amplitude, shape [nbatch, nx, nt, 1].
    - params (torch.Tensor): Parameters [c, alpha, beta, gamma, omega], shape [nbatch, 5].
    - t
    Returns:
    - residual (torch.Tensor): The computed residual, same shape as u.
    """
    nbatch, nx, nt = u.shape[0], u.shape[1], u.shape[2]
    c = params[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(u)
    alpha = params[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(u)
    beta = params[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(u)
    gamma = params[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(u)
    omega = params[:, 4].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(u)
    u_x, u_t, u_tt = compute_fdm_derivatives(u.squeeze(-1), x, t)  
    # Compute the right-hand side of the equation
    rhs = c**2 * u_x.unsqueeze(-1) + alpha * u_t.unsqueeze(-1) + beta * u + gamma * torch.sin(omega * u)

    # Compute the residual: difference between the second time derivative and rhs
    residual = u_tt.unsqueeze(-1) - rhs

    return residual