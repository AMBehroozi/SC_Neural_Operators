# %%
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sourcedefender
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
import argparse
import os
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from lib.util import MHPI
from lib.DerivativeComputer import batchJacobian_AD


class BurgersODE(torch.nn.Module):
    def __init__(self, dx, nx, n_zones):
        super(BurgersODE, self).__init__()
        self.dx = dx
        self.nx = nx
        self.n_zones = n_zones
        self.points_per_zone = nx // n_zones
        
    def get_zoned_params(self, params):
        """
        Expands zoned parameters to match spatial grid points.
        params shape: [batch_size, n_zones]
        returns: [batch_size, nx]
        """
        param_expanded = []
        for i in range(self.n_zones):
            zone_param = params[:, i:i+1].repeat(1, self.points_per_zone)
            param_expanded.append(zone_param)
        
        param_full = torch.cat(param_expanded, dim=1)
        if param_full.shape[1] < self.nx:
            remainder = self.nx - param_full.shape[1]
            param_full = torch.cat([param_full, param_full[:, -1:].repeat(1, remainder)], dim=1)
        return param_full
    
    def forward(self, t, u, params):
        # First n_zones parameters are alphas, next n_zones are As
        alphas = params[:self.n_zones]
        As = params[self.n_zones:2*self.n_zones]
        nu, omega = [p.unsqueeze(1) for p in params[2*self.n_zones:2*self.n_zones+2]]
        
        # Get alpha and A values for each spatial point
        alpha_full = self.get_zoned_params(alphas.T)  # [batch_size, nx]
        A_full = self.get_zoned_params(As.T)  # [batch_size, nx]
        
        # Implement periodic boundary conditions
        u_padded = torch.cat([u[:, -1:], u, u[:, :1]], dim=1)
        
        u_xx = (u_padded[:, 2:] - 2 * u_padded[:, 1:-1] + u_padded[:, :-2]) / (self.dx ** 2)
        u_x = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * self.dx)
        
        F = A_full * torch.sin(2 * omega * torch.pi * t)
        
        dudt = 0.5 * nu * u_xx - alpha_full * u * u_x + F
        
        return dudt / torch.pi

def solve_burgers(model, params, initial_conditions, t, device):
    params = params.to(device)
    initial_conditions = initial_conditions.to(device)
    
    def func(t, y):
        return model(t, y, params)

    solution = odeint(func, initial_conditions, t, method='rk4')
    return solution.permute(1, 2, 0)  # Shape: [batch_size, nx, timesteps]

def create_initial_condition(x, B, device):
    mean = 0.5
    sigma = 0.3
    return B.unsqueeze(1) * (torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + torch.sin(0.5 * torch.pi * x))



def solution(model, parameters, t, step_save, nx, device):
    x = torch.linspace(0, 1, nx, device=device)
    
    # Fix: get B parameter (it's the last parameter)
    initial_params = parameters[:, -1]  # B parameter is still the last one
    
    # Fix: get all parameters except B for ODE
    ode_params = parameters[:, :-1]  # All zone alphas, zone As, plus nu, omega
    
    initial_conditions = create_initial_condition(x, initial_params, device)
    
    u_solutions = solve_burgers(model, ode_params.T, initial_conditions, t, device)
    jump = int(u_solutions.shape[2] / step_save)
    u_solutions = u_solutions[:, :, ::jump]

    nb, s0, nt = u_solutions.shape
    du_dP = batchJacobian_AD(u_solutions.reshape(nb, s0 * nt), ode_params, graphed=False)

    return u_solutions.detach(), du_dP.reshape(nb, s0, nt, ode_params.shape[1]).detach()

def create_and_save_dataset(t_end, steps_solve, step_save, device, params, dataset_segment_size, dx, nx, n_zones):
    t_tensor = torch.linspace(0, t_end, steps_solve, device=device)
    model = BurgersODE(dx, nx, n_zones).to(device)
    
    temp_list_u, temp_list_grads, temp_list_parameters = [], [], []
    outer_loop = tqdm(range(len(params) // dataset_segment_size), desc="Progress", position=0)
    
    for i in outer_loop:
        batch_params = params[i * dataset_segment_size:(i + 1) * dataset_segment_size].to(device)
        
        text = f'Preparing {i}-th dataset segment...'
        batch_solutions, du_d_P = solution(model, batch_params, t_tensor, step_save, nx, device)
        temp_list_u.append(batch_solutions.unsqueeze(-1))
        temp_list_grads.append(du_d_P.unsqueeze(-1))
        temp_list_parameters.append(batch_params)

        del batch_params, batch_solutions, du_d_P
        torch.cuda.empty_cache()
        outer_loop.set_description(text)
        
    parameters_tensor = torch.cat(temp_list_parameters, dim=0)
    u_tensor = torch.cat(temp_list_u, dim=0)
    grads_tensor = torch.cat(temp_list_grads, dim=0)
    main_dataset = TensorDataset(parameters_tensor[..., :-1].detach(), u_tensor[:, ::1, ...], grads_tensor[:, ::1, ...])
    
    return main_dataset

'''
Parametric Burgers' Equation:
∂u/∂t + α(x) * u * ∂u/∂x = ν * ∂²u/∂x² + A(x) * sin(ω * t)
Where:
u(x,t) is the dependent variable (e.g., velocity in fluid dynamics)
t is time
α(x): Spatially varying advection coefficient
A(x): Spatially varying forcing amplitude

Parameters:
α(x): Advection coefficient for each zone (controls the nonlinear advection strength)
ν: Viscosity (controls the diffusion strength)
A(x): Amplitude of the forcing term for each zone
ω: Frequency of the time-dependent forcing

Initial Condition:
u(x, 0) = u₀(x) = B * (exp(-((x - mean)^2) / (2 * sigma^2)) + sin(0.5 * pi * x))
where B is the amplitude factor affecting both the Gaussian pulse and the sinusoidal term,
mean is the center of the Gaussian pulse at 0.5,
sigma is the width of the Gaussian pulse at 0.3.

Boundary Conditions: Periodic, such that u(0, t) = u(L, t)
Spatial Domain: x ∈ [0, L]
Temporal Domain: t ∈ [0, T]
'''

n_zones = 40
L = 1.0
dx = 0.0125
nx = int(L / dx)
t_end = torch.pi
steps_solve = 540
step_save = 30

# Parameter ranges
alpha_range = (0.1, 1.0)   
nu_range = (0.05, 0.1)
A_range = (0.1, 0.5)
omega_range = (0.01, 0.1)
B_range = (1.0, 1.0)

# Dataset sizes
Sample_number = 1300
train_size = 1000
eval_size = 150
test_size = Sample_number - train_size - eval_size
dataset_segment_size = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

# Create parameters tensor with n_zones alphas + n_zones As + other parameters
params = torch.rand(Sample_number, 2*n_zones + 3, device=device)  # 2*n_zones for alpha and A + nu, omega, B

# Set parameter ranges for alphas (first n_zones parameters)
for i in range(n_zones):
    params[:, i] = params[:, i] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]

# Set parameter ranges for As (second n_zones parameters)
for i in range(n_zones):
    params[:, n_zones+i] = params[:, n_zones+i] * (A_range[1] - A_range[0]) + A_range[0]

# Set other parameter ranges
params[:, 2*n_zones] = params[:, 2*n_zones] * (nu_range[1] - nu_range[0]) + nu_range[0]  # nu
params[:, 2*n_zones+1] = params[:, 2*n_zones+1] * (omega_range[1] - omega_range[0]) + omega_range[0]  # omega
params[:, 2*n_zones+2] = params[:, 2*n_zones+2] * (B_range[1] - B_range[0]) + B_range[0]  # B

params.requires_grad_(True)

# Create and save dataset
dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, params, 
                                dataset_segment_size, dx, nx, n_zones)

# Split and save datasets
train_dataset, eval_dataset, test_dataset = random_split(dataset, 
                                                        [train_size, eval_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

torch.save(train_dataset, '/storage/work/amb10399/project/SC-Neural-Operator/fno_pde/PDEs/PDE4/datasets' + f'/train_dataset_zones_2_{n_zones}.pt')
torch.save(eval_dataset, '/storage/work/amb10399/project/SC-Neural-Operator/fno_pde/PDEs/PDE4/datasets' + f'/eval_dataset_zones_2_{n_zones}.pt')
torch.save(test_loader, '/storage/work/amb10399/project/SC-Neural-Operator/fno_pde/PDEs/PDE4/datasets' + f'/test_loader_zones_2_{n_zones}.pt')

print(f"Dataset created and saved with {n_zones} zones")

# %% 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

# Path to the saved test dataset
PATH = "train_dataset_zones_2_40.pt"

# Load the test dataset
test_loader = torch.load(PATH, weights_only=False)
# %%

def animate_solution(sample_solution):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, sample_solution.shape[0])
    line, = ax.plot(x, sample_solution[:, 0])
    
    ax.set_ylim(sample_solution.min(), sample_solution.max())
    ax.set_xlabel('Space')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution Evolution Over Time')

    def update(frame):
        line.set_ydata(sample_solution[:, frame])
        return line,

    anim = FuncAnimation(fig, update, frames=sample_solution.shape[1], interval=400, blit=True)
    plt.close(fig)
    return anim
ff = 4

sample_idx = 20
sample_solution = test_loader.dataset.tensors[1][sample_idx, :, :, 0].cpu().detach().numpy()
# sample_solution = test_loader.dataset.tensors[-1][sample_idx, :, :, ff, 0].cpu().detach().numpy()

print(f'idx {sample_idx}: {test_loader.dataset.tensors[0][sample_idx]}')
anim = animate_solution(sample_solution)
anim.save(f'reaction_diffusion_advection_zone_2_{sample_idx}.gif', writer='pillow')
# %%

for i in range(129):
    plt.plot(test_loader.dataset.tensors[1][i, :, -1, 0].cpu().detach().numpy())
# %%
for i in range(129):
    plt.plot(test_loader.dataset.tensors[2][i, :, -1, 3, 0].cpu().detach().numpy())

# # # %%
