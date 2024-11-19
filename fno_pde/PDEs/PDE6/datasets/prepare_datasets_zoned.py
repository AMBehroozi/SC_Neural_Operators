# %%
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sourcedefender
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
import argparse
import os
from tqdm import tqdm
from numpy.fft import fft, ifft, fftfreq


import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from lib.util import MHPI
from lib.DerivativeComputer import batchJacobian_AD

'''
Parametric Allen-Cahn Equation:
∂u/∂t = ϵ(x)∂²u/∂x² + α(x)u - β(x)u³

Parameters:
ϵ(x): Diffusion coefficient (spatially varying)
α(x): Linear term coefficient (spatially varying)
β(x): Cubic term coefficient (spatially varying)

Initial Condition:
u(x,0) = A*tanh(kx)

Boundary Conditions: Periodic, such that u(0, t) = u(L, t)
Spatial Domain: x ∈ [0, 1]
Temporal Domain: t ∈ [0, T]
'''

import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from lib.DerivativeComputer import batchJacobian_AD

class AllenCahnODE(torch.nn.Module):
    def __init__(self, dx, nx, n_zones):
        super(AllenCahnODE, self).__init__()
        self.dx = dx
        self.nx = nx
        self.n_zones = n_zones
        self.points_per_zone = nx // n_zones
        
    def get_zoned_params(self, params):
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
        # Extract parameters
        epsilons = params[:self.n_zones]
        alphas = params[self.n_zones:2*self.n_zones]
        betas = params[2*self.n_zones:3*self.n_zones]
        
        # Get parameter values for each spatial point
        epsilon_full = self.get_zoned_params(epsilons.T)
        alpha_full = self.get_zoned_params(alphas.T)
        beta_full = self.get_zoned_params(betas.T)
        
        # Implement periodic boundary conditions
        u_padded = torch.cat([u[:, -1:], u, u[:, :1]], dim=1)
        u_xx = (u_padded[:, 2:] - 2 * u_padded[:, 1:-1] + u_padded[:, :-2]) / (self.dx ** 2)
        
        # Compute the full equation
        dudt = (epsilon_full * u_xx + 
                alpha_full * u - 
                beta_full * u * u * u)
        
        return dudt

def solve_allen_cahn(model, params, initial_conditions, t, device):
    params = params.to(device)
    initial_conditions = initial_conditions.to(device)
    
    def func(t, y):
        return model(t, y, params)

    solution = odeint(func, initial_conditions, t, method='rk4')
    return solution.permute(1, 2, 0)

def create_initial_condition(x, A, k, device):
    A = A.unsqueeze(1)
    k = k.unsqueeze(1)
    return A * torch.tanh(k * x)

def solution(model, parameters, t, step_save, nx, device):
    x = torch.linspace(0, 1, nx, device=device)
    
    # Get A and k parameters
    A = parameters[:, -2]
    k = parameters[:, -1]
    
    # Get ODE parameters
    ode_params = parameters[:, :-2]
    
    initial_conditions = create_initial_condition(x, A, k, device)
    
    u_solutions = solve_allen_cahn(model, ode_params.T, initial_conditions, t, device)
    jump = int(u_solutions.shape[2] / step_save)
    u_solutions = u_solutions[:, :, ::jump]

    nb, s0, nt = u_solutions.shape
    du_dP = batchJacobian_AD(u_solutions.reshape(nb, s0 * nt), parameters, graphed=False)

    return u_solutions.detach(), du_dP.reshape(nb, s0, nt, parameters.shape[1]).detach()

def create_and_save_dataset(t_end, steps_solve, step_save, device, params, dataset_segment_size, dx, nx, n_zones):
    t_tensor = torch.linspace(0, t_end, steps_solve, device=device)
    model = AllenCahnODE(dx, nx, n_zones).to(device)
    
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
    main_dataset = TensorDataset(parameters_tensor.detach(), u_tensor[:, ::1, ...], grads_tensor[:, ::1, ...])
    
    return main_dataset

# Configuration
n_zones = 20
L = 1.0
dx = 0.025
nx = int(L / dx)
t_end = 0.25
steps_solve = 510
step_save = 30

# Parameter ranges
epsilon_range = (0.01, 0.1)     # Diffusion coefficient
alpha_range = (0.01, 1.0)        # Linear term
beta_range = (0.01, 1.0)         # Cubic term
A_range = (0.1, 0.9)           # Initial condition amplitude
k_range = (5.0, 10.0)          # Initial condition wavenumber

# Dataset sizes
Sample_number = 1000
dataset_segment_size = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

# Create parameters tensor
params = torch.rand(Sample_number, 3*n_zones + 2, device=device)  # 3*n_zones for epsilon, alpha, beta + A, k

# Set parameter ranges
for i in range(n_zones):
    params[:, i] = params[:, i] * (epsilon_range[1] - epsilon_range[0]) + epsilon_range[0]
    params[:, n_zones+i] = params[:, n_zones+i] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
    params[:, 2*n_zones+i] = params[:, 2*n_zones+i] * (beta_range[1] - beta_range[0]) + beta_range[0]

params[:, -2] = params[:, -2] * (A_range[1] - A_range[0]) + A_range[0]  # A
params[:, -1] = params[:, -1] * (k_range[1] - k_range[0]) + k_range[0]  # k

params.requires_grad_(True)

# Create and save dataset
dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, params, 
                                dataset_segment_size, dx, nx, n_zones)


save_path = '/storage/work/amb10399/project/SC-Neural-Operator/fno_pde/PDEs/PDE6/datasets'
torch.save(dataset, f'{save_path}/main_dataset_zones_3_{n_zones}.pt')

print(f"Dataset created and saved with {n_zones} zones")
# %% 
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# import numpy as np

# # Path to the saved test dataset
# PATH = "/storage/work/amb10399/project/SC-Neural-Operator/fno_pde/PDEs/PDE5/datasets/main_dataset_zones_3_5.pt"

# # Load the test dataset
# test_loader = torch.load(PATH, weights_only=False)
# # %%

# def animate_solution(sample_solution):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     x = np.linspace(0, 1, sample_solution.shape[0])
#     line, = ax.plot(x, sample_solution[:, 0])
    
#     ax.set_ylim(sample_solution.min(), sample_solution.max())
#     ax.set_xlabel('Space')
#     ax.set_ylabel('u(x,t)')
#     ax.set_title('Solution Evolution Over Time')

#     def update(frame):
#         line.set_ydata(sample_solution[:, frame])
#         return line,

#     anim = FuncAnimation(fig, update, frames=sample_solution.shape[1], interval=400, blit=True)
#     plt.close(fig)
#     return anim
# ff = 4

# sample_idx = 10
# sample_solution = test_loader.tensors[1][:, :, :].cpu().detach().numpy()
# # sample_solution = test_loader.dataset.tensors[-1][sample_idx, :, :, ff, 0].cpu().detach().numpy()

# print(f'idx {sample_idx}: {test_loader.dataset.tensors[0][sample_idx]}')
# anim = animate_solution(sample_solution)
# anim.save(f'reaction_diffusion_advection_zone_2_{sample_idx}.gif', writer='pillow')
# # %%

# for i in range(129):
#     plt.plot(test_loader.dataset.tensors[1][i, :, -1, 0].cpu().detach().numpy())
# # %%
# for i in range(129):
#     plt.plot(test_loader.dataset.tensors[2][i, :, -1, 3, 0].cpu().detach().numpy())

# %%
