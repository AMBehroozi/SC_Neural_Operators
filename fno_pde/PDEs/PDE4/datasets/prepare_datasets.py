# %%
import torch
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import sys
from torch.utils.data import TensorDataset, ConcatDataset
import sourcedefender
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse
import sourcedefender
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
    def __init__(self, dx, nx):
        super(BurgersODE, self).__init__()
        self.dx = dx
        self.nx = nx
    
    def forward(self, t, u, params):
        alpha, nu, A, omega = [p.unsqueeze(1) for p in params[:4]]
        
        # Implement periodic boundary conditions
        u_padded = torch.cat([u[:, -1:], u, u[:, :1]], dim=1)
        
        u_xx = (u_padded[:, 2:] - 2 * u_padded[:, 1:-1] + u_padded[:, :-2]) / (self.dx ** 2)
        u_x = (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * self.dx)
        
        x = torch.linspace(0, 1, self.nx, device=u.device)
        F = A * torch.sin( 2 * omega * torch.pi * t)
        
        dudt = 0.5 * nu * u_xx - alpha * u * u_x + F
        
        return dudt / torch.pi  # Scaling factor for numerical stability

def solve_burgers(model, params, initial_conditions, t, device):
    params = params.to(device)
    initial_conditions = initial_conditions.to(device)
    
    def func(t, y):
        return model(t, y, params)

    solution = odeint(func, initial_conditions, t, method='rk4')
    return solution.permute(1, 2, 0)  # Shape: [batch_size, nx, timesteps]

def create_initial_condition(x, B, device):
    # Gaussian pulse centered at the middle of the domain
    mean = 0.5  # Center of the pulse
    sigma = 0.3  # Width of the pulse
    return B.unsqueeze(1) * (torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + torch.sin(0.5 * torch.pi * x))

def solution(model, parameters, t, step_save, nx, device):
    x = torch.linspace(0, 1, nx, device=device)
    
    initial_params = parameters[:, -1]
    ode_params = parameters[:, :-1] # Transpose to separate parameters
    
    initial_conditions = create_initial_condition(x, initial_params, device)
    
    u_solutions = solve_burgers(model, ode_params.T, initial_conditions, t, device)
    jump = int(u_solutions.shape[2] / step_save)
    u_solutions = u_solutions[:, :, ::jump]

    nb, s0, nt = u_solutions.shape
    du_dP = batchJacobian_AD(u_solutions.reshape(nb, s0 * nt), ode_params, graphed=False)

    return u_solutions.detach(), du_dP.reshape(nb, s0, nt, ode_params.shape[1]).detach()

def create_and_save_dataset(t_end, steps_solve, step_save, device, params, dataset_segment_size, dx, nx):
    t_tensor = torch.linspace(0, t_end, steps_solve, device=device)
    model = BurgersODE(dx, nx).to(device)
    
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
    
    del u_tensor, grads_tensor, parameters_tensor
    del temp_list_u, temp_list_grads, temp_list_parameters

    return main_dataset

'''
Parametric Burgers' Equation:
∂u/∂t + α * u * ∂u/∂x = ν * ∂²u/∂x² + F(x,t)
Where:
u(x,t) is the dependent variable (e.g., velocity in fluid dynamics)
t is time
F(x,t) = A * sin(ω * t) is an optional forcing term

Parameters:
α: Advection coefficient (controls the nonlinear advection strength)
ν: Viscosity (controls the diffusion strength)
A: Amplitude of the forcing term
ω: Frequency of the time-dependent forcing

Initial Condition:
u(x, 0) = u₀(x) = B * (exp(-((x - mean)^2) / (2 * sigma^2)) + sin(0.5 * pi * x))
where B is the amplitude factor affecting both the Gaussian pulse and the sinusoidal term,
mean is the center of the Gaussian pulse at 0.5,
sigma is the width of the Gaussian pulse at 0.3.
This initial condition combines a Gaussian pulse centered at the middle of the domain
with a sinusoidal wave.

Boundary Conditions: Periodic, such that u(0, t) = u(L, t)
Spatial Domain: x ∈ [0, L]
Temporal Domain: t ∈ [0, T]

'''
L = 1.0
dx = 0.025
nx = int(L / dx)
t_end = torch.pi
steps_solve = 210
step_save = 30

MHPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')
PATH = "../"

Sample_number = 1300
train_size = 1000
eval_size = 150
test_size = Sample_number - train_size - eval_size
dataset_segment_size = 100

# Parameter ranges
alpha_range = (0.1, 1.0)   
nu_range = (0.05, 0.1) # gamma
A_range = (0.1, 0.5)  # delta
omega_range = (0.01, 0.1) 
B_range = (1.0, 1.0)  

params = torch.rand(Sample_number, 5, device=device)
params[:, 0] = params[:, 0] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
params[:, 1] = params[:, 1] * (nu_range[1] - nu_range[0]) + nu_range[0]
params[:, 2] = params[:, 2] * (A_range[1] - A_range[0]) + A_range[0]
params[:, 3] = params[:, 3] * (omega_range[1] - omega_range[0]) + omega_range[0]
params[:, 4] = params[:, 4] * (B_range[1] - B_range[0]) + B_range[0]

params.requires_grad_(True)

# Create and save the dataset
dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, params, dataset_segment_size, dx, nx)
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
torch.save(train_dataset, PATH + 'datasets/train_dataset.pt')
torch.save(eval_dataset, PATH + 'datasets/eval_dataset.pt')
torch.save(test_loader, PATH + 'datasets/test_loader.pt')

# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

# Path to the saved test dataset
PATH = "train_dataset.pt"

# Load the test dataset
test_loader = torch.load(PATH)
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
anim.save(f'reaction_diffusion_advection{sample_idx}.gif', writer='pillow')
# %%

for i in range(129):
    plt.plot(test_loader.dataset.tensors[1][i, :, -1, 0].cpu().detach().numpy())
# %%
for i in range(129):
    plt.plot(test_loader.dataset.tensors[2][i, :, -1, 3, 0].cpu().detach().numpy())

# %%
