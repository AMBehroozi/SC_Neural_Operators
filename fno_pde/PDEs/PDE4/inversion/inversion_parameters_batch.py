# %%
from IPython.display import display, clear_output
import numpy as np
import torch
import sys
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, dataset
import pandas as pd
from IPython import display

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from lib.util import MHPI, calculate_rmse, calculate_r2
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, plot_losses_from_excel, plot_realtime_metrics_batch
from lib.DerivativeComputer import batchJacobian_AD

from models.Polynomial_Neural_Operator import PNO1DTime
from models.Convolutional_Neural_Operators2d import CNO1DTime
from models.DeepONet2d import DNO1DTime
from models.FNO_2d import FNO2d
from models.WNO_2d import WNO2d
from models.MultiWaveletConv_2d import MWNO2d

import os
from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset
from fno_pde.PDEs.PDE1.pde1 import skewed_gaussian_fixed_range
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Operator = 'FNO'    # PMO, CNO, DeepONet
enable_ig_loss = True   
label = 'SC_' + Operator if enable_ig_loss else Operator

if (enable_ig_loss == False):
    Mode = Operator + "_Data"
    ss = 1
# Data + IG
if (enable_ig_loss == True):
    Mode = Operator + "_Data_IG" 
    ss = 1 + 6

PATH = "../"
# PATH = "fno_pde/PDEs/PDE1/"

plot_losses_from_excel(PATH + 'loss/' + f'losses_data_{Mode}_.xlsx', lable=Mode)
# %%

# %%
L = 1.0
dx = 0.025
nx = int(L / dx)
x = torch.linspace(0, L, nx, device=device)

t0 = 0.0
t_end = torch.pi
steps_solve = 210
step_save = 30
T_in = 5
T_out = step_save - T_in

testdataloder = torch.load(PATH +  'datasets/test_loader.pt')
dataset = testdataloder.dataset
test_loader = DataLoader(dataset, batch_size=1)

state_size, parameter_size = 1, 4

checkpoint = torch.load(PATH + 'saved_models/' + Mode + '_saved_model.pth')

if Operator == 'CNO':
    width = checkpoint['width']
    depth = checkpoint['depth']
    kernel_size = checkpoint['kernel_size']
    model = CNO1DTime(nx, T_in, T_out, state_size, parameter_size, width, depth, kernel_size).to(device)

elif Operator == 'PNO':
    poly_degree = checkpoint['poly_degree']
    width = checkpoint['width']
    depth = checkpoint['depth']
    model = PNO1DTime(nx, T_in, T_out, state_size, parameter_size, poly_degree, width, depth).to(device)

elif Operator =='DeepONet':
    branch_layers = checkpoint['branch_layers']
    trunk_layers = checkpoint['trunk_layers']
    model = DNO1DTime(nx, T_in, T_out, state_size, parameter_size, branch_layers, trunk_layers).to(device)

elif Operator =='FNO':
    modes1 = checkpoint['modes1']
    modes2 = checkpoint['modes2']
    width = checkpoint['width']
    model = FNO2d(modes1, modes2,  width, T_in, T_out, parameter_size=parameter_size, state_size=state_size).to(device)

elif Operator =='WNO':
    levels = checkpoint['levels']
    size = checkpoint['size']
    width = checkpoint['width']
    model = WNO2d(levels=levels, size=[nx, nx], width=width, T_in=T_in, T_out=T_out, state_size=1, parameter_size=parameter_size).to(device)

elif Operator =='MWNO':
    levels = checkpoint['levels']
    size = checkpoint['size']
    width = checkpoint['width']
    model = MWNO2d(levels=levels, size=[nx, nx], width=width, T_in=T_in, T_out=T_out, state_size=1, parameter_size=parameter_size).to(device)


# Load a batch of data
batch_size = 100  # Set your desired batch size
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch_data in test_loader:
    batch_data_1 = [item.to(device) for item in batch_data]
    parameter_target, u_target, du_dparam_true = batch_data_1
    u_in_target, u_out_target = u_target[..., :T_in, :], u_target[..., T_in:, :]
    break

print(f"u_in_target shape: {u_in_target.shape}")
print(f"u_out_target shape: {u_out_target.shape}")
print(f"parameter_target shape: {parameter_target.shape}")

# %%
# Extract parameters from 'parameter_target'
# Parameter ranges
# alpha_range = (0.1, 1.0)   
# nu_range = (0.05, 0.1) # gamma
# A_range = (0.1, 0.5)  # delta
# omega_range = (0.01, 0.1) 

alpha = parameter_target[:, 0:1].detach()        # alpha
gamma = parameter_target[:, 1:2].detach()    # gamma
mu_target = parameter_target[:, 2:3].detach()        # delta
omega = parameter_target[:, 3:4].detach()        # omega


# Initialize 'mu' for the batch
mu = torch.full_like(mu_target, 0.0, requires_grad=True)

def update_parameters():
    return torch.cat((alpha, gamma, mu, omega), dim=-1)



# mu_target = parameter_target[:, 0:1].detach()        # alpha
# gamma = parameter_target[:, 1:2].detach()    # gamma
# delta = parameter_target[:, 2:3].detach()        # delta
# omega = parameter_target[:, 3:4].detach()        # omega


# # Initialize 'mu' for the batch
# mu = torch.full_like(mu_target, 0.15, requires_grad=True)

# def update_parameters():
#     return torch.cat((mu, gamma, delta, omega), dim=-1)
# %%
# Setup model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.AdamW([mu], lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)
loss_fn = nn.MSELoss()

# %%
# Main loop
iner_num_iterations = 1000
update_plot = plot_realtime_metrics_batch(iner_num_iterations, batch_size)
loss_list = []
for i in range(iner_num_iterations):
    optimizer.zero_grad()
    parameters = update_parameters()
    if Operator in ['FNO', 'WNO', 'MWNO']:
        t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(batch_size, 1).to(device)
        x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(batch_size, 1).to(device)
        U_pred = model(u_in_target, x_tensor_, t_tensor_, parameters)
    else:
        U_pred = model(u_in_target, parameters)

    loss = loss_fn(U_pred, u_out_target)
    
    loss.backward()
    loss_list.append(loss)
    if i % 20 == 0:
        mu_diff = update_plot(i, loss.item(), mu, mu_target)
    
    optimizer.step()
    scheduler.step(loss)
    
    if loss.item() < 1e-8:
        print(f"Converged at iteration {i}")
        break

torch.save(loss_list, 'SC_FNO.pt')
# Final update to the plot
# Calculate average performance metrics
avg_absolute_difference = torch.mean(torch.abs(mu - mu_target)).item()
avg_relative_error = torch.mean(torch.abs((mu - mu_target) / mu_target)).item() * 100

# Print average results
print("\nBatch Average Results:")
print(f"Average Absolute Difference: {avg_absolute_difference:.4e}")
print(f"Average Relative Error (%): {avg_relative_error:.3f}")

# Calculate and print R-squared
y_mean = torch.mean(mu_target)
ss_tot = torch.sum((mu_target - y_mean) ** 2)
ss_res = torch.sum((mu_target - mu) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared.item():.4f}")

plt.ioff()
plt.show()

# %% 
# omega
    # SC-FNO
        # Batch Average Results:
        # Average Absolute Difference: 7.5875e-04
        # Average Relative Error (%): 1.385
        # R-squared: 0.9981
    # FNO
        # Batch Average Results:
        # Average Absolute Difference: 1.8554e-03
        # Average Relative Error (%): 4.421
        # R-squared: 0.9917

# delta
    # SC-FNO
        # Batch Average Results:
        # Average Absolute Difference: 2.5044e-03
        # Average Relative Error (%): 1.151
        # R-squared: 0.9991
    # FNO
        # Batch Average Results:
        # Average Absolute Difference: 1.3062e-02
        # Average Relative Error (%): 4.592
        # R-squared: 0.9682

# gamma
    # SC-FNO
        # Batch Average Results:
        # Average Absolute Difference: 4.3206e-04
        # Average Relative Error (%): 0.603
        # R-squared: 0.9988
    # FNO
        # Batch Average Results:
        # Average Absolute Difference: 1.4590e-03
        # Average Relative Error (%): 6.170
        # R-squared: 0.9238

# 