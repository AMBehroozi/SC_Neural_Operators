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
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, plot_losses_from_excel, plot_realtime_metrics
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
dx = 0.05
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

state_size, parameter_size = 1, 6

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


idx = 3  # the index of the batch you want

for i, batch_data in enumerate(test_loader):
    if i == idx:
        batch_data_1 = [item.to(device) for item in batch_data]
        parameter_target, u_target, du_dparam_true = batch_data_1
        u_in_target, u_out_target = u_target[..., :T_in, :], u_target[..., T_in:, :]
        break

# Extract other parameters from 'parameter_target' but detach them to prevent gradient computation
c = parameter_target[:, 0:1].detach()
alpha = parameter_target[:, 1:2].detach()
beta = parameter_target[:, 2:3].detach()
gamma = parameter_target[:, 3:4].detach()
omega = parameter_target[:, 4:5].detach()

# Define mu range
mu_range = (0.00, 0.50)

# Initialize 'mu' uniformly within the specified range
# mu = (torch.rand(1, 1, device=device) * (mu_range[1] - mu_range[0]) + mu_range[0]).requires_grad_(True)
mu = torch.tensor([[0.25]], device=device, requires_grad=True)

# Function to update parameters tensor
def update_parameters():
    return torch.cat((c, alpha, beta, gamma, omega, mu), dim=-1)

# Setup model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = optim.AdamW([mu], lr=0.001, weight_decay=1e-5)  # Using AdamW with weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)
loss_fn = nn.MSELoss()

# Usage in the main loop:
iner_num_iterations = 1000
update_plot = plot_realtime_metrics(iner_num_iterations)

for i in range(iner_num_iterations):
    optimizer.zero_grad()
    
    # Update parameters tensor with current mu
    parameters = update_parameters()
    
    if Operator in ['FNO', 'WNO', 'MWNO']:
        t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(parameters.shape[0], 1).to(device)
        x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(parameters.shape[0], 1).to(device)
        U_pred = model(u_in_target, x_tensor_, t_tensor_, parameters)
    else:
        U_pred = model(u_in_target, parameters)

    loss = loss_fn(U_pred, u_out_target)
    
    loss.backward()
    
    if i % 5 == 0:  # Update plot every 100 iterations
        target_mu = parameter_target[0, 5].item()
        mu_diff = update_plot(i, loss.item(), mu.item(), target_mu)
    
    optimizer.step()
    scheduler.step(loss)
    
    # Early stopping condition
    if loss.item() < 1e-8:
        print(f"Converged at iteration {i}")
        break

# Final update to the plot
target_mu = parameter_target[0, 5].item()
mu_diff = update_plot(i, loss.item(), mu.item(), target_mu)
target_mu = parameter_target[0, 5].item()
mu_diff = mu.item() - target_mu
relative_error = abs(mu_diff / target_mu) * 100

print(f'Ultimate μ: {mu.item():.4f}, Target μ: {target_mu:.4f}, Prediction difference μ: {mu_diff:.4e}, Absolute relative error (%): {relative_error:.3f}')
# Turn off interactive mode
plt.ioff()
plt.show()


# %% 
# SC-FNO      : Ultimate μ: 0.1763, Target μ: 0.1746, Prediction difference μ: 1.7055e-03, Absolute relative error (%): 0.977
# FNO         : Ultimate μ: 0.1899, Target μ: 0.1746, Prediction difference μ: 1.5267e-02, Absolute relative error (%): 8.744
# SC-CNO      : Ultimate μ: 0.1814, Target μ: 0.1746, Prediction difference μ: 6.7681e-03, Absolute relative error (%): 3.877
# CNO         : Ultimate μ: 0.0387, Target μ: 0.1746, Prediction difference μ: -1.3593e-01, Absolute relative error (%): 77.860 (Failed!)
# SC-PNO      : Ultimate μ: 0.1750, Target μ: 0.1746, Prediction difference μ: 4.4492e-04, Absolute relative error (%): 0.255
# PNO         : Ultimate μ: 0.1814, Target μ: 0.1746, Prediction difference μ: 6.8028e-03, Absolute relative error (%): 3.896
# SC-DeepONet : Ultimate μ: 0.1648, Target μ: 0.1746, Prediction difference μ: -9.7791e-03, Absolute relative error (%): 5.601
# DeepONet    : Ultimate μ: 0.4583, Target μ: 0.1746, Prediction difference μ: 2.8367e-01, Absolute relative error (%): 162.479 (Failed!)
# SC-WNO      : Ultimate μ: 0.1724, Target μ: 0.1746, Prediction difference μ: -2.1408e-03, Absolute relative error (%): 1.226
# WNO         : Ultimate μ: 0.1910, Target μ: 0.1746, Prediction difference μ: 1.6408e-02, Absolute relative error (%): 9.398
