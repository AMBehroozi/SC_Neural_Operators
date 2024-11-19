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

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from lib.util import MHPI, calculate_rmse, calculate_r2, count_parameters
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, plot_losses_from_excel
from lib.DerivativeComputer import batchJacobian_AD

from models.Polynomial_Neural_Operator import PNO1DTime
from models.Convolutional_Neural_Operators2d import CNO1DTime
from models.DeepONet2d import DNO1DTime
from models.FNO_2d import FNO2d
from models.WNO_2d import WNO2d
from models.MultiWaveletConv_2d import MWNO2d

from PDE4.datasets.prepare_datasets import create_and_save_dataset
R2_state_list1 = []
R2_Grad_list1 = []
R2_state_list = []
R2_Grad_list = []
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Operator = 'FNO'    # PMO, CNO, DeepONet
enable_ig_loss = False
label = 'SC_' + Operator if enable_ig_loss else Operator

state_size, parameter_size = 1, 4

if (enable_ig_loss == False):
    Mode = Operator + "_Data"
    ss = 1
# Data + IG
if (enable_ig_loss == True):
    Mode = Operator + "_Data_IG" 
    ss = 1 + parameter_size

PATH = "../"

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
    model = FNO2d(modes1, modes2,  width, T_in, T_out, parameter_size=4, state_size=1).to(device)

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


model.load_state_dict(checkpoint['model_state_dict'])
print(f'{label}: {count_parameters(model)}')


# Parameter ranges
alpha_range = (0.1, 1.0)   
nu_range = (0.05, 0.1) 
A_range = (0.1, 0.5) 
omega_range = (0.01, 0.1) 
B_range = (1.0, 1.0)


pert = np.linspace(0, 1, 11) + 1.0

Sample_number = 150
for pert in pert:  
    alpha_range =    (1.0, 1.0 * pert)    # The Original Range: [0.0, 0.5]
    nu_range = (0.1, 0.1 * (1 + (pert-1) * 0.5)) 
    A_range = (0.5, 0.5 * pert) 
    omega_range = (0.1, 0.1 * pert) 

    # alpha_range =    (1.0 * pert, 1.0 * pert)    # The Original Range: [0.0, 0.5]
    # nu_range = (0.1 * (1 + (pert-1) * 0.5), 0.1 * (1 + (pert-1) * 0.5)) 
    # A_range = (0.5 * pert, 0.5 * pert) 
    # omega_range = (0.1 * pert, 0.1 * pert) 


    # Sample uniformly within each specified range.
    params = torch.rand(Sample_number, 5, device=device)
    params[:, 0] = params[:, 0] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
    params[:, 1] = params[:, 1] * (nu_range[1] - nu_range[0]) + nu_range[0]
    params[:, 2] = params[:, 2] * (A_range[1] - A_range[0]) + A_range[0]
    params[:, 3] = params[:, 3] * (omega_range[1] - omega_range[0]) + omega_range[0]
    params[:, 4] = params[:, 4] * (B_range[1] - B_range[0]) + B_range[0]

    params.requires_grad_(True)
                                          #t_end, steps_solve, step_save, device, params, dataset_segment_size, dx, nx-*
    test_dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, params, 75, dx, nx)
    test_loader = DataLoader(test_dataset, batch_size=2)

    U_pred_list, du_dp_list, batch_u_out_list, du_dparam_true_list = [], [], [], []
    model.eval()  # Set the model to evaluation mode.
        
    for batch_data in test_loader:
        batch_data_1 = [item.to(device) for item in batch_data]
        batch_parameters, batch_u, du_dparam_true = batch_data_1

        batch_u_in, batch_u_out = batch_u[..., :T_in, :], batch_u[..., T_in:, :]
        
        batch_size_ = batch_parameters.shape[0]

        batch_parameters.requires_grad_(True)        
        
        U_in = batch_u_in
        if Operator =='FNO' or Operator == 'WNO' or Operator == 'MWNO':
            t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
            x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(batch_size_, 1).to(device)
            U_pred = model(U_in, x_tensor_, t_tensor_, batch_parameters)
        else:
            U_pred = model(U_in, batch_parameters)

        du_dp = torch.zeros(batch_size_, nx, T_out, model.parameter_size, model.state_size).to(device)
        
        for i in range(model.state_size):
            state_tensor = U_pred[..., i]
            Jacobian = batchJacobian_AD(state_tensor.reshape(batch_size_, nx * T_out), batch_parameters, graphed=True, batchx=True)
            du_dp[..., i] = Jacobian.reshape(batch_size_, nx, T_out, model.parameter_size)

        torch.cuda.empty_cache()
        U_pred_list.append(U_pred.detach())    
        du_dp_list.append(du_dp[:, :, :, :, :].detach())
        batch_u_out_list.append(batch_u_out.detach())
        du_dparam_true_list.append(du_dparam_true)

    U_pred = torch.cat(U_pred_list, dim=0)
    du_dp = torch.cat(du_dp_list, dim=0)
    batch_u_out = torch.cat(batch_u_out_list, dim=0)
    du_dparam_true = torch.cat(du_dparam_true_list, dim=0)


    RMSE_Grads = torch.zeros(4)  # Assuming RMSE for each component, storing in a tensor
    R2_Grads = torch.zeros(4)  # Storing R² for each component

    for i in range(4):
        predictions = du_dp[:, :, :, i, 0].cpu().detach()
        true_values = du_dparam_true[:, :, T_in:, i, 0].cpu().detach()
        RMSE_Grads[i] = calculate_rmse(predictions, true_values)
        R2_Grads[i] = calculate_r2(predictions, true_values)


    RMSE_U = calculate_rmse(U_pred, batch_u_out)
    R2_U = calculate_r2(U_pred, batch_u_out)
    print(f'Mode:{label}, pert: {pert}, alpha_range: {alpha_range}, nu_range: {nu_range}, A_range: {A_range}, omega_range: {omega_range}')

    print(f'R2 state value :{R2_U:.5f}')
    for i in range(4):
        print(f'R2 du/dp{i+1}: {R2_Grads[i].item():.5f}')
    
    if enable_ig_loss:
        R2_state_list.append(R2_U)
        R2_Grad_list.append(R2_Grads)
    else:
        R2_state_list1.append(R2_U)
        R2_Grad_list1.append(R2_Grads)

# %%
R2_state_list_np1 = [tensor.cpu().numpy() for tensor in R2_state_list1]
R2_state_list_np = [tensor.cpu().numpy() for tensor in R2_state_list]


# Creating the x-axis values (index of the list)
pert = np.linspace(0, 1, 11) 

x_values = pert * 100

# Creating the plot
plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.plot(x_values, R2_state_list_np, marker='o', linestyle='-', color='red', label='SC-CNO')  # Line plot
plt.plot(x_values, R2_state_list_np1, marker='o', linestyle='-', color='b', label='CNO')  # Line plot

# Adding title and labels
plt.title('Plot of R² Values')
plt.xlabel('Index')
plt.ylabel('R² Value')
plt.legend()
# Adding grid for better readability (optional)
plt.grid(True)

# Display the plot
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# Assume R2_state_list and R2_state_list1 are lists of tensors on CUDA
# Convert tensors to NumPy arrays after moving them to CPU
R2_state_list_np = [tensor.cpu().numpy() for tensor in R2_state_list]
R2_state_list_np1 = [tensor.cpu().numpy() for tensor in R2_state_list1]

# Define parameter perturbation
pert = np.linspace(0, 1, 11) + 1.0

# Generate x-values (these could be dummy since we will replace them with labels)
x_values = np.arange(len(pert))

# Creating the plot
plt.figure(figsize=(20, 10))
plt.plot(x_values, R2_state_list_np, marker='o', linestyle='-', color='red', label='SC-FNO', markersize=10)
plt.plot(x_values, R2_state_list_np1, marker='o', linestyle='-', color='blue', label='FNO', markersize=10)

# Adding titles and labels
plt.xlabel('\nParameter Ranges', fontsize=16)
plt.ylabel('R² Value', fontsize=16)
plt.legend(fontsize=16, loc='upper right')

# Generate dynamic parameter labels based on pert
parameter_labels = []
for pert_val in pert:
    alpha_range = f'α: [1.00, {1.0 * pert_val:.2f}]'
    nu_range = f'ν: [0.10, {0.1 * (1 + (pert_val - 1) * 0.5):.2f}]'
    A_range = f'A: [0.50, {0.5 * pert_val:.2f}]'
    omega_range = f'ω: [0.10, {0.1 * pert_val:.2f}]'
    label = f'{alpha_range}\n{nu_range}\n{A_range}\n{omega_range}'
    parameter_labels.append(label)

plt.xticks(x_values, parameter_labels, rotation=0)  # Set custom x-axis tick labels without rotation

# Original parameter ranges annotation
original_params = f"Original Ranges:\nα: [0.10, 1.00]\nν: [0.05, 0.10]\nA: [0.10, 0.50]\nω: [0.01, 0.10]"
plt.figtext(0.15, 0.65, original_params, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', alpha=0.5))

# Adding grid for better readability
plt.grid(True)

# Adjust layout to make sure labels are not cut off
plt.gcf().subplots_adjust(bottom=0.25)

# Display the plot
plt.show()


# %%