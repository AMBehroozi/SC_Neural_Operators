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

from lib.util import MHPI, calculate_rmse, calculate_r2, scale_and_filter_tensor
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, plot_losses_from_excel
from lib.DerivativeComputer import batchJacobian_AD

from models.Polynomial_Neural_Operator import PNO1DTime
from models.Convolutional_Neural_Operators2d import CNO1DTime
from models.DeepONet2d import DNO1DTime
from models.FNO_2d import FNO2d
from models.WNO_2d import WNO2d
from models.MultiWaveletConv_2d import MWNO2d


from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset
from fno_pde.PDEs.PDE1.pde1 import skewed_gaussian_fixed_range
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 

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
plot_losses_from_excel(PATH + 'loss/' + f'losses_data_{Mode}_.xlsx', lable=Mode)
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

first_tensors = []

# Iterate through the DataLoader to gather the first tensor from each batch
for batch in testdataloder:
    # batch[0] should give the tensor you are interested in (assuming each batch returns tensors and optionally labels)
    first_tensor = batch[0]  # Assuming the first element is the tensor
    first_tensors.append(first_tensor)

# Concatenate all collected tensors along a new dimension (or an existing dimension, based on your needs)
parameters_original = torch.cat(first_tensors, dim=0)

# %%
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


model.load_state_dict(checkpoint['model_state_dict'])


U_pred_list, du_dp_list, batch_u_out_list, du_dparam_true_list = [], [], [], []
model.eval()  # Set the model to evaluation mode.


# %%

pert_list = np.linspace(0.1, 1, 11)
# %%
for pert in pert_list:  
    parameters = scale_and_filter_tensor(parameters_original, pert)
    print(f'original shape: {parameters_original.shape}, shifted shape: {parameters.shape}')
    # Sample uniformly within each specified range.
    parameters.requires_grad_(True)

    test_dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, parameters, 15, dx, nx)
    test_loader = DataLoader(test_dataset, batch_size=4)


    U_pred_list, du_dp_list, batch_u_out_list, du_dparam_true_list = [], [], [], []
    model.eval()  # Set the model to evaluation mode.
        
    for batch_data in test_loader:
        batch_data_1 = [item.to(device) for item in batch_data]
        batch_parameters, batch_u, du_dparam_true = batch_data_1
        # batch_parameters = batch_parameters[:, :5]
        batch_u_in, batch_u_out = batch_u[..., :T_in, :], batch_u[..., T_in:, :]
        
        batch_size_ = batch_parameters.shape[0]

        t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
        x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(batch_size_, 1).to(device)
        batch_parameters.requires_grad_(True)        
        
        U_in = batch_u_in
        U_pred = model(U_in, x_tensor_, t_tensor_, batch_parameters)
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

    RMSE_Grads = torch.zeros(6)  # Assuming RMSE for each component, storing in a tensor
    R2_Grads = torch.zeros(6)    # Storing R² for each component

    for i in range(6):
        predictions = du_dp[:, :, :, i, 0].cpu().detach()
        true_values = du_dparam_true[:, :, T_in:, i, 0].cpu().detach()
        RMSE_Grads[i] = calculate_rmse(predictions, true_values)
        R2_Grads[i] = calculate_r2(predictions, true_values)


    RMSE_U = calculate_rmse(U_pred, batch_u_out)
    R2_U = calculate_r2(U_pred, batch_u_out)
    print(f'Mode:{Mode}')
    print(f'{pert*100} percent Shifting')

    # for i in range(5):
    #     print(f'{RMSE_Grads[i].item():.5e}')

    print(f'R2 state value :{R2_U:.5f}')
    for i in range(6):
        print(f'R2 du/dp{i+1}: {R2_Grads[i].item():.5f}')

# %%
