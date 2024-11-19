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

from lib.util import MHPI, calculate_rmse, calculate_r2
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, plot_losses_from_excel
from lib.DerivativeComputer import batchJacobian_AD

from models.Polynomial_Neural_Operator import PNO1DTime
from models.Convolutional_Neural_Operators2d import CNO1DTime
from models.DeepONet2d import DNO1DTime

from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset
from fno_pde.PDEs.PDE1.pde1 import skewed_gaussian_fixed_range
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lib.util import calculate_rmse, calculate_r2
from lib.DerivativeComputer import batchJacobian_AD
from models.Polynomial_Neural_Operator import PNO1DTime
from models.Convolutional_Neural_Operators2d import CNO1DTime
from models.DeepONet2d import DNO1DTime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(Operator, enable_ig_loss):
    Mode = f"{Operator}_Data{'_IG' if enable_ig_loss else ''}"
    PATH = "../"
    
    checkpoint = torch.load(PATH + 'saved_models/' + Mode + '_saved_model.pth')
    
    if Operator == 'CNO':
        model = CNO1DTime(nx, T_in, T_out, state_size, parameter_size, 
                          checkpoint['width'], checkpoint['depth'], checkpoint['kernel_size']).to(device)
    elif Operator == 'PNO':
        model = PNO1DTime(nx, T_in, T_out, state_size, parameter_size, 
                          checkpoint['poly_degree'], checkpoint['width'], checkpoint['depth']).to(device)
    elif Operator == 'DeepONet':
        model = DNO1DTime(nx, T_in, T_out, state_size, parameter_size, 
                          checkpoint['branch_layers'], checkpoint['trunk_layers']).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, Mode

def evaluate_model(model, test_loader):
    U_pred_list, du_dp_list, batch_u_out_list, du_dparam_true_list = [], [], [], []
    
    for batch_data in test_loader:
        batch_data_1 = [item.to(device) for item in batch_data]
        batch_parameters, batch_u, du_dparam_true = batch_data_1

        batch_u_in, batch_u_out = batch_u[..., :T_in, :], batch_u[..., T_in:, :]
        batch_size_ = batch_parameters.shape[0]

        batch_parameters.requires_grad_(True)        
        
        U_in = batch_u_in
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
    
    return U_pred, du_dp, batch_u_out, du_dparam_true

# Plot comparisons
def plot_comparison(U_pred_list, du_dp_list, Mode_list, batch_u_out_list, du_dparam_true_list, t_tensor_, sample_idx):
    n_models = len(Mode_list)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_models))
    
    # Plot U comparison
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor_.cpu().numpy(), batch_u_out_list[0][sample_idx, 5, :, 0].cpu().detach().numpy(), label='True', color='black', linestyle='--', marker='o')
    for i, (U_pred, Mode) in enumerate(zip(U_pred_list, Mode_list)):
        plt.plot(t_tensor_.cpu().numpy(), U_pred[sample_idx, 5, :, 0].cpu().detach().numpy(), label=Mode, color=colors[i])
    plt.title('Comparison of Predicted and True U')
    plt.xlabel('Time')
    plt.ylabel('U')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot derivatives
    labels_P = [r'$\frac{\partial u}{\partial c}$', r'$\frac{\partial u}{\partial \alpha}$', 
                r'$\frac{\partial u}{\partial \beta}$', r'$\frac{\partial u}{\partial \gamma}$',
                r'$\frac{\partial u}{\partial \omega}$', r'$\frac{\partial u}{\partial \mu}$']
    
    fig, axs = plt.subplots(6, 1, figsize=(12, 36))
    for i in range(6):
        axs[i].plot(t_tensor_.cpu().numpy(), du_dparam_true_list[0][sample_idx, 10, T_in:, i, 0].cpu().detach().numpy(), 
                    label='True Derivative', color='black', marker='o', markersize=3, linestyle='None')
        for j, (du_dp, Mode) in enumerate(zip(du_dp_list, Mode_list)):
            axs[i].plot(t_tensor_.cpu().numpy(), du_dp[sample_idx, 10, :, i, 0].cpu().detach().numpy(), 
                        label=Mode, color=colors[j], linestyle='-')
        axs[i].set_xlabel('Time', fontsize=14)
        axs[i].set_ylabel(labels_P[i], fontsize=20)
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.show()

# Global variables

L, dx = 1.0, 0.05
nx = int(L / dx)
t0, t_end = 0.0, torch.pi
steps_solve, step_save = 210, 30
T_in = 5
T_out = step_save - T_in
state_size, parameter_size = 1, 6

# Load test data
PATH = "../"
testdataloder = torch.load(PATH + 'datasets/test_loader.pt')
dataset = testdataloder.dataset
test_loader = DataLoader(dataset, batch_size=4)

# Main execution
U_pred_list, du_dp_list, Mode_list, batch_u_out_list, du_dparam_true_list = [], [], [], [], []

for Operator in ['PNO', 'CNO', 'DeepONet']:
    for enable_ig_loss in [False]:
        model, Mode = load_model(Operator, enable_ig_loss)
        U_pred, du_dp, batch_u_out, du_dparam_true = evaluate_model(model, test_loader)
        U_pred_list.append(U_pred)
        du_dp_list.append(du_dp)
        Mode_list.append(Mode) 
        batch_u_out_list.append(batch_u_out)
        du_dparam_true_list.append(du_dparam_true)
       
        # Calculate and print metrics
        RMSE_U = calculate_rmse(U_pred, batch_u_out)
        R2_U = calculate_r2(U_pred, batch_u_out)
        print(f'Mode: {Mode}')
        print(f'L2 state value: {R2_U:.5f}')
        for i in range(6):
            R2_Grad = calculate_r2(du_dp[:, :, :, i, 0], du_dparam_true[:, :, T_in:, i, 0])
            print(f'L2 du/dp{i+1}: {R2_Grad:.5f}')
        print()
# %%
t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].to(device)
sample_idx = 0
# Call the plotting function
plot_comparison(U_pred_list, du_dp_list, Mode_list, batch_u_out_list, du_dparam_true_list, t_tensor_, sample_idx)

# %%