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
from models.FNO_2d import FNO2d
from models.WNO_2d import WNO2d
from models.MultiWaveletConv_2d import MWNO2d


from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset
from fno_pde.PDEs.PDE1.pde1 import skewed_gaussian_fixed_range
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Operator = 'FNO'    # PMO, CNO, DeepONet
enable_ig_loss = False
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


model.load_state_dict(checkpoint['model_state_dict'])


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


RMSE_Grads = torch.zeros(6)  # Assuming RMSE for each component, storing in a tensor
R2_Grads = torch.zeros(6)  # Storing R² for each component

for i in range(6):
    predictions = du_dp[:, :, :, i, 0].cpu().detach()
    true_values = du_dparam_true[:, :, T_in:, i, 0].cpu().detach()
    RMSE_Grads[i] = calculate_rmse(predictions, true_values)
    R2_Grads[i] = calculate_r2(predictions, true_values)


RMSE_U = calculate_rmse(U_pred, batch_u_out)
R2_U = calculate_r2(U_pred, batch_u_out)
print(f'Mode:{label}')


print(f'R2 state value :{R2_U:.5f}')
for i in range(6):
    print(f'R2 du/dp{i+1}: {R2_Grads[i].item():.5f}')

# In[15]:
sample_idx = 64  
t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].to(device)
data_true = batch_u_out[sample_idx, 5, :, 0].cpu().detach().numpy()
data1 = U_pred[sample_idx, 5, :, 0].cpu().detach().numpy()

plt.figure(figsize=(6, 3))
plt.plot(t_tensor_.cpu().numpy(), data_true, label='True', linestyle='--', marker='o')

plt.plot(t_tensor_.cpu().numpy(), data1, label=label)

plt.title('Comparison of Predicted and True U')
plt.xlabel('Time')
plt.ylabel('U')
plt.legend()
plt.grid(True)
plt.show()


labels_P = [r'$\frac{\partial u}{\partial c}$', r'$\frac{\partial u}{\partial \alpha}$', 
            r'$\frac{\partial u}{\partial \beta}$', r'$\frac{\partial u}{\partial \gamma}$',
            r'$\frac{\partial u}{\partial \omega}$', r'$\frac{\partial u}{\partial \mu}$']
fig, axs = plt.subplots(6 * 1, 1, figsize=(6, 18))
RMSE_Grads = torch.zeros(1, 6)
for i in range(6):
    for j in range(1):
        idx = i * 1 + j
        axs[idx].plot(t_tensor_.cpu().numpy(), du_dparam_true[sample_idx, 10, T_in:, i, j].cpu().detach().numpy(), label=f'True Drivetive', color='black', marker='o', markersize=3, linestyle='None')

        axs[idx].plot(t_tensor_.cpu().numpy(), du_dp[sample_idx, 10, :, i, j].cpu().detach().numpy(), label=label, linestyle='-')
        axs[idx].set_xlabel('Time', fontsize = 14)
        axs[idx].set_ylabel(labels_P[i], fontsize = 20)
        axs[idx].grid(True)
        axs[idx].legend()

plt.tight_layout()
plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np

# Define the modes and corresponding R2 values
modes = [
    "MWNO", "SC_MWNO", "WNO", "SC_WNO", "PNO", "SC_PNO",
    "CNO", "SC_CNO", "DeepONet", "SC_DeepONet", "FNO", "SC_FNO"
]
state_values = [
    0.97795, 0.95178, 0.98135, 0.98992, 0.97681, 0.99550,
    0.97689, 0.99542, 0.97437, 0.95453, 0.99421, 0.99122
]

# Determine colors for bars (red for 'SC_' prefixed modes, 'skyblue' otherwise)
colors = ['red' if 'SC_' in mode else 'skyblue' for mode in modes]

# Calculate ranks based on state values
ranks = np.argsort(np.argsort(-np.array(state_values))) + 1  # Get ranks, +1 to start from 1

# Determine bar widths and positions to close gaps between related models
bar_width = 0.6
adjusted_bar_positions = np.array([2*i for i in range(len(modes)//2)])

# Create a bar chart for the R2 state values of each adjusted mode
plt.figure(figsize=(14, 8))
bars = plt.bar(adjusted_bar_positions - bar_width/2, state_values[::2], width=bar_width, color='skyblue', label='Original Mode')
sc_bars = plt.bar(adjusted_bar_positions + bar_width/2, state_values[1::2], width=bar_width, color='red', label='Sensitivity Constrained')

plt.xlabel('Model', fontsize=12)
plt.ylabel('R² State Value', fontsize=12)
plt.title('Comparison of R2 State Values Across Models')

# Adjust x-tick labels to include "SC-" where applicable
xtick_labels = [mode if 'SC_' not in mode else "SC-" + mode.replace("SC_", "") for mode in modes[::2]]
plt.xticks(adjusted_bar_positions, xtick_labels, rotation=45, ha="right")

plt.ylim(0.9, 1.0)  # Set y-axis limit to focus on R2 values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with the corresponding rank
for bar, sc_bar, rank, sc_rank in zip(bars, sc_bars, ranks[::2], ranks[1::2]):
    plt.text(bar.get_x() + bar.get_width()/2, 0.92, f'Rank {rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)
    plt.text(sc_bar.get_x() + sc_bar.get_width()/2, 0.92, f'Rank {sc_rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)

# Add a legend to the plot
plt.legend(fontsize=12)

plt.show()

# %%
# Define the R2 values for du/dp1 to du/dp6 for each mode
du_dp_values = {
    'du/dp1': [0.59612, 0.93969, 0.66183, 0.98960, 0.53076, 0.98714, 0.72397, 0.99177, 0.11720, 0.50881, 0.77799, 0.98807],
    'du/dp2': [0.09014, 0.95632, 0.40215, 0.98333, 0.23733, 0.95160, 0.21714, 0.99013, 0.11031, 0.24364, 0.26960, 0.99172],
    'du/dp3': [0.86299, 0.98532, 0.92121, 0.99426, 0.76861, 0.99341, 0.90463, 0.99641, 0.77892, 0.82700, 0.95005, 0.99655],
    'du/dp4': [0.53083, 0.65921, 0.57206, 0.83457, 0.44405, 0.93109, 0.50436, 0.98389, 0.48335, 0.57251, 0.60261, 0.85109],
    'du/dp5': [0.55810, 0.68484, 0.61420, 0.82382, 0.38898, 0.93214, 0.59012, 0.97398, 0.49730, 0.53104, 0.66059, 0.79912],
    'du/dp6': [0.32990, 0.97910, 0.72723, 0.99369, 0.78732, 0.98958, -0.26047, 0.99427, -0.43031, 0.78301, -0.07204, 0.99575]
}

# Create a figure and a set of subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

# Plot each du/dp with similar settings as for the state values
for idx, (key, values) in enumerate(du_dp_values.items()):
    # Calculate ranks for the current du/dp set
    ranks = np.argsort(np.argsort(-np.array(values))) + 1
    # Create bars for the models and their SC versions
    bars = axes[idx].bar(adjusted_bar_positions - bar_width/2, values[::2], width=bar_width, color='skyblue', label='Original Mode')
    sc_bars = axes[idx].bar(adjusted_bar_positions + bar_width/2, values[1::2], width=bar_width, color='red', label='Sensitivity Constrained')
    
    # Set axis properties
    axes[idx].set_xlabel('Model', fontsize=12)
    axes[idx].set_ylabel(f'R² {key}', fontsize=12)
    axes[idx].set_title(f'Comparison of R² Values for {key}')
    axes[idx].set_xticks(adjusted_bar_positions)
    axes[idx].set_xticklabels(xtick_labels, rotation=45, ha="right")
    axes[idx].set_ylim(0, 1.1)  # Slightly above 1 to accommodate ranks
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate each bar with the corresponding rank
    for bar, sc_bar, rank, sc_rank in zip(bars, sc_bars, ranks[::2], ranks[1::2]):
        axes[idx].text(bar.get_x() + bar.get_width()/2, 0.05, f'Rank {rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)
        axes[idx].text(sc_bar.get_x() + sc_bar.get_width()/2, 0.05, f'Rank {sc_rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)

# Adjust layout
plt.tight_layout()
# Add a legend in the first subplot for clarity
axes[0].legend(fontsize=12)

plt.show()

# %%
# Calculate the mean of du/dp values for each model across all parameters
mean_du_dp_values = np.mean(list(du_dp_values.values()), axis=0)

# Calculate ranks for the mean values
mean_ranks = np.argsort(np.argsort(-mean_du_dp_values)) + 1

# Create a bar chart for the mean R2 values of du/dp for each mode
plt.figure(figsize=(14, 8))
mean_bars = plt.bar(adjusted_bar_positions - bar_width/2, mean_du_dp_values[::2], width=bar_width, color='skyblue', label='Original Mode')
mean_sc_bars = plt.bar(adjusted_bar_positions + bar_width/2, mean_du_dp_values[1::2], width=bar_width, color='red', label='Sensitivity Constrained')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean R² du/dP', fontsize=12)
plt.title('Comparison of Mean R² Values for du/dP Across Models')
plt.xticks(adjusted_bar_positions, xtick_labels, rotation=45, ha="right")

plt.ylim(0, 1.1)  # Set y-axis limit to accommodate ranks and data range
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with the corresponding rank
for bar, sc_bar, rank, sc_rank in zip(mean_bars, mean_sc_bars, mean_ranks[::2], mean_ranks[1::2]):
    plt.text(bar.get_x() + bar.get_width()/2, 0.05, f'Rank {rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)
    plt.text(sc_bar.get_x() + sc_bar.get_width()/2, 0.05, f'Rank {sc_rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)

# Add a legend to the plot
plt.legend(fontsize=12)

plt.show()

# %%