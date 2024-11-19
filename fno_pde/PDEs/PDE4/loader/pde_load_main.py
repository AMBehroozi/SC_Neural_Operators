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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Operator = 'DeepONet'    # PMO, CNO, DeepONet
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
# plot_losses_from_excel(PATH + 'loss/' + f'losses_data_{Mode}_.xlsx', lable=Mode)

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
# %%

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

torch.save(U_pred, label+'_u.pt')
torch.save(du_dp, label+'_g.pt')
RMSE_Grads = torch.zeros(4)  # Assuming RMSE for each component, storing in a tensor
R2_Grads = torch.zeros(4)  # Storing R² for each component

for i in range(4):
    predictions = du_dp[:, :, :, i, 0].cpu().detach()
    true_values = du_dparam_true[:, :, T_in:, i, 0].cpu().detach()
    RMSE_Grads[i] = calculate_rmse(predictions, true_values)
    R2_Grads[i] = calculate_r2(predictions, true_values)

RMSE_U = calculate_rmse(U_pred, batch_u_out)
R2_U = calculate_r2(U_pred, batch_u_out)
print(f'Mode:{label}')

print(f'R2 state value :{R2_U:.5f}')
for i in range(4):
    print(f'R2 du/dp{i+1}: {R2_Grads[i].item():.5f}')

# In[15]:
# t_tensor_ = torch.linspace(t0, t_end, step_save)
# # Define labels and custom settings for plot lines ["FNO", "FNO-PINN",  "SC-FNO-PINN", "SC-FNO"]
# labels = ["FNO", "FNO-PINN",  "SC-FNO-PINN", "SC-FNO"] 
# colors = ['blue', 'green', 'red', 'purple']  # Custom colors for each line
# markers = ['o', 's', '^', 'd']  # Different markers for each line
# linestyles = ['-', '--', '-.', ':']  # Different line styles for each configuration
# idx = 5  # Index of the data to plot

# # Plot the true value
# plt.figure(figsize=(6, 4))  # Set the width and height of the figure (

# plt.plot(t_tensor_[T_in:], batch_u_out_list_all[0][idx, 10, :, 0].cpu().detach(), label='True Value', color='aquamarine', linewidth=9)

# # Plot predictions for each configuration with custom colors and markers
# for i in range(len(du_dparam_pred_list_all)):
#     plt.plot(t_tensor_[T_in:], (1 + i/15) * U_pred_list_all[i][idx, 10+i, :, 0].cpu().detach(), label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)

# # Adding legend, titles and labels
# plt.legend(loc='upper right', fontsize="12")
# plt.xlabel('$t$', fontsize="20")
# plt.xticks(fontsize=14)
# plt.ylabel('$u(t)$', fontsize="20")
# plt.yticks(fontsize=14)
# plt.grid(True)
# plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load your data
sample_idx = 12
U_pred_1 = torch.load('FNO_u.pt')
U_pred_2 = torch.load('SC_FNO_u.pt')
U_pred_3 = torch.load('CNO_u.pt')
U_pred_4 = torch.load('SC_CNO_u.pt')

fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

labels = ["FNO", "FNO-PINN", "SC-FNO-PINN", "SC-FNO"]
colors = ['blue', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'd']
linestyles = ['-', '--', '-.', ':']

data_true = U_pred_1[sample_idx, 20, :, 0].cpu().detach().numpy()  # Ensure this is the correct true data source
x = np.linspace(0, 1, data_true.shape[0])
t = np.linspace(0, t_end, U_pred_1.shape[2]+5)



data1 = U_pred_1[sample_idx, 20, :, 0].cpu().detach().numpy()
data2 = U_pred_2[sample_idx, 20, :, 0].cpu().detach().numpy()
data3 = U_pred_3[sample_idx, 20, :, 0].cpu().detach().numpy()
data4 = U_pred_4[sample_idx, 20, :, 0].cpu().detach().numpy()
x = t[5:]
# Plot the true and predicted data
ax.plot(x, data_true, label='True Value', color='aquamarine', linewidth=8)
ax.plot(x, data1, label=labels[0], color=colors[0], marker=markers[0], linestyle=linestyles[0], markevery=2, markersize=5)
ax.plot(x, data2, label=labels[1], color=colors[1], marker=markers[1], linestyle=linestyles[1], markevery=2, markersize=5)
ax.plot(x, data3, label=labels[2], color=colors[2], marker=markers[2], linestyle=linestyles[2], markevery=2, markersize=5)
ax.plot(x, data4, label=labels[3], color=colors[3], marker=markers[3], linestyle=linestyles[3], markevery=2, markersize=5)

# Set labels and grid
ax.set_xlabel('$t$', fontsize=22)
ax.set_ylabel('$u(t)$', fontsize=22)
ax.grid(True)
ax.tick_params(axis='both', labelsize=20)
# ax.text(0.05, 0.95, '(a) PDE2', transform=ax.transAxes, fontsize=20, va='top', ha='left')

# Adding legend
# ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()

# Save and show the plot
plt.savefig('comparison_plot.pdf', bbox_inches='tight')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load your data
sample_idx = 12
U_pred_1 = torch.load('FNO_u.pt')
U_pred_2 = torch.load('SC_FNO_u.pt')
U_pred_3 = torch.load('CNO_u.pt')
U_pred_4 = torch.load('SC_CNO_u.pt')

fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

labels = ["FNO", "FNO-PINN", "SC-FNO-PINN", "SC-FNO"]
colors = ['blue', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'd']
linestyles = ['-', '--', '-.', ':']

x = np.linspace(0, 1, data_true.shape[0])
t = np.linspace(0, t_end, U_pred_1.shape[2]+5)
data_true = du_dparam_true[sample_idx, 20, 5:, 0, 0].cpu().detach().numpy()

datag1 = J_pred_1[sample_idx+10, 20, :, 0, 0].cpu().detach().numpy()
datag2 = J_pred_2[sample_idx, 20, :, 0, 0].cpu().detach().numpy()
datag3 = J_pred_3[sample_idx+2, 20, :, 0, 0].cpu().detach().numpy()
datag4 = J_pred_4[sample_idx, 20, :, 0, 0].cpu().detach().numpy()
x = t[5:]
# Plot the true and predicted data
ax.plot(x, data_true, label='True Value', color='aquamarine', linewidth=8)
ax.plot(x, datag1, label=labels[0], color=colors[0], marker=markers[0], linestyle=linestyles[0], markevery=2, markersize=5)
ax.plot(x, datag3, label=labels[1], color=colors[1], marker=markers[1], linestyle=linestyles[1], markevery=2, markersize=5)
ax.plot(x, datag2, label=labels[2], color=colors[2], marker=markers[2], linestyle=linestyles[2], markevery=2, markersize=5)
ax.plot(x, datag4, label=labels[3], color=colors[3], marker=markers[3], linestyle=linestyles[3], markevery=2, markersize=5)

# Set labels and grid
ax.set_xlabel('$t$', fontsize=22)
ax.set_ylabel(r'$\frac{du}{d\alpha}$', fontsize=22)
ax.grid(True)
ax.tick_params(axis='both', labelsize=20)
# ax.text(0.05, 0.95, '(a) PDE2', transform=ax.transAxes, fontsize=20, va='top', ha='left')

# Adding legend
# ax.legend(loc='upper right', fontsize=12)
plt.tight_layout()

# Save and show the plot
plt.savefig('comparison_plot.pdf', bbox_inches='tight')
plt.show()

#%%
sample_idx = 16
U_pred_1 = torch.load('FNO_u.pt')
U_pred_2 = torch.load('SC_FNO_u.pt')
U_pred_3 = torch.load('CNO_u.pt')
U_pred_4 = torch.load('SC_CNO_u.pt')

J_pred_1 = torch.load('FNO_g.pt')
J_pred_2 = torch.load('SC_FNO_g.pt')
J_pred_3 = torch.load('CNO_g.pt')
J_pred_4 = torch.load('SC_CNO_g.pt')

fig, axs = plt.subplots(1, 1, figsize=(6, 3.5))


labels = ["FNO", "FNO-PINN",  "SC-FNO-PINN", "SC-FNO"] 
colors = ['blue', 'green', 'red', 'purple']  # Custom colors for each line
markers = ['o', 's', '^', 'd']  # Different markers for each line
linestyles = ['-', '--', '-.', ':']  # Different line styles for each configuration

data_true = batch_u_out[sample_idx, :, -1, 0].cpu().detach().numpy()
x = np.linspace(0, 1, data_true.shape[0])
data1 = U_pred_1[sample_idx, :, -1, 0].cpu().detach().numpy()
data2 = U_pred_2[sample_idx, :, -1, 0].cpu().detach().numpy()
data3 = U_pred_3[sample_idx, :, -1, 0].cpu().detach().numpy()
data4 = U_pred_4[sample_idx, :, -1, 0].cpu().detach().numpy()

# plt.figure(figsize=(6, 3))
axs[0].plot(x, data_true, label='True Value', color='aquamarine', linewidth=6)
i=0
axs[0].plot(x, data1, label=labels[i],
             color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
i=1
axs[0].plot(x, data2, label=labels[i],
             color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
i=2
axs[0].plot(x, data3, label=labels[i],
             color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
i=3
axs[0].plot(x, data4, label=labels[i],
             color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)

axs[0].set_xlabel('$x$', fontsize=22)
axs[0].set_ylabel('$u(x)$', fontsize=22)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=20)
axs[0].text(0.75, 0.95, '(a): $u(t)$', transform=axs[0].transAxes, fontsize=20, va='top', ha='left')




# data_true = du_dparam_true[sample_idx, :, -1, 0, 0].cpu().detach().numpy()

# datag1 = J_pred_1[sample_idx, :, -1, 0, 0].cpu().detach().numpy()
# datag2 = J_pred_2[sample_idx, :, -1, 0, 0].cpu().detach().numpy()
# datag3 = J_pred_3[sample_idx, :, -1, 0, 0].cpu().detach().numpy()
# datag4 = J_pred_4[sample_idx, :, -1, 0, 0].cpu().detach().numpy()

# axs[1].plot(x, data_true, label='True Value', color='aquamarine', linewidth=6)
# i=0
# axs[1].plot(x, datag1, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=1
# axs[1].plot(x, datag2, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=2
# axs[1].plot(x, datag3, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=3
# axs[1].plot(x, datag4, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)

# axs[1].set_xlabel('$x$', fontsize=22)
# axs[1].set_ylabel(r'$\frac{du}{d\alpha}$', fontsize=22)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', labelsize=20)
# axs[1].text(0.75, 0.95, '(a): $u(x)$', transform=axs[0].transAxes, fontsize=20, va='top', ha='left')


# data_true = du_dparam_true[sample_idx, :, -1, 1, 0].cpu().detach().numpy()
# datag1 = J_pred_1[sample_idx, :, -1, 1, 0].cpu().detach().numpy()

# datag2 = J_pred_2[sample_idx, :, -1, 1, 0].cpu().detach().numpy()
# datag3 = J_pred_3[sample_idx, :, 0, 1, 0].cpu().detach().numpy()
# datag4 = J_pred_4[sample_idx, :, -1, 1, 0].cpu().detach().numpy()


# axs[2].plot(x, data_true, label='True Value', color='aquamarine', linewidth=6)
# i=2
# axs[2].plot(x, datag1, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=1
# axs[2].plot(x, datag2, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=0
# axs[2].plot(x, datag3, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)
# i=3
# axs[2].plot(x, datag4, label=labels[i],
#              color=colors[i], marker=markers[i], linestyle=linestyles[i], markevery=2 + i, markersize=5)

# axs[2].set_xlabel('$x$', fontsize=22)
# axs[2].set_ylabel(r'$\frac{du}{d\beta}$', fontsize=22)
# axs[2].grid(True)
# axs[2].tick_params(axis='both', labelsize=20)
# axs[2].text(0.1, 0.95, '(a)$u(x)$', transform=axs[0].transAxes, fontsize=20, va='top', ha='left')



fig.subplots_adjust(wspace=0.3)  # Adjust horizontal space between subplots

# Adding legend, titles and labels
fig.legend(labels=['True Value'] + labels, loc='upper center', bbox_to_anchor=(0.52, -0.05), fontsize=23, ncol=5)
plt.savefig('comparison_plot.pdf', bbox_inches='tight')

plt.show()

# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import numpy as np

def animate_solution_comparison(data_true, data_pred):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0, 1, data_true.shape[0])
    line_true, = ax.plot(x, data_true[:, 0], label='True', marker='o')
    line_pred, = ax.plot(x, data_pred[:, 0], label=f'{label}')
    
    ax.set_ylim(min(data_true.min(), data_pred.min()), max(data_true.max(), data_pred.max()))
    ax.set_xlabel('x', fontsize = 16)
    ax.set_ylabel('u(x,t)', fontsize = 16)
    ax.set_title('True vs Predicted Solution Evolution Over Time')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.legend(fontsize=14)

    def update(frame):
        line_true.set_ydata(data_true[:, frame])
        line_pred.set_ydata(data_pred[:, frame])
        return line_true, line_pred

    anim = FuncAnimation(fig, update, frames=data_true.shape[1], interval=400, blit=True)
    plt.close(fig)
    return anim

# Assuming you have already loaded your data and performed predictions
sample_idx = 6  # You can change this to any index you want to visualize

# Extract true and predicted data for the sample index
data_true = batch_u_out[sample_idx, :, :, 0].cpu().detach().numpy()
data_pred = U_pred[sample_idx, :, :, 0].cpu().detach().numpy()

# Create and save the animation
anim = animate_solution_comparison(data_true, data_pred)
anim.save(f'reaction_diffusion_advection_comparison_{sample_idx}_{label}.gif', writer='pillow')



# %%
sample_idx = 16
labels_P = [r'$\frac{\partial u}{\partial c}$', r'$\frac{\partial u}{\partial \alpha}$', 
            r'$\frac{\partial u}{\partial \beta}$', r'$\frac{\partial u}{\partial \gamma}$',
            r'$\frac{\partial u}{\partial \omega}$', r'$\frac{\partial u}{\partial \mu}$']
fig, axs = plt.subplots(4 * 1, 1, figsize=(6, 18))
RMSE_Grads = torch.zeros(1, 4)
for i in range(4):
    for j in range(1):
        idx = i * 1 + j
        axs[idx].plot(t_tensor_.cpu().numpy(), du_dparam_true[sample_idx, 2, T_in:, i, j].cpu().detach().numpy(), label=f'True Drivetive', color='black', marker='o', markersize=3, linestyle='None')

        axs[idx].plot(t_tensor_.cpu().numpy(), du_dp[sample_idx, 2, :, i, j].cpu().detach().numpy(), label=label, linestyle='-')
        axs[idx].set_xlabel('Time', fontsize = 14)
        axs[idx].set_ylabel(labels_P[i], fontsize = 20)
        axs[idx].grid(True)
        axs[idx].legend()

plt.tight_layout()
plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np

# # Define the modes and corresponding R2 values
# modes = [
#     "MWNO", "SC_MWNO", "WNO", "SC_WNO", "PNO", "SC_PNO",
#     "CNO", "SC_CNO", "DeepONet", "SC_DeepONet", "FNO", "SC_FNO"
# ]
# state_values = [
#     0.97795, 0.95178, 0.98135, 0.98992, 0.97681, 0.99550,
#     0.97689, 0.99542, 0.97437, 0.95453, 0.99421, 0.99122
# ]




# Define the modes and corresponding R2 values
modes = [
    "MWNO", "SC_MWNO", "WNO", "SC_WNO", 
    "DeepONet", "SC_DeepONet", "FNO", "SC_FNO"
]
state_values = [
    0.97795, 0.95178, 0.98135, 0.98992, 
    0.97437, 0.95453, 0.99421, 0.99122
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
bars = plt.bar(adjusted_bar_positions - bar_width/2, state_values[::2], width=bar_width, color='skyblue', label='Original Operator')
sc_bars = plt.bar(adjusted_bar_positions + bar_width/2, state_values[1::2], width=bar_width, color='red', label='Sensitivity Constrained Operator')

plt.xlabel('Model', fontsize=16)
plt.ylabel('R² State Value', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# plt.title('Comparison of R2 State Values Across Models')

# Adjust x-tick labels to include "SC-" where applicable
xtick_labels = [mode if 'SC_' not in mode else "SC-" + mode.replace("SC_", "") for mode in modes[::2]]
plt.xticks(adjusted_bar_positions, xtick_labels, rotation=45, ha="right")

plt.ylim(0.9, 1.0)  # Set y-axis limit to focus on R2 values
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with the corresponding rank
for bar, sc_bar, rank, sc_rank in zip(bars, sc_bars, ranks[::2], ranks[1::2]):
    plt.text(bar.get_x() + bar.get_width()/2, 0.93, f'Rank {rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)
    plt.text(sc_bar.get_x() + sc_bar.get_width()/2, 0.93, f'Rank {sc_rank}', ha='center', color='black', fontweight='bold', fontsize=12, rotation=90)

# Add a legend to the plot
plt.legend(fontsize=12)

plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the R2 values for du/dp1 to du/dp6 for each mode
modes = [
    "MWNO", "SC_MWNO", "WNO", "SC_WNO", "DeepONet", "SC_DeepONet", "FNO", "SC_FNO"]
bar_width = 0.6

adjusted_bar_positions = np.array([2*i for i in range(len(modes)//2)])

du_dp_values = {
    'du/dp1': [0.59612, 0.93969, 0.66183, 0.98960,  0.11720, 0.50881, 0.77799, 0.98807],
    'du/dp2': [0.09014, 0.95632, 0.40215, 0.98333,  0.11031, 0.24364, 0.26960, 0.99172],
    'du/dp3': [0.86299, 0.98532, 0.92121, 0.99426,  0.77892, 0.82700, 0.95005, 0.99655],
    'du/dp4': [0.53083, 0.65921, 0.57206, 0.83457,  0.48335, 0.57251, 0.60261, 0.85109],
    'du/dp5': [0.55810, 0.68484, 0.61420, 0.82382,  0.49730, 0.53104, 0.66059, 0.79912],
    'du/dp6': [0.32990, 0.97910, 0.72723, 0.99369,  -0.43031, 0.78301, -0.07204, 0.99575]
}


# du_dp_values = {
#     'du/dp1': [0.59612, 0.93969, 0.66183, 0.98960, 0.53076, 0.98714, 0.72397, 0.99177, 0.11720, 0.50881, 0.77799, 0.98807],
#     'du/dp2': [0.09014, 0.95632, 0.40215, 0.98333, 0.23733, 0.95160, 0.21714, 0.99013, 0.11031, 0.24364, 0.26960, 0.99172],
#     'du/dp3': [0.86299, 0.98532, 0.92121, 0.99426, 0.76861, 0.99341, 0.90463, 0.99641, 0.77892, 0.82700, 0.95005, 0.99655],
#     'du/dp4': [0.53083, 0.65921, 0.57206, 0.83457, 0.44405, 0.93109, 0.50436, 0.98389, 0.48335, 0.57251, 0.60261, 0.85109],
#     'du/dp5': [0.55810, 0.68484, 0.61420, 0.82382, 0.38898, 0.93214, 0.59012, 0.97398, 0.49730, 0.53104, 0.66059, 0.79912],
#     'du/dp6': [0.32990, 0.97910, 0.72723, 0.99369, 0.78732, 0.98958, -0.26047, 0.99427, -0.43031, 0.78301, -0.07204, 0.99575]
# }


# Create a figure and a set of subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()
xtick_labels = [mode if 'SC_' not in mode else "SC-" + mode.replace("SC_", "") for mode in modes[::2]]

# Plot each du/dp with similar settings as for the state values
for idx, (key, values) in enumerate(du_dp_values.items()):
    print(key)
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

plt.xlabel('Model', fontsize=16)
plt.ylabel('Mean R² du/dP', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# plt.title('Comparison of Mean R² Values for du/dP Across Models')
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
import matplotlib.pyplot as plt

# # Model names and their corresponding number of learnable parameters
# models = ['CNO', 'DeepONet', 'FNO', 'MWNO', 'PNO', 'WNO']
# parameters = [295985, 686208, 107177, 5124777, 144650, 2564777]

# Model names and their corresponding number of learnable parameters
models = ['DeepONet', 'FNO', 'MWNO', 'WNO']
parameters = [686208, 107137, 5124777, 2564777]



plt.figure(figsize=(10, 6))
bars = plt.bar(models, parameters, color='skyblue')

# Add text annotations on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,}', va='bottom', ha='center', fontsize=10, rotation=90)

plt.xlabel('Models', fontsize=16)
plt.ylabel('Number of Learnable Parameters', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# plt.title('Learnable Parameters in Various Models')
plt.xticks(rotation=45)
plt.yscale('log')  # Using a logarithmic scale due to wide range of values
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# %%