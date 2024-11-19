# %%
from IPython.display import display, clear_output
import numpy as np
import torch
import sys
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, dataset
import pandas as pd

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")


import argparse
import sourcedefender
from lib.util import MHPI
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss
from lib.DerivativeComputer import batchJacobian_AD
from models.MultiWaveletConv_2d import MWNO2d

from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset, compute_residual
Operator = 'MWNO'

# %%
MHPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')
PATH = "fno_pde/PDEs/PDE4/"
# PATH = "../"

# %%
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--enable_ig_loss', type=lambda x: x.lower() == 'true', 
                    choices=[True, False], default=False,
                    help='Enable IG loss (True/False, default: False)')
args = parser.parse_args()
enable_ig_loss = args.enable_ig_loss

enable_eq_loss = False
plot_live_loss = False

t0 = 0.0
t_end = torch.pi
steps_solve = 210
step_save = 30
T_in = 5
T_out = step_save - T_in

state_size = 1  # State size
parameter_size = 4  # Number of PDE parameters

# optimizer and training configurations
epochs = 500
batch_size = 1
scheduler_step = 100
scheduler_gamma = 0.9

if (enable_ig_loss == False):
    Mode = Operator + "_Data"
    ss = 1
# Data + IG
if (enable_ig_loss == True):
    Mode = Operator + "_Data_IG" 
    ss = 1 + parameter_size

print(f'Mode: {Mode}\n')

# %%
L = 1.0
dx = 0.025
nx = int(L / dx)
x = torch.linspace(0, L, nx, device=device)

# %%

train_dataset = torch.load( PATH + 'datasets/train_dataset.pt')
eval_dataset = torch.load( PATH +  'datasets/eval_dataset.pt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

# %%
levels = 2
width = 20
model = MWNO2d(levels=levels, size=[nx, nx], width=width, T_in=T_in, T_out=T_out, state_size=1, parameter_size=parameter_size).to(device)
print(model)

# %%
learning_rate = 0.001
train_fnolosses, train_pdelosses, train_iglosses, val_losses = [], [], [], []
coieffs_list = []
awl = AutomaticWeightedLoss(ss)
optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': 1e-5},
                {'params': awl.parameters(), 'weight_decay': 0}
            ])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

# %%
# Loss function
criterion_1 = nn.MSELoss()

# Start the training loop
outer_loop = tqdm(range(epochs), desc="Progress", position=0)
torch.cuda.empty_cache()

for ep in outer_loop:
    model.train()
    train_fnoloss_accumulated = 0.0
    train_igloss_accumulated = 0.0
    train_pde_loss_accumulated = 0.0

    for batch_data in train_loader:
        
        batch_data = [item.to(device) for item in batch_data]
        batch_parameters, batch_u, du_dparam_true = batch_data

        batch_u_in, batch_u_out = batch_u[..., :T_in, :], batch_u[..., T_in:, :]
        
        batch_size_ = batch_parameters.shape[0]
 
        t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
        x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(batch_size_, 1).to(device)
        t_tensor_.requires_grad_(True)  
        x_tensor_.requires_grad_(True)  
        batch_parameters.requires_grad_(True)        
        
        optimizer.zero_grad()
        
        U_in = batch_u_in
        U_pred = model(U_in, x_tensor_, t_tensor_, batch_parameters)  #u, x, t, par
        data_loss = criterion_1(U_pred, batch_u_out)
               
        
        num_samples_x = 10  # Number of random samples along the second axis (x)
        num_samples_t = 15  # Number of random samples along the third axis (t)

        # Assuming U_pred.shape is (N, X, T, C)
        N, X, T, C = U_pred.shape

        # Generate unique random indices for the desired axes
        random_indices_x = torch.randperm(X)[:num_samples_x]
        random_indices_t = torch.randperm(T)[:num_samples_t]
        
        du_dp = torch.zeros(N, num_samples_x, num_samples_t, model.parameter_size, C).to(device)
        
        for i in range(model.state_size):
            state_tensor = U_pred[:, random_indices_x, :, :][:, :, random_indices_t, :][..., i]
            Jacobian = batchJacobian_AD(state_tensor.reshape(N, num_samples_x * num_samples_t), batch_parameters, graphed=True, batchx=True)
            du_dp[..., i] = Jacobian.reshape(N, num_samples_x, num_samples_t, model.parameter_size)

        ig_loss_list = []
        for i in range(model.parameter_size):
            true_grads = du_dparam_true[:, :, T_in:, :, :][:, random_indices_x, :, :, :][:, :, random_indices_t, :, :][:, :, :, i, :]
            ig_loss_individuals = criterion_1(du_dp[:, :, :, i, :], true_grads)
            ig_loss_list.append(ig_loss_individuals)


        # data , data + ig , data + eq , data + ig + eq
        # Data only
        if ((enable_ig_loss == False) and (enable_eq_loss == False)):
            loss = awl(data_loss)
            fnoloss = data_loss
            coieffs = awl.params.data.clone().detach()
            coieffs[0].item() * data_loss
            ig_loss = sum(1.0 * ig_loss_list[i] for i in range(len(ig_loss_list)))
        
        # Data + IG
        if ((enable_ig_loss == True) and (enable_eq_loss == False)):
            loss = awl(data_loss, *[x for x in ig_loss_list])
            coieffs = awl.params.data.clone().detach()
            fnoloss =  coieffs[0].item() * data_loss
            ig_loss = sum(coieffs[i+1] * ig_loss_list[i] for i in range(len(ig_loss_list)))

        # Data + Eq
        if ((enable_ig_loss == False) and (enable_eq_loss == True)):
            loss = awl(data_loss, eq_loss)
            coieffs = awl.params.data.clone().detach()
            fnoloss =  coieffs[0].item() * data_loss
            ig_loss = sum(1.0 * ig_loss_list[i] for i in range(len(ig_loss_list)))
            eq_loss =  coieffs[-1].item() * eq_loss

        # Data + Eq + IG
        if ((enable_ig_loss == True) and (enable_eq_loss == True)):
            loss = awl(data_loss, *[x for x in ig_loss_list], eq_loss)
            coieffs = awl.params.data.clone().detach()
            fnoloss =  coieffs[0].item() * data_loss
            ig_loss = sum(coieffs[i+1] * ig_loss_list[i] for i in range(len(ig_loss_list)))
            eq_loss =  coieffs[-1].item() * eq_loss
        loss.backward()
        optimizer.step()
        
        train_fnoloss_accumulated += fnoloss.item() * batch_size_
        train_igloss_accumulated += ig_loss.item() * batch_size_

    if enable_ig_loss:
        coieffs_list.append(coieffs)
    epoch_fnoloss = train_fnoloss_accumulated / len(train_loader.dataset)
    epoch_igloss = train_igloss_accumulated / len(train_loader.dataset)
    
    train_fnolosses.append(epoch_fnoloss)
    train_iglosses.append(epoch_igloss)

    model.eval()
    val_loss_accumulated = 0.0
    with torch.no_grad():
        for batch_data in eval_loader:

            batch_data = [item.to(device) for item in batch_data]
            batch_parameters, batch_u, du_dparam_true = batch_data

            batch_u_in, batch_u_out = batch_u[..., :T_in, :], batch_u[..., T_in:, :]

            batch_size_ = batch_parameters.shape[0]
            t_tensor_ = torch.linspace(t0, t_end, step_save)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
            x_tensor_ = torch.linspace(0, L, nx).unsqueeze(0).repeat(batch_size_, 1).to(device)
            
            U_in = batch_u_in
            U_pred = model(U_in, x_tensor_, t_tensor_, batch_parameters)  #u, x, t, par

            val_fnoloss = criterion_1(U_pred, batch_u_out)
            val_loss_accumulated += val_fnoloss.item() * batch_size_

        epoch_val_loss = val_loss_accumulated / len(eval_loader.dataset)
        val_losses.append(epoch_val_loss)

    losses_dict = {'Training Data Loss': train_fnolosses, 'Training IG Loss': train_iglosses, 'Validation Loss': val_losses}
    df = pd.DataFrame(losses_dict)
    
    if ep % 5 == 0:
        torch.save({
            'epoch': ep,
            'levels': levels,
            'size': [nx, nx],
            'width': width,           
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH + 'saved_models/' + Mode + '_saved_model.pth')

        df.to_excel(PATH + 'loss/' + f'losses_data_{Mode}_.xlsx', index=True, engine='openpyxl')
    
        if plot_live_loss:
            loss_live_plot(losses_dict)  # Update the live plot after each epoch

    scheduler.step()
    outer_loop.set_description(f"{Mode} Progress (Epoch {ep + 1}/{epochs})")
    outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}', eval_loss=f'{epoch_val_loss:.2e}')
    
print("Training complete")
#%%