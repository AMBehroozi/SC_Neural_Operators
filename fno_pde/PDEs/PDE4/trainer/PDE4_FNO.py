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
import random
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")


import argparse
import sourcedefender
from lib.util import MHPI
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss, LearnableMultiGradientLossMixer, AdaptiveLossWeights
from lib.DerivativeComputer import batchJacobian_AD
from models.FNO_2d import FNO2d

from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset, compute_residual
Operator = 'FNO'

# %%
MHPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')
PATH = "fno_pde/PDEs/PDE4/"
# PATH = "../"

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--enable_ig_loss', type=lambda x: x.lower() == 'true', 
                    choices=[True, False], default=True,
                    help='Enable IG loss (True/False, default: False)')
args = parser.parse_args()
enable_ig_loss = args.enable_ig_loss
# enable_ig_loss = True


enable_eq_loss = False
plot_live_loss = False

t0 = 0.0
t_end = torch.pi
step_save = 30
T_in = 5
T_out = step_save - T_in

# optimizer and training configurations
epochs = 500
batch_size = 1
learning_rate = 0.01
scheduler_step = 100
scheduler_gamma = 0.9

parameter_size = 82
state_size = 1
n_zones = 40

# Specify the number of samples for each subset (total: 1300 samples)
train_sample = 50
eval_sample = 150
test_sample = 150

if (enable_ig_loss == False):
    Mode = Operator + f"_Data_zoned_2_{n_zones}_{train_sample}"
    ss = 1
# Data + IG
if (enable_ig_loss == True):
    Mode = Operator + f"_Data_IG_zoned_2_{n_zones}_{train_sample}" 
    ss = 1 + parameter_size

print(f'Mode: {Mode}\n')

# %%
L = 1.0
dx = 0.0125
nx = int(L / dx)
x = torch.linspace(0, L, nx, device=device)

# %%
filename = f'datasets/train_dataset_zones_2_{n_zones}.pt'

# Load the dataset from a file
train_dataset = torch.load(PATH + filename)

# Set a random seed for reproducibility
random.seed(42)  # You can choose any seed you prefer
torch.manual_seed(42)


# Calculate the total number of samples to allocate to subsets
total_samples = train_sample + eval_sample + test_sample

# Ensure the dataset has enough data points
if len(train_dataset) < total_samples:
    raise ValueError("The dataset does not contain enough samples for the desired splits.")

# Shuffle the dataset indices
indices = torch.randperm(len(train_dataset))

# Split indices for each dataset
train_indices = indices[:train_sample]
eval_indices = indices[train_sample:train_sample + eval_sample]
test_indices = indices[train_sample + eval_sample:total_samples]
rest_indices = indices[total_samples:]  # Remaining data after allocation to train, eval, test

# Use the indices to create data subsets
train_data = torch.utils.data.Subset(train_dataset, train_indices)
eval_data = torch.utils.data.Subset(train_dataset, eval_indices)
test_data = torch.utils.data.Subset(train_dataset, test_indices)
rest_data = torch.utils.data.Subset(train_dataset, rest_indices)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=batch_size)


# %%
modes1 = 8
modes2 = 8
width = 20

model = FNO2d(modes1, modes2,  width, T_in, T_out, parameter_size=train_dataset.dataset.tensors[0].shape[1], state_size=train_dataset.dataset.tensors[1].shape[-1]).to(device)
# %%
learning_rate = 0.001
train_fnolosses, train_pdelosses, train_iglosses, val_losses = [], [], [], []
coieffs_list = []
awl = AutomaticWeightedLoss(ss)
# awl = LearnableMultiGradientLossMixer(ss).to(device)
optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': 0.0},
                {'params': awl.parameters(), 'lr': 0.01}
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


        # data , data+ig , data + eq , data + ig + eq
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
            'modes1': modes1,
            'modes2':modes2,
            'width': width,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH + 'saved_models/' + Mode + '_saved_model.pth')
        df.to_excel(PATH + 'loss/' + f'losses_data_{Mode}.xlsx', index=True, engine='openpyxl')   
        
        if plot_live_loss:
            loss_live_plot(losses_dict)  # Update the live plot after each epoch

    scheduler.step()
    outer_loop.set_description(f"{Mode} Progress (Epoch {ep + 1}/{epochs})")
    outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}', eval_loss=f'{epoch_val_loss:.2e}')
    

print("Training complete")
#%%