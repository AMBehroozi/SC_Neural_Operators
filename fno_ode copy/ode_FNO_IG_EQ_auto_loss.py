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

sys.path.append("../")
sys.path.append("./")

from lib.util import MHPI
from lib.utiltools import loss_live_plot, GaussianRandomFieldGenerator, generate_batch_parameters, AutomaticWeightedLoss
from lib.DerivativeComputer import batchJacobian_AD
from models.FNO_1d_simple import FNO1d
from ode_fno.ODEs.ODE3.ode3 import creat_dataset, ode_residual

# %%
MHPI()
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')

# %%
# Define constants and parameters
enable_ig_loss = False
enable_eq_loss = True

Sample_number = 5000
training_sample = 10
dataset_segment_size = 1000

plot_live_loss = True
Create_new_dataset = False

modes = 8
width = 20

alpha_ = 1.5
tau = 0.5

t0 = 0.0
t_end = 2
steps = 100
T_in = 10
T_out = steps - T_in

# optimizer and training configurations
epochs = 500
batch_size = 16
learning_rate = 0.01
scheduler_step = 100
scheduler_gamma = 0.9

# %%
t_tensor = torch.linspace(t0, t_end, steps)  # Shape: [Steps]
# Initialize GaussianRandomFieldGenerator
grf_generator = GaussianRandomFieldGenerator(alpha=alpha_, tau=tau, N=Sample_number)


# alpha, beta, gamma, delta, omega, ic1, ic2
alfa = generate_batch_parameters(grf_generator, [(0.02, 0.06), 
                                                 (0.01, 0.03), 
                                                 (20, 60), 
                                                 (0.5, 1.5),
                                                 (0.2, 0.6)])

initial_conditions = generate_batch_parameters(grf_generator, [(0.0, 0.2), 
                                                               (0.0, 0.2)])
parameters = torch.cat((alfa, initial_conditions), dim=-1) # Shape: [Steps, 3]  (alfa, beta , ic)
parameters.requires_grad_(True)

# %%
PATH = '/projects/mhpi/mehdi/projects/FNO_SWE_2D/ode_fno/ODEs/ODE2'
# Create dataset < This part can be changed based on different cases>
if Create_new_dataset:
    dataset = creat_dataset(t_tensor, parameters, T_in)
    torch.save(dataset, PATH + '/datasets/main_dataset_ODE2.pt')
    # Calculate sizes for train/eval/test split
    train_size = int(0.75 * Sample_number)
    eval_size = int(0.15 * Sample_number)
    test_size = Sample_number - train_size - eval_size

    # Create DataLoaders
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size]) # Split the dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)

    torch.save(train_loader, PATH + '/datasets/train_dataset_ODE2.pt')
    torch.save(eval_loader, PATH + '/datasets/eval_dataset_ODE2.pt')
    torch.save(test_loader, PATH + '/datasets//test_dataset_ODE2.pt')

else:
    dataset = torch.load(PATH + '/datasets/main_dataset_ODE2.pt')
    train_loader = torch.load(PATH + '/datasets/train_dataset_ODE2.pt')
    eval_loader = torch.load(PATH + '/datasets/eval_dataset_ODE2.pt')
    test_loader = torch.load(PATH + '/datasets/test_dataset_ODE2.pt')


# %%
model = FNO1d(modes, width, T_in, T_out, state_size=dataset.tensors[1].shape[-1], parameters_size=dataset.tensors[0].shape[-1]).to(device)
print(model)

# %%
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=1.025)
train_fnolosses, train_odelosses, train_iglosses, val_losses = [], [], [], []
coieffs_list = []
awl = AutomaticWeightedLoss(9)
optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': learning_rate},
                {'params': awl.parameters(), 'weight_decay': 0}
            ])

# %%
# Loss function
criterion_1 = nn.MSELoss()
# Assuming all initial setups are done as before

# Start the training loop
outer_loop = tqdm(range(epochs), desc="Progress", position=0)
torch.cuda.empty_cache()

for ep in outer_loop:
    model.train()
    train_fnoloss_accumulated = 0.0
    train_igloss_accumulated = 0.0
    # train_odeloss_accumulated = 0.0

    for batch_data in train_loader:
        batch_data = [item.to(device) for item in batch_data]
        batch_parameters, batch_u_in, batch_u_out, du_dparam_true = batch_data
        # du_dparam_true.requires_grad_(True)
        batch_parameters.requires_grad_(True)
        
        batch_size_ = batch_parameters.shape[0]
        t_tensor_ = torch.linspace(t0, t_end, steps)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
        t_tensor_.requires_grad_(True)
        optimizer.zero_grad()

        U_in = batch_u_in
        U_pred = model(U_in, t_tensor_, batch_parameters)
        
        data_loss = criterion_1(U_pred, batch_u_out)
        
        du_dp = torch.zeros(batch_size_, T_out, model.parameters_size, model.state_size).to(device)
        for i in range(model.state_size):
            du_dp[..., i] = batchJacobian_AD(U_pred[..., i], batch_parameters, graphed=True, batchx=True)

        ig_loss_list = []
        for i in range(model.parameters_size):
            ig_loss_individuals = criterion_1(du_dp[:, :, i, :], du_dparam_true[:, T_in:, i, :])
            ig_loss_list.append(ig_loss_individuals)
        
        residual = ode_residual(U_pred, batch_parameters, t_tensor_)
        eq_loss = criterion_1(residual, torch.zeros_like(residual))
        
        loss = awl(data_loss, *[x for x in ig_loss_list], eq_loss)   
        loss.backward()
        optimizer.step()
        coieffs = awl.params.data.clone().detach()
        fnoloss =  coieffs[0].item() * data_loss.item()
        
        ig_loss = sum(coieffs[i+1].item() * ig_loss_list[i].item() for i in range(len(ig_loss_list)))
        
        train_fnoloss_accumulated += fnoloss * batch_size_
        train_igloss_accumulated += ig_loss * batch_size_
        # train_odeloss_accumulated += eq_loss * batch_size_

    coieffs_list.append(coieffs)
    epoch_fnoloss = train_fnoloss_accumulated / len(train_loader.dataset)
    epoch_igloss = train_igloss_accumulated / len(train_loader.dataset)
    # epoch_pdeloss = train_odeloss_accumulated / len(train_loader.dataset)
    
    train_fnolosses.append(epoch_fnoloss)
    train_iglosses.append(epoch_igloss)
    # train_odelosses.append(epoch_pdeloss)
    # Evaluation phase (if applicable)
    model.eval()
    val_loss_accumulated = 0.0
    with torch.no_grad():
        for batch_data in eval_loader:
            batch_data = [item.to(device) for item in batch_data]
            batch_parameters, batch_u_in, batch_u_out = batch_data[:3]
            batch_size_ = batch_parameters.shape[0]
            t_tensor_ = torch.linspace(t0, t_end, steps)[T_in:].unsqueeze(0).repeat(batch_size_, 1).to(device)
            U_in = batch_u_in
            U_pred = model(U_in, t_tensor_, batch_parameters)

            val_fnoloss = criterion_1(U_pred, batch_u_out)

            val_loss_accumulated += val_fnoloss.item() * batch_size_

        epoch_val_loss = val_loss_accumulated / len(eval_loader.dataset)
        val_losses.append(epoch_val_loss)


    losses_dict = {'Training FNO Loss': train_fnolosses, 'Training IG Loss': train_iglosses, 'Validation Loss': val_losses}
    if ep%10 == 0:
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, PATH + '/saved_model/ODE_2_saved_Model_Data_IG_EQ_saved.pth')

        # loss_live_plot(losses_dict)  # Update the live plot after each epoch

    outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs})")
    outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}', eval_loss=f'{epoch_val_loss:.2e}')

print("Training complete")

# # %%
# import matplotlib.pyplot as plt
# import torch
# from lib.utiltools import RMSE, nse_cdf_plot
# error_calc = RMSE()
# def plot_results(model, data_loader, device):
#     model.eval()  # Set the model to evaluation mode.
    
#     for batch_data in data_loader:
#         # Load a batch and move to the device
#         batch_data = [item.to(device) for item in batch_data]
#         batch_parameters, batch_u_in, batch_u_out, du_dparam_true = batch_data
#         batch_parameters.requires_grad_(True)
#         # Prepare inputs and make predictions
#         batch_size_ = batch_parameters.shape[0]
#         U_in = batch_u_in
#         t_tensor = torch.linspace(t0, t_end, steps)[T_in:].unsqueeze(0).repeat(batch_parameters.shape[0], 1).to(device)
#         U_pred = model(U_in, t_tensor, batch_parameters)  # Assume model outputs have an extra dimension for channels
        
#         du_dp = torch.zeros(batch_size_, T_out, model.parameters_size, model.state_size).to(device)
#         for i in range(model.state_size):
#             du_dp[..., i] = batchJacobian_AD(U_pred[..., i], batch_parameters, graphed=False, batchx=True)
#         # Detach and move data back to CPU for plotting
#         U_pred = U_pred[..., 0].detach().cpu()
#         batch_u_out = batch_u_out[..., 0].detach().cpu()


    
#     # Sample index for plotting
#     sample_idx = 0  # Assuming a valid index from the batch
#     RMSE_U = error_calc(U_pred, batch_u_out)

#         # # Example usage
#     true_dict = {'u': batch_u_out}
#     pred_dict = {'u': U_pred}

#     # nse_cdf_plot( true_dict, pred_dict)
#     # Plotting U_pred and U_out
#     plt.figure(figsize=(14, 8))
#     plt.plot(t_tensor.cpu().numpy()[sample_idx, ...], U_pred[sample_idx, ...], label='U Prediction')
#     plt.plot(t_tensor.cpu().numpy()[sample_idx, ...], batch_u_out[sample_idx, ...], label='U True', linestyle='--')
#     plt.title('Comparison of Predicted and True U')
#     plt.xlabel('Time')
#     plt.ylabel('U')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


#     # Plotting derivatives
#     np, nu = du_dp.shape[-2], du_dp.shape[-1]
#     fig, axs = plt.subplots(np * nu, 1, figsize=(6, 18))
#     RMSE_Grads = torch.zeros(nu, np)
#     for i in range(np):
#         for j in range(nu):
#             idx = i * nu + j
#             RMSE_Grads[j, i] =error_calc(du_dparam_true[:, T_in:, i, j].cpu(), du_dp[:, :, i, j].cpu())
#             axs[idx].plot(t_tensor.cpu().numpy()[sample_idx], du_dparam_true[sample_idx, T_in:, i, j].cpu(), label=f'du_{j}/dp_{i} True')
#             axs[idx].plot(t_tensor.cpu().numpy()[sample_idx], du_dp[sample_idx, :, i, j].cpu(), label=f'du_{j}/dp_{i} Prediction', linestyle='--')
#             axs[idx].set_title(f'Derivative du_{j}/dp_{i}')
#             axs[idx].set_xlabel('Time')
#             axs[idx].set_ylabel(f'du_{j}/dp_{i}')
#             axs[idx].grid(True)
#             axs[idx].legend()
#     plt.tight_layout()
#     plt.show()
    
#     print(f'        U RMSE: {RMSE_U:.2e}')
#     for i in range(np):
#         for j in range(nu):
#             print(f'du_{j}/dp_{i} RMSE: {RMSE_Grads[j, i]:.2e}')
# # Example of how to use this function
# plot_results(model, test_loader, device)
