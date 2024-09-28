import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
import scipy.io
import h5py
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
from scipy.fftpack import idct
import torch.utils.data
import torch
import torch.nn as nn
import os
import pandas as pd
import torch.nn.functional as F
from IPython import display


def plot_realtime_metrics(max_iterations):
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    losses, mu_diffs, mus, mu_targets = [], [], [], []
    iterations = []
    
    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])
    line3, = ax3.plot([], [])
    target_line, = ax3.plot([], [], 'r--')

    # ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Iterations')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale

    # ax2.set_xlabel('Iterations')
    ax2.set_ylabel('μ Difference')
    ax2.set_title('|μ - μ_target| / μ_target vs Iterations')

    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('μ')
    ax3.set_title('μ vs Iterations')

    def calculate_mu_difference(mu_current, mu_target):
        return abs(mu_current - mu_target) / abs(mu_target)

    def update_plot(iteration, loss, current_mu, target_mu):
        iterations.append(iteration)
        losses.append(loss)
        mus.append(current_mu)
        mu_targets.append(target_mu)
        
        # Calculate normalized difference between target_mu and current_mu
        mu_diff = calculate_mu_difference(current_mu, target_mu)
        mu_diffs.append(mu_diff)
        
        line1.set_xdata(iterations)
        line1.set_ydata(losses)
        line2.set_xdata(iterations)
        line2.set_ydata(mu_diffs)
        line3.set_xdata(iterations)
        line3.set_ydata(mus)
        target_line.set_xdata(iterations)
        target_line.set_ydata(mu_targets)
        
        for ax in (ax1, ax2, ax3):
            ax.relim()
            ax.autoscale_view()
        
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        return mu_diff

    return update_plot


def plot_realtime_metrics_batch(max_iterations, batch_size):
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    losses, mu_diffs = [], []
    iterations = []
    
    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])
    scatter = ax3.scatter([], [])
    perfect_line, = ax3.plot([], [], 'r--', label='Perfect prediction')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Iterations')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Avg μ Difference')
    ax2.set_title('Avg |μ - μ_target| / μ_target vs Iterations')

    ax3.set_xlabel('Target α')
    ax3.set_ylabel('Predicted α')
    ax3.set_title('Current Predicted vs Target μ')
    ax3.legend()

    # Make ax3 square
    ax3.set_aspect('equal', adjustable='box')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    def calculate_mu_difference(mu_current, mu_target):
        return torch.abs(mu_current - mu_target) / torch.abs(mu_target)

    def update_plot(iteration, loss, current_mus, target_mus):
        iterations.append(iteration)
        losses.append(loss)
        
        mu_diff = calculate_mu_difference(current_mus, target_mus).mean().item()
        mu_diffs.append(mu_diff)
        
        line1.set_xdata(iterations)
        line1.set_ydata(losses)
        line2.set_xdata(iterations)
        line2.set_ydata(mu_diffs)
        
        # Update scatter plot with current values only
        current_mus_np = current_mus.detach().cpu().numpy()
        target_mus_np = target_mus.cpu().numpy()
        scatter.set_offsets(np.c_[target_mus_np, current_mus_np])
        
        # Update perfect prediction line
        mu_min = min(current_mus_np.min(), target_mus_np.min())
        mu_max = max(current_mus_np.max(), target_mus_np.max())
        perfect_line.set_xdata([mu_min, mu_max])
        perfect_line.set_ydata([mu_min, mu_max])
        
        for ax in (ax1, ax2, ax3):
            ax.relim()
            ax.autoscale_view()
        
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        return mu_diff

    return update_plot


def plot_losses_from_excel(excel_path, lable='Training FNO Loss'):
    """
    Plots the training and validation losses from an Excel file, using a logarithmic scale for the y-axis.

    Args:
    excel_path (str): Path to the Excel file containing the loss data.
    """
    # Read the data from Excel
    df = pd.read_excel(excel_path, engine='openpyxl', index_col=0)
    
    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.plot(df['Training Data Loss'], label='Training Data Loss')
    plt.plot(df['Training IG Loss'],  label='Training IG Loss')
    plt.plot(df['Validation Loss'],   label='Validation Loss')
    
    # Adding titles and labels
    plt.title(f'Losses Over Epochs\n{lable}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Show the plot
    plt.grid(True)
    plt.show()

def generate_batch_parameters(grf_generator, param_ranges):
    """
    Generate batches of parameters using GaussianRandomFieldGenerator.

    :param grf_generator: Instance of GaussianRandomFieldGenerator.
    :param param_ranges: List of tuples (A, B) representing the range for each parameter.
    :return: Tensor of shape [n_batch, n_params] with generated parameters.
    """
    parameters = [grf_generator.generate(A=param_range[0], B=param_range[1]) for param_range in param_ranges]
    return torch.tensor(np.stack(parameters, axis=1), dtype=torch.float32)



# def loss_live_plot(losses_dict, figsize=(7,5)):
#     '''
#     Example usage:
#     losses_dict = {'Training FNO Loss': train_fnolosses, 'Training IG Loss': train_iglosses, 
#                    'Training ODE Loss': train_odelosses, 'Validation Loss': val_losses}
#     if ep%10 == 0:
#         loss_live_plot(losses_dict)  # Update the live plot after each epoch

#     '''
#     clear_output(wait=True)
#     plt.figure(figsize=figsize)
#     for label, data in losses_dict.items():
#         plt.plot(data, label=label)
#     plt.title('Training and Validation Losses Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def loss_live_plot(losses_dict, figsize=(6, 6)):
    '''
    Plots both the raw and logarithmic base 10 of the training and validation losses.

    Example usage:
    losses_dict = {'Training FNO Loss': train_fnolosses, 'Training IG Loss': train_iglosses, 
                   'Training ODE Loss': train_odelosses, 'Validation Loss': val_losses}
    if ep % 10 == 0:
        loss_live_plot(losses_dict)  # Update the live plot after each epoch
    '''
    clear_output(wait=True)
    plt.figure(figsize=figsize)

    # Plot raw data
    plt.subplot(2, 1, 1)  # Two rows, one column, first subplot
    for label, data in losses_dict.items():
        plt.plot(data, label=label)
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot log10 data
    plt.subplot(2, 1, 2)  # Two rows, one column, second subplot
    for label, data in losses_dict.items():
        if data:  # Ensure data is not empty
            log_data = np.log10(data)
            plt.plot(log_data, label=label)
    plt.title('Log10 of Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Log10(Loss)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()



def nse(u_true, u_prediction):
    # # Example usage
    # true_dict = {'Batch1': torch.randn(50, 50), 'Batch2': torch.randn(50, 50), 'Batch3': torch.randn(50, 50)}
    # pred_dict = {'Batch1': torch.randn(50, 50), 'Batch2': torch.randn(50, 50), 'Batch3': torch.randn(50, 50)}

    numerator = torch.sum((u_true - u_prediction)**2, dim=1)
    denominator = torch.sum((u_true - torch.mean(u_true, dim=1, keepdim=True))**2, dim=1)
    nse_values = 1 - numerator / denominator
    return nse_values

def nse_cdf_plot(true_dict, pred_dict):
    # Iterate over keys in true_dict
    for key in true_dict.keys():
        # Compute NSE values
        nse_values = nse(true_dict[key], pred_dict[key])
        
        # Sort NSE values
        sorted_nse = torch.sort(nse_values)[0]
        
        # Calculate CDF
        cdf = np.arange(1, len(sorted_nse) + 1) / len(sorted_nse)
        
        # Plot NSE-CDF curve
        plt.plot(sorted_nse, cdf, marker='o', linestyle='-', label=key)
    
    plt.xlabel('NSE')
    plt.ylabel('CDF')
    plt.title('NSE-CDF Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

# class LearnableMultiGradientLossMixer(nn.Module):
#     def __init__(self, num_losses, use_mgda=True, eps=1e-5):
#         super().__init__()
#         self.num_losses = num_losses
#         self.use_mgda = use_mgda
#         self.eps = eps
        
#         self.log_scales = nn.Parameter(torch.zeros(num_losses))
        
#         if use_mgda:
#             self.mgda_weights = nn.Parameter(torch.ones(num_losses) / num_losses)

#     def compute_mgda_weights(self, grads):
#         valid_grads = [g for g in grads if g is not None]
#         if not valid_grads:
#             return torch.ones(self.num_losses, device=self.log_scales.device) / self.num_losses
        
#         G = torch.stack([g.view(-1) for g in valid_grads])
#         GGT = torch.mm(G, G.t())
#         GGT = GGT + self.eps * torch.eye(len(valid_grads), device=GGT.device)
        
#         weights = self._solve_qp(GGT)
        
#         full_weights = torch.zeros(self.num_losses, device=weights.device)
#         valid_idx = 0
#         for i in range(self.num_losses):
#             if grads[i] is not None:
#                 full_weights[i] = weights[valid_idx]
#                 valid_idx += 1
#         return full_weights

#     def _solve_qp(self, G):
#         n = G.size(0)
#         w = torch.ones(n, device=G.device) / n
#         for i in range(20):
#             Gw = torch.mv(G, w)
#             j = torch.argmin(Gw)
#             s = torch.zeros_like(w)
#             s[j] = 1
#             gamma = 2 / (2 + i)
#             w = (1 - gamma) * w + gamma * s
#         return w

#     def forward(self, losses, shared_params):
#         if len(losses) != self.num_losses:
#             raise ValueError(f"Expected {self.num_losses} losses, but got {len(losses)}")
        
#         scaled_losses = [loss * torch.exp(scale) if loss is not None else None 
#                          for loss, scale in zip(losses, self.log_scales)]
        
#         grads = [torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=True, allow_unused=True)[0] 
#                  if loss is not None and loss.requires_grad else None
#                  for loss in scaled_losses]
        
#         if self.use_mgda:
#             with torch.no_grad():
#                 mgda_weights = self.compute_mgda_weights(grads)
#             combined_weights = F.softmax(self.mgda_weights, dim=0) * mgda_weights
#         else:
#             combined_weights = F.softmax(self.mgda_weights, dim=0)
        
#         total_loss = sum(w * l for w, l in zip(combined_weights, scaled_losses) if l is not None)
        
#         return total_loss, scaled_losses, combined_weights

#     def get_scales(self):
#         return torch.exp(self.log_scales)

#     def get_weights(self):
#         return F.softmax(self.mgda_weights, dim=0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableMultiGradientLossMixer(nn.Module):
    def __init__(self, num_losses, init_weights=None):
        super().__init__()
        self.num_losses = num_losses
        
        if init_weights is None:
            init_weights = torch.ones(num_losses)
        else:
            init_weights = torch.tensor(init_weights)
        
        self.log_weights = nn.Parameter(torch.log(init_weights))

    def forward(self, losses):
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, but got {len(losses)}")
        
        weights = torch.exp(self.log_weights)
        
        weighted_losses = [w * l for w, l in zip(weights, losses) if l is not None]
        total_loss = sum(weighted_losses)
        
        return total_loss, weighted_losses, weights

    def get_weights(self):
        return torch.exp(self.log_weights)

class AdaptiveLossWeights:
    def __init__(self, num_losses, device):
        self.num_losses = num_losses
        self.device = device
        self.log_weights = torch.nn.Parameter(torch.zeros(num_losses, device=device))
        self.running_mean = torch.zeros(num_losses, device=device)
        self.running_var = torch.ones(num_losses, device=device)
        self.momentum = 0.1

    def update_statistics(self, data_loss, ig_loss_list):
        with torch.no_grad():
            losses = torch.tensor([data_loss.item()] + [loss.item() for loss in ig_loss_list], device=self.device)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * losses
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (losses - self.running_mean)**2

    def get_weights(self):
        return torch.exp(self.log_weights) / torch.exp(self.log_weights).sum()

    def combine_losses(self, data_loss, ig_loss_list):
        self.update_statistics(data_loss, ig_loss_list)
        data_loss = data_loss.to(self.device)
        ig_losses = torch.stack([loss.to(self.device) for loss in ig_loss_list])
        all_losses = torch.cat([data_loss.unsqueeze(0), ig_losses])
        
        normalized_losses = (all_losses - self.running_mean) / (self.running_var + 1e-8).sqrt()
        weights = self.get_weights()
        return (weights * normalized_losses).sum()

        

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum






class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the Root Mean Square Error (RMSE) between predictions and targets.
        
        Parameters:
        - predictions: PyTorch tensor of predicted values
        - targets: PyTorch tensor of actual values
        
        Returns:
        - RMSE as a PyTorch tensor
        """
        mse = torch.mean((predictions - targets) ** 2)
        return torch.sqrt(mse)




class MeanRelativeAbsoluteError(nn.Module):
    def __init__(self):
        super(MeanRelativeAbsoluteError, self).__init__()

    def forward(self, u_pred, u_real):
        """
        Forward method to calculate the Mean Relative Absolute Error.

        Args:
        u_pred (torch.Tensor): Predicted values.
        u_real (torch.Tensor): Actual values.

        Returns:
        torch.Tensor: Mean Relative Absolute Error.
        """
        # Calculate absolute differences
        abs_diff = (u_pred - u_real)

        # Avoid division by zero
        u_real_safe = torch.where(u_real == 0, torch.ones_like(u_real), u_real)

        # Calculate relative errors
        relative_errors = torch.abs(abs_diff / u_real_safe)

        # Calculate mean relative absolute error
        mrae = torch.mean(relative_errors ** 2)

        return mrae





class MinMaxScaler:
    def __init__(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def scale(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def descale(self, data):
        return data * (self.max - self.min) + self.min

    def get_min_max(self):
        return self.min, self.max




class GaussianRandomFieldGenerator:
    def __init__(self, alpha, tau, N):
        """
        Initialize the generator with parameters for the Gaussian Random Field (GRF).
        :param alpha: Control parameter for the decay of the correlation.
        :param tau: Scale parameter for the correlation.
        :param N: Number of points in the domain.
        """
        self.alpha = alpha
        self.tau = tau
        self.N = N

    def generate(self, A, B):
        """
        Generate a Gaussian Random Field (GRF) using the Karhunen-Loève expansion.
        :param A: Lower bound of the normalized GRF range.
        :param B: Upper bound of the normalized GRF range.
        :return: An array of size N with the generated GRF values.
        """
        # Random variables in KL expansion
        xi = np.random.normal(0, 1, self.N)

        # Define the (square root of) eigenvalues of the covariance operator
        K = np.arange(self.N)
        coef = (self.tau**(self.alpha - 1) * (np.pi**2 * K**2 + self.tau**2)**(-self.alpha / 2))

        # Construct the KL coefficients
        L = self.N * coef * xi
        L[0] = 0  # The first coefficient is set to 0 for normalization

        # Inverse Discrete Cosine Transform
        U = idct(L, norm='ortho')

        # Normalize U to range [A, B]
        U_min = np.min(U)
        U_max = np.max(U)
        U_scaled = ((U - U_min) / (U_max - U_min)) * (B - A) + A
        return U_scaled

def pick_rows(w0, l=1):
    if w0.shape[0] < l:
        if w0.shape[0] == 0:
            return None, w0
        selected = w0
        updated_w0 = np.array([]).reshape(0, w0.shape[1])
    else:
        selected_indices = np.random.choice(w0.shape[0], size=l, replace=False)
        selected = w0[selected_indices]
        updated_w0 = np.delete(w0, selected_indices, axis=0)

    return selected, updated_w0

# prepare dataset for PDEs with 2 dependent variables
def prepare(Hu, Hv, T, T_in, S):
    # Extract initial conditions and target outputs for Hu and Hv
    train_a_u = Hu[..., :T_in]  # Initial conditions for Hu
    train_u_u = Hu[..., T_in:T + T_in]  # Target outputs for Hu

    train_a_v = Hv[..., :T_in]  # Initial conditions for Hv
    train_u_v = Hv[..., T_in:T + T_in]  # Target outputs for Hv

    # Assert statements to check dimensions
    assert (S == train_u_u.shape[-2])
    assert (T == train_u_u.shape[-1])
    assert (S == train_u_v.shape[-2])
    assert (T == train_u_v.shape[-1])

    # Reshape and repeat the initial conditions to match target outputs' shape
    train_a_u = train_a_u.reshape(1, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    train_a_v = train_a_v.reshape(1, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

    # Create a DataLoader with the processed tensors
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            train_a_u.to(dtype=torch.float),
            train_a_v.to(dtype=torch.float),
            train_u_u.to(dtype=torch.float),
            train_u_v.to(dtype=torch.float)
        )
    )

    return train_loader

# function to prepare data for 1d problems with 2 states
def prepare_data_1d_2s(solutions_u, solutions_v, T_in, T_out):
    """
    Convert numpy arrays to tensors, split and reshape the solution tensors into two parts based on T_in and T_out.
    The input tensors are reshaped to [batch_size, T_out, T_in].

    :param solutions_u: Tensor of shape [batch_size, T_in + T_out] for u.
    :param solutions_v: Tensor of shape [batch_size, T_in + T_out] for v.
    :param T_in: The size of the input sequence.
    :param T_out: The size of the output sequence.
    :return: Four tensors, two of shape [batch_size, T_in] and two of shape [batch_size, T_out, T_in].
    """

    solutions_u_in = solutions_u[:, :T_in]
    solutions_u_out = solutions_u[:, T_in:T_in + T_out]

    solutions_v_in = solutions_v[:, :T_in]
    solutions_v_out = solutions_v[:, T_in:T_in + T_out]

    # Reshape and repeat the input tensors
    solutions_u_in = solutions_u_in.unsqueeze(1).repeat(1, T_out, 1)
    solutions_v_in = solutions_v_in.unsqueeze(1).repeat(1, T_out, 1)
    return solutions_u_in, solutions_v_in, solutions_u_out, solutions_v_out



# prepare dataset for PDEs with 3 dependent variables
import torch
import torch.utils.data

def prepare3v(Hh, Hu, Hv, T, T_in, S):
    nb = Hh.shape[0]
    train_a_h = Hh[..., :T_in]
    train_u_h = Hh[..., T_in:T + T_in]

    train_a_u = Hu[..., :T_in]
    train_u_u = Hu[..., T_in:T + T_in]

    train_a_v = Hv[..., :T_in]
    train_u_v = Hv[..., T_in:T + T_in]

    assert (S == train_u_h.shape[-2])
    assert (T == train_u_h.shape[-1])
    assert (S == train_u_u.shape[-2])
    assert (T == train_u_u.shape[-1])
    assert (S == train_u_v.shape[-2])
    assert (T == train_u_v.shape[-1])

    train_a_h = train_a_h.reshape(nb, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    train_a_u = train_a_u.reshape(nb, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    train_a_v = train_a_v.reshape(nb, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
    #
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            train_a_h.to(dtype=torch.float),
            train_a_u.to(dtype=torch.float),
            train_a_v.to(dtype=torch.float),
            train_u_h.to(dtype=torch.float),
            train_u_u.to(dtype=torch.float),
            train_u_v.to(dtype=torch.float)
        )
    )
    return train_loader
    # return train_a_h.to(dtype=torch.float), train_a_u.to(dtype=torch.float), train_a_v.to(dtype=torch.float), train_u_h.to(dtype=torch.float), train_u_u.to(dtype=torch.float), train_u_v.to(dtype=torch.float)

# customize loss
class Lp_Loss(nn.Module):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(Lp_Loss, self).__init__()
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def forward(self, x, y):
        num_examples = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)

        # Absolute Lp loss
        abs_diff = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        abs_loss = (h ** (self.d / self.p)) * abs_diff

        # Relative Lp loss
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        rel_loss = abs_diff / y_norms

        # Combine losses
        combined_loss = abs_loss + rel_loss

        if self.reduction:
            if self.size_average:
                return torch.mean(combined_loss)
            else:
                return torch.sum(combined_loss)

        return combined_loss


def split_tensor(w, a_percent, b_percent):
    """
    Split a tensor into three parts based on given percentages.

    Args:
    w (Tensor): The input tensor with shape [N, 2].
    a_percent (float): The percentage of the tensor to be in the first split.
    b_percent (float): The percentage of the tensor to be in the second split.

    Returns:
    Tuple[Tensor, Tensor, Tensor]: Three tensors representing the splits.
    """
    N = w.size(0)
    a_count = int(N * a_percent)
    b_count = int(N * b_percent)

    # Shuffle indices to select random elements
    indices = torch.randperm(N)

    a_indices = indices[:a_count]
    b_indices = indices[a_count:a_count + b_count]
    remaining_indices = indices[a_count + b_count:]

    return w[a_indices], w[b_indices], w[remaining_indices]

def weighted_average(*values):
    epsilon = 1e-6
    inv_values = [1 / (value + epsilon) for value in values]

    # Calculate the total of the inverse values
    total_inv = sum(inv_values)

    # Normalize the weights
    weights = [inv_value / total_inv for inv_value in inv_values]

    # Calculate weighted average
    weighted_avg = sum(weight * value for weight, value in zip(weights, values))
    return weighted_avg

def get_unique_filename(base_path):
    """
    Returns a unique filename by appending a suffix if the file already exists.
    The suffix is an incrementing number: _1, _2, _3, etc.
    Checks for existence with the '.yml' extension but returns the filename without it.
    """
    counter = 1
    unique_path = base_path
    while os.path.exists(unique_path + '.yml'):
        unique_path = f"{base_path}_{counter}"
        counter += 1
    return unique_path

# Function to shuffle columns of a tensor independently
def shuffle_tensor_cols(tensor):
    """
    Shuffle the columns of a tensor independently.

    :param tensor: A 2D tensor of shape [n_batch, n_cols].
    :return: A new tensor with shuffled columns.
    """
    shuffled_tensor = tensor.clone()
    n_batch, n_cols = tensor.shape

    # Shuffle each column independently
    for i in range(n_cols):
        shuffled_tensor[:, i] = tensor[torch.randperm(n_batch), i]

    return shuffled_tensor


# function to stacks a list of tensors of shape [nbatch, nx, ny] along a new time dimension to form a tensor of shape [nbatch, nx, ny, nt].
def stack_tensors(tensor_list):
    """
    Stacks a list of tensors of shape [nbatch, nx, ny] along a new time dimension to form a tensor of shape [nbatch, nx, ny, nt].

    Parameters:
    - tensor_list (list of torch.Tensor): A list of 'nt' tensors, each with shape [nbatch, nx, ny]

    Returns:
    - torch.Tensor: A single tensor with shape [nbatch, nx, ny, nt]
    """
    # Stack the tensors along a new dimension (at the end), resulting in shape [nbatch, nx, ny, nt]
    stacked_tensor = torch.stack(tensor_list, dim=-1)

    return stacked_tensor
