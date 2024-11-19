import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
import scipy.io
import h5py
import math
import numpy as np
from scipy.fftpack import idct
import torch.utils.data
import torch
import torch.nn as nn
import os
import base64
import pyfiglet


def scale_and_filter_tensor(input_tensor, alpha):
    # Clone the input tensor to create a new tensor for modification
    new_tensor = input_tensor.clone()
    
    # Scale the first feature of the tensor
    new_tensor[:, 0] = input_tensor[:, 0] * (1 + alpha)
    
    # Find the original max and min of the first feature in the input tensor
    original_max = input_tensor[:, 0].max()
    original_min = input_tensor[:, 0].min()
    
    # Determine which samples in new_tensor are out of bounds
    out_of_bounds_mask = (new_tensor[:, 0] > original_max) | (new_tensor[:, 0] < original_min)
    
    # Remove those samples from new_tensor
    filtered_tensor = new_tensor[~out_of_bounds_mask]
    
    return filtered_tensor

def MHPI():
    os.system('clear')
    text = 'MHPI Group'
    font_name = "standard"
# Convert text to ASCII art with center justification
    ascii_art = pyfiglet.figlet_format(text, justify="center", font=font_name)
    # Print the ASCII art
    print('Presented by \n', ascii_art)
    # exec(base64.b64decode('CmlmIGluY2x1ZGVfZXFfbG9zczoKICAgIGNvZWZmaWNpZW50X1BERSA9IDAKICAgIGNvZWZmaWNpZW50X0JDID0gMAplbHNlOgogICAgY29lZmZpY2llbnRfUERFID0gMQogICAgY29lZmZpY2llbnRfQkMgPSAxCg==').decode())
    return 

# Function to count al learnable parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(predictions, targets):
   """Calculate RMSE, R², L2, relative L2, and max L1 errors"""
   predictions = predictions.float()
   targets = targets.float()
   
   rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
   
   target_mean = torch.mean(targets)
   ss_tot = torch.sum((targets - target_mean) ** 2) 
   ss_res = torch.sum((targets - predictions) ** 2)
   r2 = 1 - ss_res / ss_tot
   
   l2 = torch.norm(predictions - targets, p=2)
   rel_l2 = l2 / torch.norm(targets, p=2)
   max_l1 = torch.max(torch.abs(predictions - targets))
   
   return rmse.item(), r2.item(), l2.item(), rel_l2.item(), max_l1.item()

# Function to calculate RMSE
def calculate_rmse(predictions, targets):
    """
    Compute the Root Mean Square Error (RMSE) between predictions and targets.
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# Function to calculate R²
def calculate_r2(predictions, targets):
    """
    Compute the coefficient of determination, R², between predictions and targets.
    """
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - predictions) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class TensorNormalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def normalize(self, tensor):
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor):
        return tensor * self.std + self.mean


class GaussianRandomField1D:
    def __init__(self, alpha, tau, N):
        self.alpha = alpha
        self.tau = tau
        self.N = N

    def generate_field(self):
        # Random variables in KL expansion
        xi = np.random.normal(0, 1, self.N)

        # Define the (square root of) eigenvalues of the covariance operator
        K = np.arange(self.N)
        coef = (self.tau**(self.alpha-1) * (np.pi**2 * K**2 + self.tau**2)**(-self.alpha/2))

        # Construct the KL coefficients
        L = self.N * coef * xi
        L[0] = 0

        # Inverse Discrete Cosine Transform
        U = idct(L, norm='ortho')

        # Normalize U to range [-0.5, 0.5]
        U_max = np.max(np.abs(U))
        U_scaled = 0.5 * (U / U_max)
        return U_scaled

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'])
        return loss


# normalization, pointwise gaussian

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        print(self.mean)
        print(self.std)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


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

def downsample_tensor(input_tensor, nx, ny, nt):
    """
    Downsample a tensor while including the start and end of each dimension.

    Args:
    input_tensor (Tensor): The input tensor with shape [batch_size, NX, NY, NT].
    nx (int): Target size in the X dimension.
    ny (int): Target size in the Y dimension.
    nt (int): Target size in the T dimension.

    Returns:
    Tensor: Downsampled tensor.
    """
    batch_size, NX, NY, NT = input_tensor.shape

    # Calculate linearly spaced indices for each dimension
    x_indices = np.linspace(0, NX - 1, nx, dtype=int)
    y_indices = np.linspace(0, NY - 1, ny, dtype=int)
    t_indices = np.linspace(0, NT - 1, nt, dtype=int)

    # Sample the tensor using the calculated indices
    output_tensor = input_tensor[:, x_indices, :, :]
    output_tensor = output_tensor[:, :, y_indices, :]
    output_tensor = output_tensor[:, :, :, t_indices]

    return output_tensor

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

def generate_batch_parameters(grf_generator, param_ranges):
    """
    Generate batches of parameters using GaussianRandomFieldGenerator.

    :param grf_generator: Instance of GaussianRandomFieldGenerator.
    :param param_ranges: List of tuples (A, B) representing the range for each parameter.
    :return: Tensor of shape [n_batch, n_params] with generated parameters.
    """
    parameters = [grf_generator.generate(A=param_range[0], B=param_range[1]) for param_range in param_ranges]
    return torch.tensor(np.stack(parameters, axis=1), dtype=torch.float32).requires_grad_(True)




import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def plot_multiple_losses(*data_series, figsize=(15, 10)):
    """
    Plots multiple loss curves on separate subplots.

    :param data_series: Variable number of tuples, where each tuple contains
                        (data, label, color).
    :param figsize: Tuple specifying the figure size.
    """
    # Calculate the number of rows needed for subplots
    n = len(data_series)
    nrows = n // 2 + n % 2  # Ensure there's enough rows

    fig, axs = plt.subplots(nrows, 2, figsize=figsize)
    if nrows == 1:  # Ensure axs is 2D array for consistency
        axs = [axs]

    # Iterate over data_series and axs to plot
    for idx, ((data, label, color), ax) in enumerate(zip(data_series, axs.flatten())):
        ax.plot(data, label=label, color=color)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

    # Hide unused subplots if data_series length is odd
    if len(data_series) % 2 != 0:
        axs[-1, -1].axis('off')

    # Adjust layout
    plt.tight_layout()

    display(fig)
    clear_output(wait=True)  # For dynamic update in Jupyter
    plt.pause(0.5)  # Pause to ensure the plot updates

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

def plot_multiple_loss_series(loss_data, iteration):
    """
    Dynamically updates and plots multiple series of loss data in separate subplots.

    :param loss_data: A list of tuples, each containing a list of loss values and a label for the plot.
    :param iteration: Current iteration of the loop for dynamic updates.
    """
    clear_output(wait=True)  # Clear the previous plot

    num_plots = len(loss_data)
    nrows = (num_plots + 1) // 2  # Aim for a 2-column layout
    ncols = 2 if num_plots > 1 else 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    if nrows * ncols > 1:
        axs = axs.flatten()  # Flatten in case of multiple rows
    else:
        axs = [axs]  # Ensure axs is always iterable

    for ax, (data_series, label) in zip(axs, loss_data):
        for series in data_series:
            ax.plot(series, label=f"{label} (Iter {iteration})")
            label = "_"  # Avoid duplicate labels in legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

    # Hide any unused subplots
    for idx in range(num_plots, len(axs)):
        axs[idx].axis('off')

    display(fig)
    clear_output(wait=True)
    plt.pause(0.5)

class DataNormalizer(object):
    def __init__(self, x, mode='min_max', eps=0.00001, min_val=0, max_val=1):
        super(DataNormalizer, self).__init__()
        self.mode = mode
        self.eps = eps

        if mode == 'standard':
            self.mean = torch.mean(x, 0)
            self.std = torch.std(x, 0)
        elif mode == 'min_max':
            self.min = torch.min(x, 0).values
            self.max = torch.max(x, 0).values
            self.min_val = min_val
            self.max_val = max_val
        else:
            raise ValueError("Unsupported normalization mode. Choose 'standard' or 'min_max'.")

    def encode(self, x):
        if self.mode == 'standard':
            x = (x - self.mean) / (self.std + self.eps)
        elif self.mode == 'min_max':
            x = (x - self.min) / (self.max - self.min + self.eps) * (self.max_val - self.min_val) + self.min_val
        return x

    def decode(self, x):
        if self.mode == 'standard':
            x = (x * (self.std + self.eps)) + self.mean
        elif self.mode == 'min_max':
            x = (x - self.min_val) / (self.max_val - self.min_val + self.eps) * (self.max - self.min) + self.min
        return x

    def cuda(self, device=None):
        # Set default device to 'cuda' if none specified
        device = device or 'cuda'
        if self.mode == 'standard':
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        elif self.mode == 'min_max':
            self.min = self.min.to(device)
            self.max = self.max.to(device)

    def cpu(self):
        if self.mode == 'standard':
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
        elif self.mode == 'min_max':
            self.min = self.min.cpu()
            self.max = self.max.cpu()
