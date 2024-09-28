import torch
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split, dataset
import argparse
import sourcedefender
import os

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from lib.util import MHPI
from fno_pde.PDEs.PDE1.pde1 import create_and_save_dataset

# %%
MHPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n')
# PATH = "fno_pde/PDEs/PDE1/"
PATH = "../"

# %%
Sample_number = 1300
train_size = 1000
eval_size = 150
test_size = Sample_number - train_size - eval_size
dataset_segment_size = 100



t0 = 0.0
t_end = torch.pi
steps_solve = 210
step_save = 30

L = 1.0
dx = 0.05
nx = int(L / dx)

# %%
# Initialize GaussianRandomFieldGenerator
# c, alpha, beta, gamma, omega, mu
c_range     = (0.00, 0.25)
alpha_range = (0.00, 0.10)  
beta_range  = (0.00, 0.25)  
gamma_range = (0.00, 0.25)  
omega_range = (0.00, 0.25)
mu_range =    (0.00, 0.50)


# Sample uniformly within each specified range.
c = torch.rand(Sample_number, 1, device=device) * (c_range[1] - c_range[0]) + c_range[0]
alpha = torch.rand(Sample_number, 1, device=device) * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
beta = torch.rand(Sample_number, 1, device=device) * (beta_range[1] - beta_range[0]) + beta_range[0]
gamma = torch.rand(Sample_number, 1, device=device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
omega = torch.rand(Sample_number, 1, device=device) * (omega_range[1] - omega_range[0]) + omega_range[0]
mu = torch.rand(Sample_number, 1, device=device) * (mu_range[1] - mu_range[0]) + mu_range[0]
parameters = torch.cat((c, alpha, beta, gamma, omega, mu), dim=-1)
parameters.requires_grad_(True)

# %%
dataset = create_and_save_dataset(t_end, steps_solve, step_save, device, parameters, dataset_segment_size, dx, nx)
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size]) # Split the dataset

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


torch.save(train_dataset, PATH + 'datasets/train_dataset.pt')
torch.save(eval_dataset, PATH +  'datasets/eval_dataset.pt')
torch.save(test_loader, PATH +  'datasets/test_loader.pt')
# %%