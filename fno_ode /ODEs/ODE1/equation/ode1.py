import numpy as np
import torch
import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("ODEs")

from lib.DerivativeComputer import batchJacobian_AD
from torch.utils.data import TensorDataset

def compute_values(t, parameters):

    # Extract alpha, beta, gamma from the parameters tensor
    alpha = parameters[:, 0].unsqueeze(1)  # Shape [nbatch, 1]
    beta = parameters[:, 1].unsqueeze(1)   # Shape [nbatch, 1]
    gamma = parameters[:, 2].unsqueeze(1)  # Shape [nbatch, 1]

    # Compute u(t)
    u = (1/torch.pi) * (-torch.cos(alpha * torch.pi * t) + torch.sin(beta * torch.pi * t) + 1) + torch.sin(gamma * torch.pi)

    # Compute du/dalpha
    du_dalpha = t * torch.sin(alpha * torch.pi * t)

    # Compute du/dbeta
    du_dbeta = t * torch.cos(beta * torch.pi * t)

    # Compute du/dgamma
    du_dgamma = torch.pi * torch.cos(gamma * torch.pi)

    # Replicate du_dgamma across time dimension to match shape [nbatch, nt]
    du_dgamma = du_dgamma.repeat(1, t.shape[0])
    du_dp = torch.cat((du_dalpha.unsqueeze(-1), du_dbeta.unsqueeze(-1), du_dgamma.unsqueeze(-1)), dim=-1)

    return u.detach(), du_dp.detach()



def creat_dataset(t_tensor, parameters, T_in):
    # Compute u and du/dp (derivatives of u with respect to parameters)
    u, du_dp = compute_values(t_tensor, parameters)
    u_in, u_out = u[:, :T_in], u[:, T_in:]    
    dataset = TensorDataset(parameters, u_in, u_out, du_dp)
    print(f'Dataset with {parameters.shape[0]} samples is created ...!')
    return dataset


def ode_residual(U_pred, params_ab, t):
    # Extract alpha and beta from params_ab tensor
    alpha = params_ab[:, 0].unsqueeze(1)
    beta = params_ab[:, 1].unsqueeze(1)
    du_t_pred = torch.diagonal((batchJacobian_AD(U_pred[...,0], t, graphed=True, batchx=True)), dim1=-2, dim2=-1)
    # Compute the ODE right-hand side
    rhs = alpha * torch.sin(alpha * torch.tensor(np.pi) * t) + beta * torch.cos(beta * torch.tensor(np.pi) * t)
    
    # Compute the residual
    residual = du_t_pred - rhs
    return residual