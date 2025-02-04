import numpy as np
import torch
import sys
sys.path.append("../")
sys.path.append("./")
from lib.DerivativeComputer import batchJacobian_AD
from torch.utils.data import TensorDataset
from torchdiffeq import odeint
from torch.utils.data import DataLoader


def generalized_ode_system(U, alfa):
    # Use broadcasting for operations with alfa
    du0_dt = alfa[:, 0] * U[:, 0] - alfa[:, 1] * U[:, 0] * U[:, 1]
    du1_dt = alfa[:, 3] * U[:, 0] * U[:, 1] - alfa[:, 2] * U[:, 1]
    return torch.stack([du0_dt, du1_dt], dim=-1)


def segment_solver(t_tensor, parameters):
    parameters.requires_grad_(True)
    batch_alfa = parameters[:, :4]
    batch_initial_conditions = parameters[:, 4:]
    # Solve the ODE for the batch
    batch_solutions = odeint(lambda t, U: generalized_ode_system(U, batch_alfa), batch_initial_conditions, t_tensor).permute(1, 0, 2)

    # Compute Jacobian of state u0 w.r.t. parameters
    du0_d_alfa = batchJacobian_AD(batch_solutions[..., 0], batch_alfa, graphed=False, batchx=True)  # Shape: [nb, steps, n_params]
    du1_d_alfa = batchJacobian_AD(batch_solutions[..., 1], batch_alfa, graphed=False, batchx=True)  # Shape: [nb, steps, n_params]

    # Compute Jacobian of state u0 w.r.t. initial conditions
    du0_d_initial_condition = batchJacobian_AD(batch_solutions[..., 0], batch_initial_conditions, graphed=False, batchx=True)  # Shape: [nb, steps, n_states]
    du1_d_initial_condition = batchJacobian_AD(batch_solutions[..., 1], batch_initial_conditions, graphed=False, batchx=True)  # Shape: [nb, steps, n_states]
    # solutions_u_in, solutions_v_in, solutions_u_out, solutions_v_out = prepare_data_1d_2s(batch_solutions[..., 0], batch_solutions[..., 1], T_in, T_out)

    return batch_solutions.detach(), torch.cat((du0_d_alfa, du0_d_initial_condition.detach(), du1_d_alfa.detach(), du1_d_initial_condition.detach()), dim=-1)



def creat_dataset(t_tensor, parameters, T_in):
    # Compute u and du/dp (derivatives of u with respect to parameters)
    u, du_dp = segment_solver(t_tensor, parameters)
    u_in, u_out = u[:, :T_in, :], u[:, T_in:, :]    
    dataset = TensorDataset(parameters, u_in, u_out, du_dp)
    print(f'Dataset with {parameters.shape[0]} samples is created ...!')
    return dataset

def ode_residual(U_pred, batch_parameters, t):
    u, v = U_pred[..., 0], U_pred[..., 1]
    
    # Reshape alfa for broadcasting
    alfa_u = batch_parameters[:, 0].unsqueeze(1)  # Shape becomes [3, 1]
    alfa_v = batch_parameters[:, 1].unsqueeze(1)  # Shape becomes [3, 1]
    alfa_predator = batch_parameters[:, 2].unsqueeze(1)  # Shape becomes [3, 1]
    alfa_prey = batch_parameters[:, 3].unsqueeze(1)  # Shape becomes [3, 1]
    u_t = torch.diagonal((batchJacobian_AD(u, t, graphed=True, batchx=True)), dim1=-2, dim2=-1)
    v_t = torch.diagonal((batchJacobian_AD(v, t, graphed=True, batchx=True)), dim1=-2, dim2=-1)
    # Compute residuals
    residual_du_dt = u_t - (alfa_u * u - alfa_v * u * v)
    residual_dv_dt = v_t - (-alfa_predator * v + alfa_prey * u * v)

    # loss_ode = 0.5 * (torch.mean(residual_du_dt) + torch.mean(residual_dv_dt))

    return torch.cat((residual_du_dt.unsqueeze(-1), residual_dv_dt.unsqueeze(-1)), dim=-1)
