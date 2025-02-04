import torch
import sys
sys.path.append("../")
sys.path.append("./")
from lib.DerivativeComputer import batchJacobian_AD
from torch.utils.data import TensorDataset
from torchdiffeq import odeint
from torch.utils.data import DataLoader
import sourcedefender
# from lib.batchJacobian import batchJacobian_FNO
# Define the ODE function that includes parameters


def ode_func(t, y, alpha, beta, gamma, delta, omega):
    u, u_prime = y[..., 0], y[..., 1]  # Unpack u and u'
    dudt = u_prime  # First derivative of u
    du_prime_dt = (gamma * torch.cos(torch.pi * omega * t) - delta * u_prime - alpha * t - beta * t**3)  # Second derivative of u
    return torch.stack([dudt, du_prime_dt], -1)  # Stack to make the output the same shape as y

# Solve the ODE with parameters wrapped in the function call
def solve_ode(parameters, t_tensor):
    alpha, beta, gamma, delta, omega, ic1, ic2 = parameters[..., 0], parameters[..., 1], parameters[..., 2], parameters[..., 3], parameters[..., 4], parameters[..., 5], parameters[..., 6] 
    initial_conditions = torch.stack([ic1, ic2], dim=-1)  # [u(0), u'(0)]
    
    # Wrap parameters using a lambda to create a closure
    solution = odeint(lambda t, y: ode_func(t, y, alpha, beta, gamma, delta, omega), initial_conditions, t_tensor)
    
    return solution[..., 0].unsqueeze(-1).permute(1, 0, 2)

def creat_dataset(t_tensor, parameters, T_in):
    sol = solve_ode(parameters, t_tensor)
    nb, nt, ny = sol.shape[0], sol.shape[1], sol.shape[2]
    nx = parameters.shape[1]
    du_dp = torch.zeros(nb, nt, nx, ny)
    
    for i in range(ny):
        du_dp[..., i] = batchJacobian_AD(sol[..., i], parameters, graphed=False, batchx=True)
    
    dataset = TensorDataset(parameters, sol[..., :T_in, :].detach(), sol[..., T_in:, :].detach(), du_dp.detach())
    return dataset
        
    
'''
# Example usage
nbatch = 16
t_span = 2 * 3.14
steps = 100
t_tensor = torch.linspace(0, t_span, steps)

# Directly define each parameter with the specific range
alpha = 0.02 + 0.04 * torch.rand(nbatch)  # Range [0.02, 0.06]
beta = 0.01 + 0.02 * torch.rand(nbatch)   # Range [0.01, 0.03]
gamma = 20 + 40 * torch.rand(nbatch)      # Range [20, 60]
delta = 0.5 + 1.0 * torch.rand(nbatch)    # Range [0.5, 1.5]
omega = 0.2 + 0.4 * torch.rand(nbatch)    # Range [0.2, 0.6]
ic1 = torch.zeros(nbatch)                 # Initial condition 1 is always 0
ic2 = torch.zeros(nbatch)                 # Initial condition 2 is also always 0
parameters = torch.cat((alpha.unsqueeze(-1), beta.unsqueeze(-1), gamma.unsqueeze(-1), delta.unsqueeze(-1), omega.unsqueeze(-1), ic1.unsqueeze(-1), ic2.unsqueeze(-1)), dim=-1)
parameters.requires_grad_(True)


dataset = creat_dataset(t_tensor, parameters, T_in=10)

'''




def ode_residual(U_pred, batch_parameters, t):
    # U_pred:           [nb, T_out, 1]
    # t:                [T_out]
    # batch_parameters: [nb, 7]
    # alpha, beta, gamma, delta, omega, ic1, ic2 
    u = U_pred[..., 0] #[nb, T_out]
    u_t = torch.diagonal((batchJacobian_AD(u, t, graphed=True, batchx=True)), dim1=-2, dim2=-1)    #[nb, T_out]
    u_tt = torch.diagonal((batchJacobian_AD(u_t, t, graphed=True, batchx=True)), dim1=-2, dim2=-1) #[nb, T_out]
    # Reshape alfa for broadcasting
    alpha = batch_parameters[:, 0].unsqueeze(1)
    beta =  batch_parameters[:, 1].unsqueeze(1)
    gamma = batch_parameters[:, 2].unsqueeze(1)
    delta = batch_parameters[:, 3].unsqueeze(1)
    omega = batch_parameters[:, 4].unsqueeze(1)
    
    # Compute residuals
    residual = delta * u_t + alpha * t + beta * t**3 - gamma * torch.cos(omega * torch.pi * t) - u_tt
    return residual
