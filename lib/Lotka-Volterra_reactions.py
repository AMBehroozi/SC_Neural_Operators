import time
import torch
import numpy as np
import sys
sys.path.insert(0, '../lib')
from diff2d import *
start_time = time.perf_counter()


# Initialize parameters
L = 50  # Length of the domain
Nx, Ny = 50, 50  # Number of grid points in x and y
dx = dy = L / Nx  # Spatial step size
T = 50  # Total time
dt = 0.01  # Time step
Nt = int(T / dt)  # Number of time steps
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
Du, Dv = 0.1, 0.05  # Diffusion coefficients

# Initial conditions
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
sigma = L / 4
u_initial = 10 * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (2 * sigma**2))
v_initial = np.full((Nx, Ny), 5.0)

# Convert numpy arrays to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

u_tensor = torch.zeros((Nx, Ny, Nt), dtype=torch.float32, device=device)
v_tensor = torch.zeros((Nx, Ny, Nt), dtype=torch.float32, device=device)

# Set the initial conditions
u_tensor[:, :, 0] = torch.tensor(u_initial, dtype=torch.float32, device=device)
v_tensor[:, :, 0] = torch.tensor(v_initial, dtype=torch.float32, device=device)


# Initialize the DerivativeComputer
deriv_computer = diff2d(Nx, Ny, Nt, dx, dy, dt, device=device)

# Time integration
# Time integration
for n in range(Nt - 1):
    # Compute derivatives for the current time step
    du_dx, du_dy, du_dxx, du_dyy = deriv_computer.compute_derivatives(u_tensor[:, :, n])
    dv_dx, dv_dy, dv_dxx, dv_dyy = deriv_computer.compute_derivatives(v_tensor[:, :, n])

    # Lotka-Volterra reactions with diffusion
    dudt = Du * (du_dxx + du_dyy) + alpha * u_tensor[:, :, n] - beta * u_tensor[:, :, n] * v_tensor[:, :, n]
    dvdt = Dv * (dv_dxx + dv_dyy) + delta * u_tensor[:, :, n] * v_tensor[:, :, n] - gamma * v_tensor[:, :, n]

    # Update u and v for the next time step
    u_next = u_tensor[:, :, n] + dudt * dt
    v_next = v_tensor[:, :, n] + dvdt * dt

    # Apply Zero Flux (Neumann) Boundary Conditions
    # u_next[0, :], u_next[-1, :], u_next[:, 0], u_next[:, -1] = u_next[1, :], u_next[-2, :], u_next[:, 1], u_next[:, -2]
    # v_next[0, :], v_next[-1, :], v_next[:, 0], v_next[:, -1] = v_next[1, :], v_next[-2, :], v_next[:, 1], v_next[:, -2]

    u_next[0, :], u_next[-1, :], u_next[:, 0], u_next[:, -1] = 0.0, 0.0, 0.0, 0.0
    v_next[0, :], v_next[-1, :], v_next[:, 0], v_next[:, -1] = 0.0, 0.0, 0.0, 0.0


    # Store the results for the next time step
    u_tensor[:, :, n + 1] = u_next
    v_tensor[:, :, n + 1] = v_next



# Convert the final results back to numpy, if needed
u_final = u_tensor.cpu().numpy()
v_final = v_tensor.cpu().numpy()

# Post-processing or visualization
# ...



end_time = time.perf_counter()
print(end_time - start_time)
