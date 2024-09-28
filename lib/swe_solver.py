import torch
class SWE_Solver:
    def __init__(self,
                 device,              # Device to use (e.g., 'cuda' or 'cpu')
                 L_x=1E+6,            # Length of domain in x-direction
                 L_y=1E+6,            # Length of domain in y-direction
                 g=9.81,              # Acceleration of gravity [m/s^2]
                 H=100,               # Depth of fluid [m]
                 f_0=1E-4,            # Fixed part of Coriolis parameter [1/s]
                 landa=2E-11,         # Gradient of Coriolis parameter [1/ms]
                 rho_0=1024.0,        # Density of fluid [kg/m^3]
                 tau_0=0.1,           # Amplitude of wind stress [kg/ms^2]
                 use_coriolis=True,   # True if you want Coriolis force
                 use_friction=False,  # True if you want bottom friction
                 use_wind=False,      # True if you want wind stress
                 use_landa=True,      # True if you want variation in Coriolis
                 N_x=150,             # Number of grid points in x-direction
                 N_y=150,             # Number of grid points in y-direction
                 max_time_step=5000,  # Total number of time steps in simulation
                 sample_interval=5000,# How often to sample for time series
                 anim_interval=25     # How often to sample for time series
                 ):
        self.device = torch.device(device)

        # Assigning the parameters
        self.L_x = torch.tensor(L_x, device=self.device)
        self.L_y = torch.tensor(L_y, device=self.device)
        self.g = torch.tensor(g, device=self.device)
        self.H = torch.tensor(H, device=self.device)
        self.f_0 = torch.tensor(f_0, device=self.device)
        self.landa = torch.tensor(landa, device=self.device)
        self.rho_0 = torch.tensor(rho_0, device=self.device)
        self.tau_0 = torch.tensor(tau_0, device=self.device)
        self.use_coriolis = use_coriolis
        self.use_friction = use_friction
        self.use_wind = use_wind
        self.use_landa = use_landa
        self.N_x = N_x
        self.N_y = N_y
        self.max_time_step = max_time_step
        self.sample_interval = sample_interval
        self.anim_interval = anim_interval

        # Computational parameters
        self.dx = self.L_x / (self.N_x - 1)
        self.dy = self.L_y / (self.N_y - 1)
        self.dt = 0.1 * min(self.dx, self.dy) / torch.sqrt(self.g * self.H)
        self.x = torch.linspace(-self.L_x / 2, self.L_x / 2, self.N_x, device=self.device)
        self.y = torch.linspace(-self.L_y / 2, self.L_y / 2, self.N_y, device=self.device)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        self.X = self.X.t()
        self.Y = self.Y.t()

    def run_batch(self, w_batch):
        n_batch = w_batch.shape[0]
        alfa_batch = w_batch[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, self.N_x, self.N_y)
        beta_batch = w_batch[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, self.N_x, self.N_y)

        # Expand X, Y to match batch size
        X_batch = self.X.expand(n_batch, -1, -1)
        Y_batch = self.Y.expand(n_batch, -1, -1)

        # Initialize state tensors for the batch
        u_n = torch.zeros(n_batch, self.N_x, self.N_y, device=self.device)
        v_n = torch.zeros(n_batch, self.N_x, self.N_y, device=self.device)
        eta_n = torch.exp(-((X_batch - alfa_batch * self.L_x) ** 2 / (2 * (0.05E+6) ** 2) +
                            (Y_batch - beta_batch * self.L_y) ** 2 / (2 * (0.05E+6) ** 2)))

        # Define friction, wind stress, and Coriolis parameters for the batch
        if self.use_friction:
            kappa_0 = 1 / (5 * 24 * 3600)
            kappa = torch.ones((n_batch, self.N_x, self.N_y), device=self.device) * kappa_0
        if self.use_wind:
            tau_x = -self.tau_0 * torch.cos(torch.pi * Y_batch / self.L_y) * 0
            tau_y = torch.zeros((n_batch, 1, self.N_x), device=self.device)

        eta_list = []
        u_list = []
        v_list = []

        for time_step in range(1, self.max_time_step + 1):
            u_np1 = u_n - self.g * self.dt / self.dx * (torch.roll(eta_n, -1, dims=1) - eta_n)
            v_np1 = v_n - self.g * self.dt / self.dy * (torch.roll(eta_n, -1, dims=2) - eta_n)

            # Add friction, wind stress, and Coriolis effects
            if self.use_friction:
                u_np1 -= self.dt * kappa * u_n
                v_np1 -= self.dt * kappa * v_n
            if self.use_wind:
                u_np1 += self.dt * tau_x / (self.rho_0 * self.H)
                v_np1 += self.dt * tau_y / (self.rho_0 * self.H)
            if self.use_coriolis:
                f = self.f_0 + self.landa * Y_batch
                alpha = self.dt * f
                beta_c = alpha ** 2 / 4
                u_np1 = (u_np1 - beta_c * u_n + alpha * v_n) / (1 + beta_c)
                v_np1 = (v_np1 - beta_c * v_n - alpha * u_n) / (1 + beta_c)

            # Apply boundary conditions
            v_np1[:, :, -1] = 0.0
            u_np1[:, -1, :] = 0.0

            # Upwind scheme for eta
            h_e = torch.where(u_np1 > 0, eta_n, torch.roll(eta_n, -1, dims=1)) + self.H
            h_w = torch.where(u_np1 > 0, torch.roll(eta_n, 1, dims=1), eta_n) + self.H
            h_n = torch.where(v_np1 > 0, eta_n, torch.roll(eta_n, -1, dims=2)) + self.H
            h_s = torch.where(v_np1 > 0, torch.roll(eta_n, 1, dims=2), eta_n) + self.H

            uhwe = u_np1 * h_e - torch.roll(u_np1, 1, dims=1) * h_w
            vhns = v_np1 * h_n - torch.roll(v_np1, 1, dims=2) * h_s

            eta_np1 = eta_n - self.dt * (uhwe / self.dx + vhns / self.dy)

            # Update arrays for the next iteration
            u_n, v_n, eta_n = u_np1, v_np1, eta_np1
                        # Store eta and (u, v) every anim_interval time step for animations.
            if time_step % self.anim_interval == 0:
                eta_list.append(eta_n)
                u_list.append(u_n)
                v_list.append(v_n)

        return torch.stack(eta_list).permute(1, 2, 3, 0), torch.stack(u_list).permute(1, 2, 3, 0), torch.stack(v_list).permute(1, 2, 3, 0)
