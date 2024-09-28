import torch

class diff2d:
    def __init__(self, Nx, Ny, Nt, hx, hy, ht, device=None):
        self.Nx, self.Ny, self.Nt = Nx, Ny, Nt
        self.hx, self.hy, self.ht = hx, hy, ht
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute the differentiation matrices
        self.Dx = self._create_diff_matrix(Nx, hx)
        self.Dy = self._create_diff_matrix(Ny, hy)
        self.Dt = self._create_diff_matrix(Nt, ht)

    def _create_diff_matrix(self, N, h):
        D = torch.zeros((N, N), device=self.device)
        D[0, :5] = torch.tensor([-25, 48, -36, 16, -3], device=self.device)
        D[1, 1:6] = torch.tensor([-25, 48, -36, 16, -3], device=self.device)
        for i in range(2, N - 2):
            D[i, i - 2:i + 3] = torch.tensor([1, -8, 0, 8, -1], device=self.device)
        D[-2, -6:-1] = torch.tensor([3, -16, 36, -48, 25], device=self.device)
        D[-1, -5:] = torch.tensor([3, -16, 36, -48, 25], device=self.device)
        D = D / (12 * h)
        return D

    def compute_derivatives(self, u):
        # Ensure u is 2D
        if u.ndim == 2:
            # Compute spatial derivatives for 2D input
            u_dx = torch.matmul(u, self.Dx.t())
            u_dy = torch.matmul(u, self.Dy.t())

            u_dxx = torch.matmul(u_dx, self.Dx.t())
            u_dyy = torch.matmul(u_dy, self.Dy.t())

            return u_dx, u_dy, u_dxx, u_dyy
        else:
            raise ValueError("Input tensor must be 2D")
