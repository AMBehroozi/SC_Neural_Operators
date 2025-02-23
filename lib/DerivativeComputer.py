import torch
from torch import vmap


import torch

def compute_jacobian(u, t):

    """
     Parameters:
    - u (torch.Tensor): The output tensor for which the Jacobian is computed. 
                        The shape is assumed to be (nb, nx, ny, nt), where nb is the batch size, 
                        nx and ny are spatial dimensions, and nt is the dimension of the output variable.
    - t (torch.Tensor): The input tensor with respect to which the Jacobian is computed. 
                        It must have the same shape as u, (nb, nx, ny, nt), where nt is the dimension of the input variable.

    Returns:
    - torch.Tensor: The Jacobian matrix of u with respect to t. The shape of the Jacobian matrix is 
                    (nb, nx, ny, nt, nt), where the last two dimensions correspond to the partial derivatives of 
                    each element of u with respect to each element of t across the last dimension.


    The Jacobian matrix J of a vector-valued function u = f(t), where both u and t
    are vectors of dimension n, represents all possible partial derivatives of elements
    of u with respect to elements of t. Each entry J_ij of the Jacobian matrix is
    defined as the partial derivative of the ith component of u with respect to the
    jth component of t, or J_ij = ∂u_i / ∂t_j.
    
    Mathematically, if u is a function u(t) = [u_1(t), u_2(t), ..., u_n(t)]
    and t = [t_1, t_2, ..., t_n], then the Jacobian J is given by:
    
    J = | ∂u_1/∂t_1  ∂u_1/∂t_2  ...  ∂u_1/∂t_n |
        | ∂u_2/∂t_1  ∂u_2/∂t_2  ...  ∂u_2/∂t_n |
        |    ...         ...     ...     ...   |
        | ∂u_n/∂t_1  ∂u_n/∂t_2  ...  ∂u_n/∂t_n |
    
    Note:
    Given that 't' represents coordinates, the partial derivatives of any component of 'u' at 't_i'
    with respect to a different coordinate 't_j' (where i ≠ j) are mathematically independent
    and thus equal to zero. 
    Consequently, after initializing the Jacobian matrix to zero, we only need to calculate
    its diagonal elements, i.e., the derivatives of 'u' with respect to each 't_i' (the ith
    diagonal element). This simplifies the computation significantly, focusing solely on the
    direct dependencies of each 'u' component on its corresponding 't' coordinate. 
    Hence, the Jacobian effectively reduces to a simpler form where only diagonal elements are of interest for the derivative calculation of 'u' with respect to 't'.
    """                   

    # Make sure that t requires gradient
    t.requires_grad_(True)
    
    # Initialize the Jacobian to zeros
    jacobian = torch.zeros(*t.shape, *t.shape[-1:], device=t.device)
    
    # Loop over each element in the output tensor u
    for i in range(t.shape[-1]):
        # Compute the gradient of each element of u with respect to t
        # torch.autograd.grad returns a tuple of gradients for each input, we take the first
        grad = torch.autograd.grad(u[..., i], t, grad_outputs=torch.ones_like(u[..., i]),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # Assign the computed gradients to the corresponding slice of the Jacobian
        jacobian[..., i] = grad
    
    return jacobian



def compute_grads(u, x, y, t):
    """
    Computes the gradients of u with respect to t, x, and y.

    Args:
    - u (torch.Tensor): The output tensor with shape [nb, S, S, nt].
    - x (torch.Tensor): The spatial tensor x with shape [S].
    - y (torch.Tensor): The spatial tensor y with shape [S].
    - t (torch.Tensor): The time tensor with shape [nt].

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients of u with respect to t, x, and y.
    """
    nb = u.shape[0]
    nt = t.shape[0]
    nx = x.shape[0]
    ny = y.shape[0]
    
    dudx = torch.zeros_like(u.reshape(nb*ny*nt, nx))
    dudy = torch.zeros_like(u.reshape(nb*nx*nt, ny))
    dudt = torch.zeros_like(u.reshape(nb*nx*ny, nt))
    
        # Compute gradient w.r.t. x
    for i in range(x.size(0)):
        grads = torch.autograd.grad(u.reshape(nb*ny*nt, nx)[..., i].sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            dudx[..., i] = grads[i]

    # Compute gradient w.r.t. y
    for i in range(y.size(0)):
        grads = torch.autograd.grad(u.reshape(nb*nx*nt, ny)[..., i].sum(), y, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            dudy[..., i] = grads[i]

    # Compute gradient w.r.t. t
    for i in range(t.size(0)):
        grads = torch.autograd.grad(u.reshape(nb*nx*ny, nt)[..., i].sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            dudt[..., i] = grads[i]
    
    return dudx.reshape(nb, nx, ny, nt), dudy.reshape(nb, nx, ny, nt), dudt.reshape(nb, nx, ny, nt)




def differentator(h, co_v):
    nb, S, T_out = h.shape[0], h.shape[1], h.shape[-1]
    diff = torch.zeros_like(h)
    for i in range(S):
        for j in range(S):
            
            for k in range(T_out):
                ddd = torch.zeros(nb, 1, T_out)
                ddd = batchJacobian_AD(h[:, i, j, k], co_v)
                diff[:, i, j, k] = ddd[:, 0, k]
    
    return diff




def compute_gradients_wrt_alfa(u, v, alfa):
    alfa.requires_grad_(True)


    # Initialize gradient tensors
    gradients_u = torch.zeros(u.size(0), u.size(1), alfa.size(1), device=alfa.device)
    gradients_v = torch.zeros(v.size(0), v.size(1), alfa.size(1), device=alfa.device)

    # Iterate over each time step
    for i in range(u.size(1)):
        if alfa.grad is not None:
            alfa.grad.zero_()

        # Compute gradients for u at time step i
        # We sum over the batch dimension to get a scalar for backward()
        (u[:, i] * torch.ones_like(u[:, i])).sum().backward(retain_graph=True)
        gradients_u[:, i, :] = alfa.grad.clone()
        alfa.grad.zero_()

        # Compute gradients for v at time step i
        (v[:, i] * torch.ones_like(v[:, i])).sum().backward(retain_graph=True)
        gradients_v[:, i, :] = alfa.grad.clone()

    return gradients_u, gradients_v




class DerivativeComputer:
    def __init__(self, t, device):
        self.device = device
        self.D = self._build_derivative_matrix(t)

    def _build_derivative_matrix(self, t):
        nt = len(t)
        dt = t[1] - t[0]  # Assuming evenly spaced t

        # Initialize the derivative matrix
        D = torch.zeros((nt, nt), device=self.device)

        # Apply second-order one-sided differences at the boundaries
        D[0, :3] = torch.tensor([-3, 4, -1], device=self.device) / (2 * dt)
        D[1, :3] = torch.tensor([-1, 0, 1], device=self.device) / (2 * dt)
        D[-2, -3:] = torch.tensor([-1, 0, 1], device=self.device) / (2 * dt)
        D[-1, -3:] = torch.tensor([1, -4, 3], device=self.device) / (2 * dt)

        # Apply fourth-order central differences in the interior
        for i in range(2, nt - 2):
            D[i, i - 2:i + 3] = torch.tensor([1, -8, 0, 8, -1], device=self.device) / (12 * dt)

        return D

    def compute_derivative(self, u):
        # Apply the derivative matrix to u
        return torch.matmul(u, self.D.t())



def compute_derivative_t(u, t, device):
    n_batch, nt = u.shape
    dt = t[1] - t[0]  # Assuming evenly spaced t

    # Build the derivative matrix
    D = torch.zeros((nt, nt)).to(device)

    # Apply second-order one-sided differences at the boundaries
    D[0, :3] = torch.tensor([-3, 4, -1]).to(device) / (2 * dt)  # First point
    D[1, :3] = torch.tensor([-1, 0, 1]).to(device) / (2 * dt)  # Second point
    D[-2, -3:] = torch.tensor([-1, 0, 1]).to(device) / (2 * dt)  # Second-last point
    D[-1, -3:] = torch.tensor([1, -4, 3]).to(device) / (2 * dt)  # Last point

    # Apply fourth-order central differences in the interior
    for i in range(2, nt - 2):
        D[i, i - 2:i + 3] = torch.tensor([1, -8, 0, 8, -1]).to(device) / (12 * dt)

    # Apply the derivative matrix to u
    du_dt = torch.matmul(u, D.t())

    return du_dt


def batchJacobian_AD(y, x, graphed=False, batchx=True):
    # extract the jacobian dy/dx for multi-column y output (and with minibatch)
    # compared to the scalar version above, this version will call grad() ny times in parallel and store outputs in a tensor matrix
    # if batchx=True:
    #    y: [nb, ny]; x: [nb, nx]. x could also be a tuple or list of tensors. ---> Jac: [nb,ny,nx] # assuming batch elements have nothing to do with each other
    # if batchx=Flase:
    #    y: [nb, ny]; x: [nx]. x could also be a tuple or list of tensors. --> Jac: [nb,ny,nx]
    # permute and view your y to be of the above format.
    # AD jacobian is not free and may end up costing lots of time
    # output: Jacobian [nb, ny, nx] # will squeeze after the calculation
    # relying on the fact that the minibatch has nothing to do with each other!
    # if they do, i.e, they come from different time steps of a simulation, you need to put them in second dim in y!
    # view or reshape your x and y to be in this format if they are not!
    # pay attention, this operation could be expensive.

    import sourcedefender
    import ad_jac
    import torch
    return ad_jac.batchJacobian_AD_enc(y, x, graphed=True, batchx=True)


class diff3d:
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
        u_dx = u.permute(1, 2, 0).reshape(-1, self.Nx)
        du_dx = torch.matmul(u_dx, self.Dx.t()).reshape(self.Ny, self.Nt, self.Nx).permute(2, 0, 1)

        u_dy = u.permute(0, 2, 1).reshape(-1, self.Ny)
        du_dy = torch.matmul(u_dy, self.Dy.t()).reshape(self.Nx, self.Nt, self.Ny).permute(0, 2, 1)

        u_dt = u.reshape(-1, self.Nt)
        du_dt = torch.matmul(u_dt, self.Dt.t()).reshape(self.Nx, self.Ny, self.Nt)

        u_dxx = du_dx.permute(1, 2, 0).reshape(-1, self.Nx)
        du_dxx = torch.matmul(u_dxx, self.Dx.t()).reshape(self.Ny, self.Nt, self.Nx).permute(2, 0, 1)

        u_dyy = du_dy.permute(0, 2, 1).reshape(-1, self.Ny)
        du_dyy = torch.matmul(u_dyy, self.Dy.t()).reshape(self.Nx, self.Nt, self.Ny).permute(0, 2, 1)

        u_dtt = du_dt.reshape(-1, self.Nt)
        du_dtt = torch.matmul(u_dtt, self.Dt.t()).reshape(self.Nx, self.Ny, self.Nt)

        return du_dx, du_dy, du_dt, du_dxx, du_dyy, du_dtt



class diff2d:
    """
    A class for computing first-order spatial derivatives of 2D tensors using finite differences.

    Attributes:
    - Nx (int): Number of points in the x-direction.
    - Ny (int): Number of points in the y-direction.
    - hx (float): Spatial step size in the x-direction.
    - hy (float): Spatial step size in the y-direction.
    - device (torch.device): The device on which computations are performed.
    - Dx (torch.Tensor): Differentiation matrix for the x-direction.
    - Dy (torch.Tensor): Differentiation matrix for the y-direction.
    """

    def __init__(self, Nx, Ny, hx, hy, device=None):
        """
        Initializes the diff2d object with spatial dimensions, step sizes, and the computation device.

        Parameters:
        - Nx, Ny (int): Spatial dimensions.
        - hx, hy (float): Spatial step sizes.
        - device (torch.device, optional): Computation device. Defaults to CUDA if available, else CPU.
        """
        self.Nx, self.Ny = Nx, Ny
        self.hx, self.hy = hx, hy
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Dx = self._create_diff_matrix(Nx, hx)
        self.Dy = self._create_diff_matrix(Ny, hy)

    def _create_diff_matrix(self, N, h):
        """
        Creates a differentiation matrix for finite difference computations.

        Parameters:
        - N (int): Number of points in the spatial dimension.
        - h (float): Spatial step size.

        Returns:
        - torch.Tensor: A differentiation matrix for computing derivatives.
        """
        D = torch.zeros((N, N), device=self.device)

        # Second-order forward difference at the first point
        D[0, :3] = torch.tensor([-3, 4, -1], dtype=torch.float32) / (2 * h)

        # Second-order central difference at interior points
        for i in range(1, N - 1):
            D[i, i-1:i+2] = torch.tensor([-1, 0, 1], dtype=torch.float32) / (2 * h)

        # Second-order backward difference at the last point
        D[-1, -3:] = torch.tensor([1, -4, 3], dtype=torch.float32) / (2 * h)
        return D

    def compute_derivatives(self, u):
        """
        Computes the first-order derivatives of a 2D tensor along both spatial dimensions.

        Parameters:
        - u (torch.Tensor): The input tensor of shape [batch_size, Nx, Ny].

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The derivatives du/dx and du/dy.
        """
        # Compute derivative along x-direction
        # Permute and reshape for matrix multiplication, then apply Dx
        u_dx = u.permute(0, 2, 1).reshape(-1, self.Nx)
        du_dx = torch.matmul(u_dx, self.Dx.t()).reshape(-1, self.Ny, self.Nx).permute(0, 2, 1)

        # Compute derivative along y-direction
        # Reshape for matrix multiplication, then apply Dy
        u_dy = u.reshape(-1, self.Ny)
        du_dy = torch.matmul(u_dy, self.Dy.t()).reshape(-1, self.Nx, self.Ny)

        return du_dx, du_dy


import numpy as np

class PrecomputedDifferentiator:
    def __init__(self, Nx, Ny, Nt, hx, hy, ht, device='cpu'):
        self.device = torch.device(device)
        self.Dx = torch.tensor(self._create_diff_matrix(Nx, hx), dtype=torch.float32, device=self.device)
        self.Dy = torch.tensor(self._create_diff_matrix(Ny, hy), dtype=torch.float32, device=self.device)
        self.Dt = torch.tensor(self._create_diff_matrix(Nt, ht), dtype=torch.float32, device=self.device)
    
    @staticmethod
    def _create_diff_matrix(N, h):
        D = np.zeros((N, N))
        np.fill_diagonal(D[1:], -1/(2*h))
        np.fill_diagonal(D[:, 1:], 1/(2*h))
        D[0, :3] = [-3/(2*h), 4/(2*h), -1/(2*h)]
        D[-1, -3:] = [1/(2*h), -4/(2*h), 3/(2*h)]
        return D
    
    def compute_derivative(self, u, axis):
        if axis == 0:
            return torch.einsum('ij,bjkl->bikl', self.Dx, u)
        elif axis == 1:
            B, Ny, Nx, Nt = u.shape[0], u.shape[2], u.shape[1], u.shape[3]
            u_reshaped = u.permute(0, 2, 1, 3).reshape(B, Ny, Nx*Nt)
            dy_reshaped = torch.matmul(self.Dy, u_reshaped)
            return dy_reshaped.reshape(B, Ny, Nx, Nt).permute(0, 2, 1, 3)
        elif axis == 2:
            return torch.einsum('ij,bklj->bkli', self.Dt, u)
        else:
            raise ValueError("Invalid axis. Axis must be 0 (x), 1 (y), or 2 (t).")

