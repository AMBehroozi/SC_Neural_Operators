import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedGraphTemporalFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_modes_space=16, num_modes_time=4, k_neighbors=8, scaling=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes_space = num_modes_space
        self.num_modes_time = num_modes_time
        self.k = k_neighbors
        
        if scaling:
            self.scale = 1 / (in_channels * out_channels)
        else:
            self.scale = 1
            
        self.weights = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, 
                num_modes_space, num_modes_time, 
                dtype=torch.cfloat
            )
        )
        
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixt,ioxt->boxt", input, weights)
        
    def graph_laplacian(self, input_features):
        x = input_features.permute(0, 2, 1)
        D = torch.cdist(x, x)
        k = min(self.k, x.size(1) - 1)
        
        _, idx = torch.topk(-D, k=k, dim=-1)
        sigma = D.mean(dim=[-2, -1], keepdim=True)
        
        A = torch.zeros_like(D)
        batch_idx = torch.arange(x.shape[0]).view(-1, 1, 1)
        point_idx = torch.arange(x.shape[1]).view(1, -1, 1)
        A[batch_idx, point_idx, idx] = torch.exp(-D[batch_idx, point_idx, idx] / (sigma**2))
        
        A = 0.5 * (A + A.transpose(-2, -1))
        D = torch.diag_embed(A.sum(dim=-1))
        L = D - A
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.diagonal(D, dim1=-2, dim2=-1) + 1e-6))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        
        return L_norm
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        L = self.graph_laplacian(x[..., 0])
        eigvals, eigvecs = torch.linalg.eigh(L)
        idx = torch.argsort(eigvals, dim=-1)[..., :self.num_modes_space]
        basis = torch.gather(eigvecs, -1, idx.unsqueeze(-2).expand(-1, eigvecs.size(-2), -1))
        basis = basis.to(dtype=torch.cfloat)
        
        x_ft_space = torch.einsum('bnk,bcnt->bckt', basis.conj(), x.to(torch.cfloat))
        x_ft_space = x_ft_space.real
        
        x_ft = torch.fft.rfft(x_ft_space, dim=-1)
        
        out_ft = torch.zeros(
            batchsize, self.out_channels, 
            self.num_modes_space, x_ft.size(-1), 
            dtype=torch.cfloat, device=x_ft.device
        )
        
        out_ft[..., :self.num_modes_time] = \
            self.compl_mul2d(
                x_ft[..., :self.num_modes_time],
                self.weights
            )
        
        out_space = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        
        basis = basis.real
        out = torch.einsum('bnk,bckt->bcnt', basis, out_space)
        
        return out

class GFNO(nn.Module):
    def __init__(self, T_in, T_out, modes_space, modes_time, width, num_features):
        super(GFNO, self).__init__()
        self.modes_space = modes_space
        self.modes_time = modes_time
        self.width = width
        self.T_in = T_in
        self.T_out = T_out
        self.num_features = num_features
        
        self.p = nn.Linear(T_in*num_features + 3, width)
        
        self.fourier0 = BatchedGraphTemporalFourierLayer(width, width, modes_space, modes_time)
        self.fourier1 = BatchedGraphTemporalFourierLayer(width, width, modes_space, modes_time)
        self.fourier2 = BatchedGraphTemporalFourierLayer(width, width, modes_space, modes_time)
        self.fourier3 = BatchedGraphTemporalFourierLayer(width, width, modes_space, modes_time)
        
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        self.q = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, num_features)
        )
    def get_t_span(self, t_in, t_out, batch_size, device):
        """Generate future time points"""
        return torch.linspace(0, 1, t_out + t_in, device=device).expand(batch_size, -1)[..., t_in:]


    def forward(self, x, pos):
        batch_size = x.shape[0]
        device = x.device
        t = self.get_t_span(self.T_in, self.T_out, batch_size, device)
        T_out = t.shape[1]
        
        # Reshape x for T_out [B, n_point, T_out, T_in, features]
        x = x.unsqueeze(2).expand(-1, -1, T_out, -1, -1)
        x = x.reshape(batch_size, x.shape[1], T_out, -1)  # [B, n_point, T_out, T_in*features]
        
        # Expand pos and t
        pos = pos.unsqueeze(2).expand(-1, -1, T_out, -1)  # [B, n_point, T_out, 2]
        t_grid = t.unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[1], -1, -1)  # [B, n_point, T_out, 1]
        
        # Concatenate all features
        x = torch.cat([x, pos, t_grid], dim=-1)  # [B, n_point, T_out, T_in*features+3]
        
        # Initial lift
        x = self.p(x)  # [B, n_point, T_out, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, n_point, T_out]
        
        # Layer 0   x = torch.randn(batch_size, in_channels, num_points, time_steps)
        x1 = self.fourier0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Layer 1
        x1 = self.fourier1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Layer 2
        x1 = self.fourier2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Layer 3
        x1 = self.fourier3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # Project to output
        x = x.permute(0, 2, 3, 1)  # [B, n_point, T_out, width]
        x = self.q(x)  # [B, n_point, T_out, num_features]
        
        return x

def test_model():
    batch_size = 8
    n_points = 800
    T_in = 5
    T_out = 15
    num_features = 1
    modes_space = 16
    modes_time = 8
    width = 20
    
    x = torch.randn(batch_size, n_points, T_in, num_features)
    pos = torch.randn(batch_size, n_points, 2)
    t = torch.linspace(0, 1, T_out).expand(batch_size, -1)
    
    print("\nInput shapes:")
    print(f"x: {x.shape}")
    print(f"pos: {pos.shape}")
    print(f"t: {t.shape}")
    
    model = GFNO(
        T_in=T_in,
        T_out=T_out,
        modes_space=modes_space,
        modes_time=modes_time,
        width=width,
        num_features=num_features
    )
    
    output = model(x, pos)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [batch_size={batch_size}, n_points={n_points}, T_out={T_out}, num_features={num_features}]")
    
    return output

if __name__ == "__main__":
    test_model()
