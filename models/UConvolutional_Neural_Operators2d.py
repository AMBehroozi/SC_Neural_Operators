import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetCNO1DTime(nn.Module):
    def __init__(self, nx, T_in, T_out, state_size, parameter_size, width, depth, kernel_size=3, unet_depth=4):
        super(UNetCNO1DTime, self).__init__()
        self.nx = nx
        self.T_in = T_in
        self.T_out = T_out
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.width = width
        self.depth = depth
        self.kernel_size = kernel_size
        self.unet_depth = unet_depth

        # Input projection
        self.fc_in = nn.Linear(state_size + parameter_size, width - 2)  # -2 to leave room for grid channels

        # U-Net Encoder
        self.encoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width if i == 0 else width * 2**i
            out_channels = width * 2**(i+1)
            self.encoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # U-Net Bottleneck
        self.bottleneck = ResidualConvBlock2D(width * 2**unet_depth, width * 2**unet_depth, kernel_size)

        # U-Net Decoder
        self.decoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width * 2**(unet_depth-i+1)
            out_channels = width * 2**(unet_depth-i-1)
            self.decoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # Additional convolutional layers with residual connections
        self.conv_layers = nn.ModuleList([
            ResidualConvBlock2D(width, width, kernel_size) for _ in range(depth)
        ])

        # Output projection
        self.fc_out = nn.Linear(width, state_size)

    def generate_grid(self, batch_size, device):
        x = torch.linspace(0, 1, self.nx, device=device)
        t = torch.linspace(0, 1, self.T_out, device=device)
        x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
        return x_grid.unsqueeze(0).repeat(batch_size, 1, 1), t_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, u0, P):
        batch_size = u0.shape[0]
        device = u0.device

        # Generate spatial-temporal grid
        x_grid, t_grid = self.generate_grid(batch_size, device)

        # Prepare input features
        P_expanded = P.unsqueeze(1).unsqueeze(2).expand(-1, self.nx, self.T_out, -1)
        u0_interp = F.interpolate(u0.permute(0, 3, 1, 2), size=(self.nx, self.T_out), mode='bilinear', align_corners=False)
        u0_interp = u0_interp.permute(0, 2, 3, 1)

        # Combine u0 and P
        inputs = torch.cat([u0_interp, P_expanded], dim=-1)

        # Apply input projection
        x = self.fc_in(inputs)

        # Add grid information as separate channels
        x = torch.cat([x, x_grid.unsqueeze(-1), t_grid.unsqueeze(-1)], dim=-1)

        # Apply U-Net
        x = x.permute(0, 3, 1, 2)  # [batch, channels, nx, T_out]
        

        # Encoder
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            encoder_outputs.append(x)
            x = F.avg_pool2d(x, (2, 1))  # Only pool in spatial dimension

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=(2, 1), mode='bilinear', align_corners=True)
            
            encoder_output = encoder_outputs[-i-1]
            
            # Pad the smaller tensor to match the larger one
            if x.size(2) > encoder_output.size(2):
                diff = x.size(2) - encoder_output.size(2)
                encoder_output = F.pad(encoder_output, (0, 0, 0, diff))
            elif x.size(2) < encoder_output.size(2):
                diff = encoder_output.size(2) - x.size(2)
                x = F.pad(x, (0, 0, 0, diff))
            
            if x.size(3) > encoder_output.size(3):
                diff = x.size(3) - encoder_output.size(3)
                encoder_output = F.pad(encoder_output, (0, diff))
            elif x.size(3) < encoder_output.size(3):
                diff = encoder_output.size(3) - x.size(3)
                x = F.pad(x, (0, diff))
            
            x = torch.cat([x, encoder_output], dim=1)
            x = decoder_layer(x)

        # Apply additional convolutional layers with residual connections
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)

        # Apply output projection
        x = x.permute(0, 2, 3, 1)  # [batch, nx, T_out, channels]
        x = self.fc_out(x)

        return x


class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConvBlock2D, self).__init__()
        padding = (kernel_size // 2, kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.GELU()
        
        # Add a 1x1 convolution for residual connection if channel dimensions don't match
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

# # Example usage
# if __name__ == "__main__":
#     batch_size = 1
#     nx = 20
#     T_in, T_out = 5, 25
#     state_size = 1
#     parameter_size = 6
#     width, depth = 32, 4
#     kernel_size = 3
#     unet_depth = 4

#     model = UNetCNO1DTime(nx, T_in, T_out, state_size, parameter_size, width, depth, kernel_size, unet_depth)
    
#     # Generate sample input data
#     u0 = torch.randn(batch_size, nx, T_in, state_size)
#     P = torch.randn(batch_size, parameter_size)
    
#     # Run the model
#     output = model(u0, P)
    
#     print(f"Input shape: {u0.shape}")
#     print(f"\nParameter shape: {P.shape}")
#     print(f"\nOutput shape: {output.shape}")
    