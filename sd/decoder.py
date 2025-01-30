import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, h, w)

        residue = x

        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (batch_size, channels, h, w) -> (batch_size, channels, h * w)
        x = x.view((n, c, h * w))

        # (batch_size, channels, h * w) -> (batch_size, h * w, channels)
        x = x.transpose(-1, -2)

        # (batch_size, h * w, channels) -> (batch_size, h * w, channels)
        x = self.attention(x)

        # (batch_size, h * w, channels) -> (batch_size, channels, h * w)
        x = x.transpose(-1, -2)

        # (batch_size, channels, h * w) -> (batch_size, channels, h, w)
        x = x.view((n, c, h, w))

        x += residue

        return x




class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, h, w)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)

        # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # skip connection
        x += self.residual_layer(residue)

        return x

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/4, w/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/2, w/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h, w)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 128, h, w) -> (batch_size, 3, h, w)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch_size, 4, h/8, w/8)

        z /= 0.18215

        for module in self:
            z = module(z)
        
        # (batch_size, 3, h, w)
        return z