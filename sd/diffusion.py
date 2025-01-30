import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention 


class TimeEmbedding(nn.Module):

    def __init__(self, d_embed: int):
        super().__init__()

        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x


class UNet_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, d_time=1280):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time):
        # x: (batch_size, in_channels, h, w)
        # time: (1, 1280)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)

        # (batch_size, in_channels, h, w) -> (batch_size, out_channels, h, w)
        x = self.conv_1(x)

        time = F.silu(time)

        # (1, 1280) -> (1, out_channels)
        time = self.linear_time(time)

        # (batch_size, out_channels, h, w)
        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_2(merged)
        merged = F.silu(merged)
        merged = self.conv_2(merged)

        # skip connection
        merged += self.residual_layer(residue)

        return merged


class UNet_AttentionBlock(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_context=768):
        super().__init__()

        channels = n_heads * d_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, w_qkv_bias=False)

        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, w_qkv_bias=False)

        self.layer_norm_3 = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, h, w)
        # context: (batch_size, seq_length, d_context)

        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (batch_size, channels, h, w) -> (batch_size, channels, h * w)
        x = x.view((n, c, h * w))

        # (batch_size, channels, h * w) -> (batch_size, h * w, channels)
        x = x.transpose(-1, -2)

        # normalization + self-attention with skip connection

        residue_short = x

        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # normalization + cross-attention with skip connection
        residue_short = x

        x = self.layer_norm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # normalization + feed-forward with GeGLU and skip connection
        residue_short = x

        x = self.layer_norm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x *= F.gelu(gate)
        x = self.linear_geglu_2(x)

        x += residue_short

        # (batch_size, h * w, channels) -> (batch_size, channels, h * w)
        x = x.transpose(-1, -2)

        # (batch_size, channels, h * w) -> (batch_size, channels, h, w)
        x = x.view((n, c, h, w))

        x = self.conv_output(x)

        x += residue_long

        return x


class UpSample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, h, w)

        # (batch_size, channels, h, w) -> (batch_size, channels, h * 2, w * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv(x)

        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x, context, time):
        for module in self:
            if isinstance(module, UNet_AttentionBlock):
                x = module(x, context)
            elif isinstance(module, UNet_ResidualBlock):
                x = module(x, time)
            else:
                x = module(x)
        return x


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([

            # (batch_size, 4, h/8, w/8) -> (batch_size, 320, h/8, w/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # (batch_size, 320, h/8, w/8) -> (batch_size, 320, h/16, w/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (batch_size, 320, h/16, w/16) -> (batch_size, 640, h/16, w/16)
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # (batch_size, 640, h/16, w/16) -> (batch_size, 640, h/32, w/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # (batch_size, 640, h/32, w/32) -> (batch_size, 1280, h/32, w/32)
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # (batch_size, 1280, h/32, w/32) -> (batch_size, 1280, h/64, w/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),

            UNet_AttentionBlock(8, 160),

            UNet_ResidualBlock(1280, 1280)
        )

        self.decoder = nn.ModuleList([
            
            # (batch_size, 2560, h/64, w/64) -> (batch_size, 1280, h/64, w/64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            # (batch_size, 2560, h/64, w/64) -> (batch_size, 1280, h/32, w/32)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            # (batch_size, 1920, h/32, w/32) -> (batch_size, 1280, h/16, w/16)
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),
            
            # (batch_size, 1920, h/16, w/16) -> (batch_size, 640, h/16, w/16)
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            # (batch_size, 1280, h/16, w/16) -> (batch_size, 640, h/16, w/16)
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),

            # (batch_size, 960, h/16, w/16) -> (batch_size, 640, h/8, w/8)
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),

            # (batch_size, 960, h/8, w/8) -> (batch_size, 320, h/8, w/8)
            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),

            # (batch_size, 640, h/8, w/8) -> (batch_size, 320, h/8, w/8)
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

            # (batch_size, 640, h/8, w/8) -> (batch_size, 320, h/8, w/8)
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (batch_size, 4, h/8, w/8)
        # context: (batch_size, seq_length, d_embed)
        # time: (1, 1280)
        skip_connections = []
        for seq in self.encoder:
            x = seq(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for seq in self.decoder:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = seq(x, context, time)

        return x



class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 320, h/8, w/8)

        x = self.groupnorm(x)

        x = F.silu(x)

        # (batch_size, 320, h/8, w/8) -> (batch_size, 4, h/8, w/8)
        x = self.conv(x)

        return x


class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)

        self.unet = UNet()

        self.output_layer = UNet_OutputLayer(320, 4)

    def forward(self, z, context, time):
        # z: (batch_size, 4, h/8, w/8)
        # context: (batch_size, seq_length, d_embed)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, h/8, w/8) -> (batch_size, 320, h/8, w/8)
        output = self.unet(z, context, time)

        # (batch_size, 320, h/8, w/8) -> (batch_size, 4, h/8, w/8)
        output = self.output_layer(output)

        return output