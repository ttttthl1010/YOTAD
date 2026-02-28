import torch
from torch import nn

__all__ = ("FACAttention",)

from ultralytics.nn.modules.NewAttention.CoordAttention import CoordAttention
from ultralytics.nn.modules.NewAttention.EnhancedChannelAttention import EnhancedChannelAttention
from ultralytics.nn.modules.NewAttention.EnhancedSpatialAttention import EnhancedSpatialAttention


class FACAttention(nn.Module):
    """
    Fused Axial-Coordinate Attention Network (FAC-Attention)
    融合了轴向注意力、坐标编码、通道注意力和空间注意力的强大模块
    """

    def __init__(self, channels: int, out, reduction: int = 16, heads: int = 8,
                 mode: int = 3, use_multi_scale: bool = True,
                 use_relative_bias: bool = True, use_learnable_coords: bool = True,
                 dropout: float = 0.1):
        super(FACAttention, self).__init__()
        self.channels = channels

        # 轴向坐标注意力分支
        self.coord_attn = CoordAttention(
            dim=channels,
            heads=heads,
            reduction=reduction,
            use_relative_bias=use_relative_bias,
            use_learnable_coords=use_learnable_coords
        )

        # 通道注意力  多种模式
        self.channel_attn = EnhancedChannelAttention(
            channels=channels,
            reduction=reduction,
            mode=mode
        )

        # 空间注意力
        self.spatial_attn = EnhancedSpatialAttention(
            use_multi_scale=use_multi_scale
        )

        # 门控机制 - 动态权重学习
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 3, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        # 残差连接的缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # 三个并行分支
        axial_out = self.coord_attn(x)
        channel_out = self.channel_attn(x)
        spatial_out = self.spatial_attn(x)
        # 门控权重学习
        gate_input = torch.cat([axial_out, channel_out, spatial_out], dim=1)
        gates = self.gate_conv(gate_input)  # (B, 3, H, W)
        # 应用门控权重
        gated_axial = axial_out * gates[:, 0:1, :, :]
        gated_channel = channel_out * gates[:, 1:2, :, :]
        gated_spatial = spatial_out * gates[:, 2:3, :, :]
        # 融合所有分支
        combined = torch.cat([gated_axial, gated_channel, gated_spatial], dim=1)
        out = self.fusion_conv(combined)
        # 残差连接
        return residual + self.gamma * out
