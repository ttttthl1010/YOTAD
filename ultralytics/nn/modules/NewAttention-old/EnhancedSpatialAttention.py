import torch
from torch import nn


class EnhancedSpatialAttention(nn.Module):
    """增强版空间注意力 - 多尺度特征融合"""

    def __init__(self, kernel_size: int = 7, use_multi_scale: bool = True):
        super().__init__()
        self.use_multi_scale = use_multi_scale

        if use_multi_scale:
            # 多尺度卷积核
            self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
            self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
            self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.fusion = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维度的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_out, max_out], dim=1)

        if self.use_multi_scale:
            # 多尺度特征融合
            attn3 = self.conv3(spatial_cat)
            attn5 = self.conv5(spatial_cat)
            attn7 = self.conv7(spatial_cat)
            spatial_attn = self.sigmoid(self.fusion(torch.cat([attn3, attn5, attn7], dim=1)))
        else:
            spatial_attn = self.sigmoid(self.conv(spatial_cat))

        return x * spatial_attn
