import torch
from torch import nn

__all__ = ("FACAttention",)

from ultralytics.nn.modules.NewAttention.CoordAttention import CoordAttention
from ultralytics.nn.modules.NewAttention.EnhancedChannelAttention import EnhancedChannelAttention
from ultralytics.nn.modules.NewAttention.EnhancedSpatialAttention import EnhancedSpatialAttention


class FACAttention(nn.Module):
    """
    Fused Axial-Coordinate Attention Network (FAC-Attention)
    融合轴向坐标注意力、通道注意力与空间注意力；加入门控融合与残差缩放。
    为适配 Hi-C/TAD，内部默认：
      - 可选训练期对称化（symmetrize_train）
      - 可选对角先验图 DDE（use_dde），通过 1x1 conv 注入到输入特征
    注：保持构造函数签名与原始版本一致，新增能力以成员变量方式提供默认开关。

    Args (保持与原版一致):
        channels (int)
        out: 兼容旧签名，不使用
        reduction (int)
        heads (int)
        mode (int)
        use_multi_scale (bool)
        use_relative_bias (bool)
        use_learnable_coords (bool)
        dropout (float): 最终融合处的 Dropout2d
    """

    def __init__(
        self,
        channels: int,
        out,
        reduction: int = 16,
        heads: int = 8,
        mode: int = 3,
        use_multi_scale: bool = True,
        use_relative_bias: bool = True,
        use_learnable_coords: bool = True,
        dropout: float = 0.1,
    ):
        super(FACAttention, self).__init__()
        self.channels = channels

        # ======== TAD-oriented 额外开关（保持签名不变，通过成员变量控制） ========
        self.symmetrize_train: bool = True   # 训练期对称化：(x + x^T)/2
        self.use_dde: bool = True            # 是否注入对角先验图（Diagonal Distance Encoding）
        self.dde_sigma: float = 0.2          # DDE 高斯尺度（越小越贴主对角）

        # 轴向坐标注意力分支（建议你的 CoordAttention 已为 TAD 增强版）
        self.coord_attn = CoordAttention(
            dim=channels,
            heads=heads,
            reduction=reduction,
            use_relative_bias=use_relative_bias,
            use_learnable_coords=use_learnable_coords
            # 若你的 CoordAttention 已支持更多参数（如 max_pos / band / diag 等），
            # 可在其内部使用默认值；这里保持向后兼容不传入新参。
        )

        # 通道注意力分支（改良版 ECA/SE 已在该类内部实现）
        self.channel_attn = EnhancedChannelAttention(
            channels=channels,
            reduction=reduction,
            mode=mode
        )

        # 空间注意力分支（可选多尺度 + 内部可带对角先验）
        self.spatial_attn = EnhancedSpatialAttention(
            use_multi_scale=use_multi_scale
        )

        # 输入级 DDE 注入：1x1 将 (B,1,H,W) 编码到 C 维并残差注入
        self.dpe = nn.Conv2d(1, channels, kernel_size=1, bias=False) if self.use_dde else None

        # 门控机制 - 动态权重学习（去掉偏置更稳）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 3, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

        # 残差连接缩放因子（初始化为小正值，更易于学习）
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    # -------- DDE 先验生成 --------
    def _make_dde(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """
        生成 (1,1,H,W) 的对角先验图：越靠主对角值越接近 1
        prior = exp(- (|i-j| / (sigma*max(H,W)))^2)
        """
        i = torch.arange(H, device=device).view(H, 1)
        j = torch.arange(W, device=device).view(1, W)
        dist = (i - j).abs().to(dtype) / float(max(H, W))
        prior = torch.exp(- (dist / self.dde_sigma) ** 2)
        return prior.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练期可选对称化（Hi-C 天然对称）
        if self.symmetrize_train:
            x = 0.5 * (x + x.transpose(-1, -2))

        residual = x
        B, C, H, W = x.shape

        # 输入级对角先验注入
        if self.dpe is not None and self.use_dde:
            with torch.no_grad():
                dde = self._make_dde(H, W, x.device, x.dtype).expand(B, 1, H, W)
            x = x + self.dpe(dde)

        # 三个并行分支
        axial_out = self.coord_attn(x)       # (B,C,H,W)
        channel_out = self.channel_attn(x)   # (B,C,H,W)
        spatial_out = self.spatial_attn(x)   # (B,C,H,W)

        # 门控权重学习
        gate_input = torch.cat([axial_out, channel_out, spatial_out], dim=1)  # (B,3C,H,W)
        gates = self.gate_conv(gate_input)                                    # (B,3,H,W)

        # 应用门控权重
        gated_axial = axial_out * gates[:, 0:1, :, :]
        gated_channel = channel_out * gates[:, 1:2, :, :]
        gated_spatial = spatial_out * gates[:, 2:3, :, :]

        # 融合所有分支
        combined = torch.cat([gated_axial, gated_channel, gated_spatial], dim=1)  # (B,3C,H,W)
        out = self.fusion_conv(combined)                                          # (B,C,H,W)

        # 残差连接
        return residual + self.gamma * out
