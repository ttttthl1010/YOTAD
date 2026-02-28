import torch
from torch import nn


class EnhancedSpatialAttention(nn.Module):
    """
    增强版空间注意力（多尺度）+ 可选对角距离先验通道（DDE）
    - 仍然以 [avg, max] 的通道池化为基础；
    - 额外可拼接一张“越靠主对角越大的先验图”，更贴合 TAD 的对角结构；
    - 多尺度 3/5/7 卷积进行空间融合，或使用单尺度 kernel；
    - 输出一个 [0,1] 的空间权重图，对输入做逐像素重加权。

    Args:
        kernel_size (int): use_multi_scale=False 时的卷积核大小（奇偶均可，内部自动 padding）
        use_multi_scale (bool): 是否启用 3/5/7 多尺度卷积并 1x1 融合
        use_diag_prior (bool): 是否拼接对角先验通道
        prior_type (str): 'gauss' 或 'linear'；控制先验随 |i-j| 的衰减形式
        prior_sigma (float): 先验的尺度（越小越聚焦主对角）
    """

    def __init__(
        self,
        kernel_size: int = 7,
        use_multi_scale: bool = True,
        use_diag_prior: bool = True,
        prior_type: str = "gauss",
        prior_sigma: float = 0.2,
    ):
        super().__init__()
        assert prior_type in ("gauss", "linear")
        assert prior_sigma > 0.0

        self.use_multi_scale = use_multi_scale
        self.use_diag_prior = use_diag_prior
        self.prior_type = prior_type
        self.prior_sigma = float(prior_sigma)

        in_ch = 2 + (1 if use_diag_prior else 0)  # [avg, max] + [prior?]

        if use_multi_scale:
            # 多尺度卷积核（输入通道数根据是否启用先验而变化）
            self.conv3 = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, bias=False)
            self.conv5 = nn.Conv2d(in_ch, 1, kernel_size=5, padding=2, bias=False)
            self.conv7 = nn.Conv2d(in_ch, 1, kernel_size=7, padding=3, bias=False)
            self.fusion = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(
                in_ch,
                1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )

        self.sigmoid = nn.Sigmoid()

    def _diag_prior(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """
        生成 (1,1,H,W) 的对角距离先验：越靠主对角，值越大 ∈ [0,1]
        gauss: prior = exp( - (|i-j| / (sigma*max(H,W)))^2 )
        linear: prior = 1 - clip(|i-j| / (sigma*max(H,W)), 0, 1)
        """
        i = torch.arange(H, device=device).view(H, 1)
        j = torch.arange(W, device=device).view(1, W)
        dist = (i - j).abs().to(dtype) / float(max(H, W))  # [0,1]
        if self.prior_type == "gauss":
            prior = torch.exp(- (dist / self.prior_sigma) ** 2)
        else:
            prior = 1.0 - torch.clamp(dist / self.prior_sigma, 0.0, 1.0)
        return prior.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维度的平均与最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B,1,H,W)

        cat_list = [avg_out, max_out]

        if self.use_diag_prior:
            B, C, H, W = x.shape
            prior = self._diag_prior(H, W, x.device, x.dtype).expand(B, 1, H, W)
            cat_list.append(prior)

        spatial_cat = torch.cat(cat_list, dim=1)           # (B, 2/3, H, W)

        if self.use_multi_scale:
            # 多尺度特征融合
            attn3 = self.conv3(spatial_cat)
            attn5 = self.conv5(spatial_cat)
            attn7 = self.conv7(spatial_cat)
            spatial_attn = self.sigmoid(self.fusion(torch.cat([attn3, attn5, attn7], dim=1)))
        else:
            spatial_attn = self.sigmoid(self.conv(spatial_cat))

        return x * spatial_attn
