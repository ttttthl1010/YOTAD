import math
import torch
import torch.nn as nn
from typing import Optional


class CoordAttention(nn.Module):
    """
    坐标/轴向自注意力（面向 TAD 强化版）
    - 将 2D 全局注意力分解为两个 1D 轴向注意力（Height / Width）
    - 支持可学习/正弦坐标编码（对齐 head_dim）
    - 共享相对位置偏置表，动态 slice（max_pos 可配，避免 128 上限）
    - 带状注意力（banded mask）与对角先验（diagonal prior）专为 Hi-C/TAD 设计
    - QKV 两种实现：'linear'（与原实现一致） 或 'conv'（1x1 卷积，部署友好）

    Args:
        dim (int): 输入通道数 (= 多头注意力的总维度)
        heads (int): 注意力头数
        reduction (int): 坐标 MLP 的降维比
        use_relative_bias (bool): 是否启用相对位置偏置
        use_learnable_coords (bool): 是否使用可学习坐标编码；否则为正弦
        max_pos (int): 相对位置偏置表的最大位置长度（将动态 slice / clip）
        use_diag_prior (bool): 是否启用对角先验（贴合 Hi-C 主对角）
        diag_type (str): 'gauss' 或 'linear'
        diag_sigma (float): 对角先验强度/尺度，越小越聚焦对角
        use_band_mask (bool): 是否启用带状掩码
        band_ratio (float): 带宽相对于 H/W 的比例（0~1）
        qkv_mode (str): 'linear' 或 'conv'（1x1）
        dropout (float): 输出投影的 dropout
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        reduction: int = 32,
        use_relative_bias: bool = True,
        use_learnable_coords: bool = True,
        max_pos: int = 1024,
        use_diag_prior: bool = True,
        diag_type: str = "gauss",
        diag_sigma: float = 0.15,
        use_band_mask: bool = True,
        band_ratio: float = 0.25,
        qkv_mode: str = "linear",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert heads > 0, "heads must be positive"
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        assert diag_type in ("gauss", "linear")
        assert 0.0 < diag_sigma, "diag_sigma must be positive"
        assert 0.0 < band_ratio <= 1.0, "band_ratio must be in (0,1]"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.reduction = reduction
        self.use_relative_bias = use_relative_bias
        self.use_learnable_coords = use_learnable_coords
        self.max_pos = max_pos
        self.use_diag_prior = use_diag_prior
        self.diag_type = diag_type
        self.diag_sigma = diag_sigma
        self.use_band_mask = use_band_mask
        self.band_ratio = band_ratio
        self.qkv_mode = qkv_mode

        # 坐标编码维度（确保不低于 head_dim）
        reduced_dim = max(dim // reduction, self.head_dim)
        self.reduced_dim = reduced_dim

        # --- 坐标编码（输出对齐 head_dim）---
        if use_learnable_coords:
            self.coord_encoder_h = nn.Sequential(
                nn.Linear(1, reduced_dim),
                nn.GELU(),
                nn.Linear(reduced_dim, self.head_dim),
                nn.GELU(),
            )
            self.coord_encoder_w = nn.Sequential(
                nn.Linear(1, reduced_dim),
                nn.GELU(),
                nn.Linear(reduced_dim, self.head_dim),
                nn.GELU(),
            )
        else:
            # 预计算较长的正弦表（根据 max_pos 与 head_dim）
            self.register_buffer(
                "coord_emb_h",
                self.create_sinusoidal_embeddings(self.max_pos, self.head_dim),
                persistent=False,
            )
            self.register_buffer(
                "coord_emb_w",
                self.create_sinusoidal_embeddings(self.max_pos, self.head_dim),
                persistent=False,
            )

        # --- QKV 投影 ---
        if qkv_mode.lower() == "conv":
            # NCHW 风格 1x1 conv，部署友好（后续按轴向 reshape）
            self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
            self.qkv_linear_h = None
            self.qkv_linear_w = None
        else:
            # 与你原版一致：先按轴向展平，再做 Linear
            self.qkv_conv = None
            self.qkv_linear_h = nn.Linear(dim, dim * 3, bias=False)
            self.qkv_linear_w = nn.Linear(dim, dim * 3, bias=False)

        # 坐标信息投影到 Q/K 空间
        self.coord_proj_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.coord_proj_k = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 共享相对位置偏置表（H/W 共用）
        if use_relative_bias:
            self.relative_bias_table = nn.Parameter(torch.zeros(2 * self.max_pos - 1, heads))
            nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

        self.scale = self.head_dim ** -0.5

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    # ---------- 工具函数 ----------
    @staticmethod
    def create_sinusoidal_embeddings(n_positions: int, dim: int) -> torch.Tensor:
        """正弦位置编码表 (n_positions, dim)"""
        position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)  # (n,1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        emb = torch.zeros(n_positions, dim, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb  # (n_positions, dim)

    def _get_coord_emb(self, L: int, which: str, device, dtype) -> torch.Tensor:
        """返回长度为 L 的坐标嵌入 (L, head_dim)"""
        if self.use_learnable_coords:
            coord = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
            encoder = self.coord_encoder_h if which == "h" else self.coord_encoder_w
            return encoder(coord)  # (L, head_dim)
        else:
            table = self.coord_emb_h if which == "h" else self.coord_emb_w
            if L <= self.max_pos:
                return table[:L].to(device=device, dtype=dtype)
            else:
                # 超出 max_pos 时周期性或clip都可以，这里采用 clip
                return table[-1:].repeat(L, 1).to(device=device, dtype=dtype)

    def _relative_bias_2d(self, L: int, device, dtype) -> torch.Tensor:
        """
        构造 (1, heads, L, L) 的相对位置偏置张量，来自 1D 表动态 slice/clip。
        """
        # pairwise 相对位移 ∈ [-(L-1), L-1]
        idx_row = torch.arange(L, device=device)
        rel = (idx_row[:, None] - idx_row[None, :]).clamp(-self.max_pos + 1, self.max_pos - 1)
        # shift 到 [0, 2*max_pos-2]
        rel_shifted = rel + (self.max_pos - 1)  # (L,L)
        rb = self.relative_bias_table[rel_shifted]  # (L,L,heads)
        return rb.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)  # (1,heads,L,L)

    def _apply_band_mask(self, attn: torch.Tensor, L: int) -> torch.Tensor:
        """
        对注意力 logits 应用带状掩码：带宽 = band_ratio * L
        attn: (B*, heads, L, L)
        """
        bw = max(1, int(self.band_ratio * L))
        idx = torch.arange(L, device=attn.device)
        band = (idx[None, :] - idx[:, None]).abs() <= bw  # (L,L)
        return attn.masked_fill(~band.view(1, 1, L, L), float("-inf"))

    def _add_diagonal_prior(self, attn: torch.Tensor, L: int, S_index: torch.Tensor) -> torch.Tensor:
        """
        对角先验加入到 logits：
        - S_index: 当前序列的“轴向位置索引”，形状 (B*,)，例如 H 向时是列索引 w，W 向时是行索引 h
        - 生成 (B*, L) 的距离，并沿 query/key 两个维度各加一次
        attn: (B*, heads, L, L)
        """
        dtype = attn.dtype
        device = attn.device
        idx = torch.arange(L, device=device).view(1, L)  # (1,L)
        dist = (idx - S_index.view(-1, 1)).abs().to(dtype) / float(L)  # (B*, L)

        if self.diag_type == "gauss":
            prior = - (dist / self.diag_sigma) ** 2
        else:
            prior = - dist / self.diag_sigma

        # 在 query 与 key 两侧各加一次（经验更稳）
        attn = attn + prior.unsqueeze(1).unsqueeze(-1) + prior.unsqueeze(1).unsqueeze(-2)
        return attn

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=dim, H, W)
        返回: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # --------- QKV 生成 ---------
        if self.qkv_conv is not None:
            # conv 模式：在 NCHW 上一次性生成 QKV，再分别按轴向重排
            qkv_full = self.qkv_conv(x)  # (B, 3C, H, W)
            # H 轴向：先把通道拆分成 Q/K/V，再重排到 (B*W, H, C)
            q_full_h, k_full_h, v_full_h = torch.chunk(qkv_full, 3, dim=1)  # (B, C, H, W) *3
            x_h_q = q_full_h.permute(0, 3, 2, 1).reshape(B * W, H, C)
            x_h_k = k_full_h.permute(0, 3, 2, 1).reshape(B * W, H, C)
            x_h_v = v_full_h.permute(0, 3, 2, 1).reshape(B * W, H, C)

            # W 轴向
            q_full_w, k_full_w, v_full_w = q_full_h, k_full_h, v_full_h  # 复用同一组权重
            x_w_q = q_full_w.permute(0, 2, 3, 1).reshape(B * H, W, C)
            x_w_k = k_full_w.permute(0, 2, 3, 1).reshape(B * H, W, C)
            x_w_v = v_full_w.permute(0, 2, 3, 1).reshape(B * H, W, C)
        else:
            # linear 模式：与你原版一致
            # --- 高度轴 ---
            x_h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
            qkv_h = self.qkv_linear_h(x_h).chunk(3, dim=-1)
            x_h_q, x_h_k, x_h_v = qkv_h

            # --- 宽度轴 ---
            x_w = x.permute(0, 2, 1, 3).reshape(B * H, W, C)  # (B*H, W, C)
            qkv_w = self.qkv_linear_w(x_w).chunk(3, dim=-1)
            x_w_q, x_w_k, x_w_v = qkv_w

        # 统一 reshape 到多头: (B*, heads, L, head_dim)
        def reshape_heads(t, L):
            return t.reshape(t.size(0), L, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q_h = reshape_heads(x_h_q, H)
        k_h = reshape_heads(x_h_k, H)
        v_h = reshape_heads(x_h_v, H)

        q_w = reshape_heads(x_w_q, W)
        k_w = reshape_heads(x_w_k, W)
        v_w = reshape_heads(x_w_v, W)

        # --------- 加入坐标编码到 Q/K ---------
        # H 向
        coord_h = self._get_coord_emb(H, "h", device, dtype)  # (H, head_dim)
        cq_h = self.coord_proj_q(coord_h).unsqueeze(0).unsqueeze(1).to(dtype)  # (1,1,H,head_dim)
        ck_h = self.coord_proj_k(coord_h).unsqueeze(0).unsqueeze(1).to(dtype)
        q_h = q_h + cq_h
        k_h = k_h + ck_h

        # W 向
        coord_w = self._get_coord_emb(W, "w", device, dtype)  # (W, head_dim)
        cq_w = self.coord_proj_q(coord_w).unsqueeze(0).unsqueeze(1).to(dtype)  # (1,1,W,head_dim)
        ck_w = self.coord_proj_k(coord_w).unsqueeze(0).unsqueeze(1).to(dtype)
        q_w = q_w + cq_w
        k_w = k_w + ck_w

        # --------- H 轴向注意力 (B*W, heads, H, H) ---------
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale

        if self.use_relative_bias:
            rb_h = self._relative_bias_2d(H, device, dtype)  # (1,heads,H,H)
            attn_h = attn_h + rb_h

        if self.use_band_mask:
            attn_h = self._apply_band_mask(attn_h, H)

        if self.use_diag_prior:
            # 对于 H 向，每个序列对应的“列索引 w”：0..W-1，重复 B 次
            w_idx = torch.arange(W, device=device).repeat(B)  # (B*W,)
            attn_h = self._add_diagonal_prior(attn_h, H, w_idx)

        attn_h = attn_h.softmax(dim=-1)

        out_h = (attn_h @ v_h).permute(0, 2, 1, 3).reshape(B * W, H, C)
        out_h = self.to_out(out_h).reshape(B, W, H, C).permute(0, 2, 3, 1).reshape(B, C, H, W)

        # --------- W 轴向注意力 (B*H, heads, W, W) ---------
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale

        if self.use_relative_bias:
            rb_w = self._relative_bias_2d(W, device, dtype)  # (1,heads,W,W)
            attn_w = attn_w + rb_w

        if self.use_band_mask:
            attn_w = self._apply_band_mask(attn_w, W)

        if self.use_diag_prior:
            # 对于 W 向，每个序列对应的“行索引 h”：0..H-1，重复 B 次
            h_idx = torch.arange(H, device=device).repeat(B)  # (B*H,)
            attn_w = self._add_diagonal_prior(attn_w, W, h_idx)

        attn_w = attn_w.softmax(dim=-1)

        out_w = (attn_w @ v_w).permute(0, 2, 1, 3).reshape(B * H, W, C)
        out_w = self.to_out(out_w).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)

        # --------- 融合两个轴向注意力 ---------
        z = out_h + out_w
        return z
