import torch
import torch.nn as nn


class CoordAttention(nn.Module):
    """
    坐标注意力子模块 - 高度和宽度轴向注意力
    参考: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
    增强: 添加了可学习的位置编码和相对位置偏置
    将2D的全局注意力分解为两个1D的轴向注意力（高度方向和宽度方向），并显式地将位置坐标信息编码到注意力机制中
    """

    def __init__(self, dim: int, heads: int = 8, reduction: int = 32,
                 use_relative_bias: bool = True, use_learnable_coords: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads  # 每个注意力头的维度
        self.reduction = reduction  # 坐标编码MLP中的降维比率
        self.use_relative_bias = use_relative_bias
        self.use_learnable_coords = use_learnable_coords

        # 确保 reduced_dim 至少等于 head_dim
        reduced_dim = max(dim // reduction, self.head_dim)
        self.reduced_dim = reduced_dim

        # 坐标编码MLPs - 输出维度调整为 head_dim
        if use_learnable_coords:
            self.coord_encoder_h = nn.Sequential(
                nn.Linear(1, reduced_dim),  # 输入时1维坐标
                nn.GELU(),
                nn.Linear(reduced_dim, self.head_dim),  # 输出维度调整为 head_dim
                nn.GELU()
            )
            self.coord_encoder_w = nn.Sequential(
                nn.Linear(1, reduced_dim),
                nn.GELU(),
                nn.Linear(reduced_dim, self.head_dim),  # 输出维度调整为 head_dim
                nn.GELU()
            )
        else:
            # 使用固定的正弦位置编码
            self.register_buffer('coord_emb_h', self.create_sinusoidal_embeddings(1000, self.head_dim))
            self.register_buffer('coord_emb_w', self.create_sinusoidal_embeddings(1000, self.head_dim))

        # QKV投影矩阵 - 分别处理高度和宽度方向
        self.to_qkv_h = nn.Linear(dim, dim * 3, bias=False)
        self.to_qkv_w = nn.Linear(dim, dim * 3, bias=False)

        # 坐标信息的投影矩阵
        self.coord_proj_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.coord_proj_k = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        # 相对位置偏置表
        if use_relative_bias:
            # 创建一个可学习的参数表，用于存储不同相对位置关系的偏置值
            self.relative_bias_table_h = nn.Parameter(torch.zeros(2 * 128 - 1, heads))
            # print('hhhh', self.relative_bias_table_h.shape)
            self.relative_bias_table_w = nn.Parameter(torch.zeros(2 * 128 - 1, heads))
            nn.init.trunc_normal_(self.relative_bias_table_h, std=0.02)
            nn.init.trunc_normal_(self.relative_bias_table_w, std=0.02)
        self.scale = self.head_dim ** -0.5

    def create_sinusoidal_embeddings(self, n_positions: int, dim: int) -> torch.Tensor:
        """创建正弦位置编码"""
        position = torch.arange(n_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        emb = torch.zeros(n_positions, dim)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb

    def get_relative_positions(self, length: int, device: torch.device) -> torch.Tensor:
        """获取相对位置索引"""
        indices = torch.arange(length, device=device)
        relative_positions = indices[:, None] - indices[None, :]
        return relative_positions + (length - 1)  # 让索引从0开始

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        # --- 高度轴注意力 ---
        # 重塑输入: (B, C, H, W) -> (B*W, H, C)
        # 将高度轴注意力计算转换为 (B*W, H, C)，可以对每个宽度位置独立计算高度方向的注意力。
        x_h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)

        # 生成QKV 并重塑为多头格式
        qkv_h = self.to_qkv_h(x_h).chunk(3, dim=-1)
        #  (B*W, heads, H, head_dim)
        q_h, k_h, v_h = map(lambda t: t.reshape(B * W, H, self.heads, self.head_dim).permute(0, 2, 1, 3), qkv_h)

        # 添加坐标信息
        if self.use_learnable_coords:
            coord_h = torch.arange(H, device=device).float().unsqueeze(1).to(dtype=dtype)  # （H，1）
            coord_emb_h = self.coord_encoder_h(coord_h)  # (H, head_dim) 坐标编码为向量

            # 投影坐标信息到合适的空间（QK）
            coord_q = self.coord_proj_q(coord_emb_h).unsqueeze(0).unsqueeze(1)  # (1, 1, H, head_dim)
            coord_k = self.coord_proj_k(coord_emb_h).unsqueeze(0).unsqueeze(1)  # (1, 1, H, head_dim)
            # 确保坐标投影与QK数据类型一致
            coord_q = coord_q.to(dtype=dtype)
            coord_k = coord_k.to(dtype=dtype)
            # 添加坐标信息到Q和K  广播添加到所有批次和头
            q_h = q_h + coord_q
            k_h = k_h + coord_k
        else:
            # 使用预计算的正弦编码
            coord_emb_h = self.coord_emb_h[:H].to(device,dtype=dtype)  # (H, head_dim)
            coord_emb_h = coord_emb_h.unsqueeze(0).unsqueeze(1)  # (1, 1, H, head_dim)
            q_h = q_h + coord_emb_h
            k_h = k_h + coord_emb_h

        # 计算注意力分数 (B*W, heads, H, H)
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        if self.use_relative_bias:
            # 获取相对位置索引
            # print('sss', H)
            relative_positions_h = self.get_relative_positions(H, device)
            # (heads, H, H)
            # print('rH', relative_positions_h.shape)
            # print(self.relative_bias_table_h)
            relative_bias_h = self.relative_bias_table_h[relative_positions_h].permute(2, 0, 1)
            # 广播添加偏置
            relative_bias_h = relative_bias_h.to(dtype=dtype)
            attn_h = attn_h + relative_bias_h.unsqueeze(0)

        attn_h = attn_h.softmax(dim=-1)
        # 加权求和
        out_h = (attn_h @ v_h).permute(0, 2, 1, 3).reshape(B * W, H, C)
        out_h = self.to_out(out_h).reshape(B, W, H, C).permute(0, 3, 2, 1)        # -> (B, C, H, W)  结束，不要再 reshape 一次

        # --- 宽度轴注意力 ---
        # 重塑输入: (B, C, H, W) -> (B*H, W, C)
        x_w = x.permute(0, 2, 1, 3).reshape(B * H, W, C)

        # 生成QKV
        qkv_w = self.to_qkv_w(x_w).chunk(3, dim=-1)
        q_w, k_w, v_w = map(lambda t: t.reshape(B * H, W, self.heads, self.head_dim).permute(0, 2, 1, 3), qkv_w)

        # 添加坐标信息
        if self.use_learnable_coords:
            coord_w = torch.arange(W, device=device).float().unsqueeze(1).to(dtype=dtype)
            coord_emb_w = self.coord_encoder_w(coord_w)  # (W, head_dim)

            coord_q = self.coord_proj_q(coord_emb_w).unsqueeze(0).unsqueeze(1)  # (1, 1, W, head_dim)
            coord_k = self.coord_proj_k(coord_emb_w).unsqueeze(0).unsqueeze(1)  # (1, 1, W, head_dim)
            # 确保数据类型一致
            coord_q = coord_q.to(dtype=dtype)
            coord_k = coord_k.to(dtype=dtype)
            q_w = q_w + coord_q
            k_w = k_w + coord_k
        else:
            coord_emb_w = self.coord_emb_w[:W].to(device,dtype=dtype)  # (W, head_dim)
            coord_emb_w = coord_emb_w.unsqueeze(0).unsqueeze(1)  # (1, 1, W, head_dim)
            q_w = q_w + coord_emb_w
            k_w = k_w + coord_emb_w

        # 计算注意力分数
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        if self.use_relative_bias:
            relative_positions_w = self.get_relative_positions(W, device)
            relative_bias_w = self.relative_bias_table_w[relative_positions_w].permute(2, 0, 1)
            relative_bias_w = relative_bias_w.to(dtype=dtype)
            attn_w = attn_w + relative_bias_w.unsqueeze(0)

        attn_w = attn_w.softmax(dim=-1)
        out_w = (attn_w @ v_w).permute(0, 2, 1, 3).reshape(B * H, W, C)
        out_w = self.to_out(out_w).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 融合两个轴向注意力
        z = out_h + out_w
        return z
