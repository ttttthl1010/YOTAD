import math
import torch
from torch import nn


# --- 在文件头加入 ---
class TopKAvgPool2d(nn.Module):
    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.view(B, C, -1)
        k = min(self.k, t.shape[-1])
        v, _ = torch.topk(t, k, dim=-1, largest=True, sorted=False)
        return v.mean(dim=-1, keepdim=True).view(B, C, 1, 1)

class EnhancedChannelAttention(nn.Module):
    """
    结合 SE 与 ECA 的通道注意力（改良版，保持原始构造签名）
    - ECA: 使用自适应 kernel（随 C 变化，奇数且>=3）
    - SE: 两次全连接（降维-升维），通道重标定
    - mode:
        0 -> ECA + SE 融合（去掉整体 Sigmoid，加入残差缩放 gamma）
        1 -> 仅 ECA
        2 -> 仅 SE
        other -> 并联门控融合
    """

    def __init__(self, channels: int, reduction: int = 16, mode: int = 0):
        super().__init__()
        assert channels > 0, "channels must be positive"
        self.channels = channels
        self.reduction = max(1, reduction)
        self.mode = mode

        # ------- 全局池化 -------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = TopKAvgPool2d(k=4) 
        # ------- ECA 分支（自适应核大小）-------
        k = self._eca_kernel(channels)  # 自适应 kernel size
        self.eca = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False),
            nn.Sigmoid()
        )

        # ------- SE 分支 -------
        hidden = max(1, channels // self.reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

        # ------- 并联门控（用于 else 分支）-------
        gate_hidden = max(2, channels // 4)
        self.fusion_gate = nn.Sequential(
            nn.Linear(channels * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2),
            nn.Softmax(dim=-1)
        )

        # ------- 激活缓存（避免 forward 里重复创建）-------
        self.sigmoid = nn.Sigmoid()

        # ------- 残差缩放（仅在 mode=0 使用）-------
        # 不改变构造签名，默认启用并初始化为 1.0（等效直通）
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) if self.mode == 0 else None

    # ---------------- 工具 ----------------
    @staticmethod
    def _eca_kernel(c: int, gamma: float = 2.0, b: int = 1) -> int:
        """
        自适应 ECA kernel 大小（参考原论文公式）:
            k = odd( |log2(C)/gamma + b| ), 且 k >= 3
        """
        k = int(abs(math.log2(max(2, c)) / gamma + b))
        if k % 2 == 0:
            k += 1
        return max(3, k)

    def _eca_branch(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: x (B, C, H, W) -> 输出: 权重 (B, C, 1, 1)
        avg_out = self.avg_pool(x)  # (B,C,1,1)
        max_out = self.max_pool(x)  # (B,C,1,1)
        y = avg_out + max_out
        y = y.squeeze(-1).transpose(-1, -2)            # (B,1,C)
        y = self.eca(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return y

    def _se_branch(self, x: torch.Tensor) -> torch.Tensor:
        # 输出: 权重 (B, C, 1, 1)
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_attn = (avg_out + max_out).view(x.size(0), -1, 1, 1)
        return channel_attn

    # ---------------- 前向 ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 1:
            # 仅 ECA
            y = self._eca_branch(x)                # (B,C,1,1)
            return x * y

        elif self.mode == 2:
            # 仅 SE
            s = self._se_branch(x)                 # (B,C,1,1)
            return x * s

        elif self.mode == 0:
            # ECA + SE 融合（去除整体 Sigmoid，加入残差缩放 gamma）
            y = self._eca_branch(x)                # (B,C,1,1)
            s = self._se_branch(x)                 # (B,C,1,1)
            fused = x * y + x * s                  # 不做整体 Sigmoid，保留幅度空间
            if self.gamma is not None:
                # 残差形式增强稳健性（gamma 初值 1.0 -> 等效直通）
                fused = x + self.gamma * (fused - x)
            return fused

        else:
            # 并联门控融合（稳定但稍重）
            B, C, _, _ = x.shape
            s = self._se_branch(x)                 # (B,C,1,1)
            y = self._eca_branch(x)                # (B,C,1,1)
            combined = torch.cat([s, y], dim=1)    # (B,2C,1,1)
            gates = self.fusion_gate(combined.view(B, 2 * C)).view(B, 2, 1, 1)
            final = gates[:, 0:1] * s + gates[:, 1:2] * y
            return x * self.sigmoid(final)         # 维持与原实现语义一致
