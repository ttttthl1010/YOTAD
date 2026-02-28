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
    结合 SE 和 ECA 的通道注意力（ECA 核长自适应版）
    mode:
      1 -> 仅 ECA
      2 -> 仅 SE
      0 -> 串联式融合（先分别算，再按公式 m(x*y + x*se) 后 Sigmoid）
      其他 -> 并联融合（门控加权）
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        mode: int = 0,
        eca_gamma: float = 2.0,
        eca_b: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.mode = mode

        # ---- 全局池化 ----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = TopKAvgPool2d(k=4) 
        # ------- ECA 分支（自适应核大小）-------
        k = self._eca_kernel(channels)  # 自适应 kernel size
        self.eca = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False),
            nn.Sigmoid()
        )

        # ---- SE：注意防止中间维为 0 ----
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

        # ---- 并联融合的门控（仅在 mode 其他时使用）----
        self.fusion_gate = nn.Sequential(
            nn.Linear(channels * 2, max(1, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // 4), 2),
            nn.Softmax(dim=-1),
        )

        # 串联融合时的最终 Sigmoid
        self.final_act = nn.Sigmoid()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if self.mode == 1:
            # ---- 仅 ECA ----
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            y = avg_out + max_out                    # (B, C, 1, 1)
            y = y.squeeze(-1).transpose(-1, -2)      # -> (B, 1, C)
            y = self.eca(y).transpose(-1, -2).unsqueeze(-1)  # -> (B, C, 1, 1)
            return x * y

        elif self.mode == 2:
            # ---- 仅 SE ----
            avg_out = self.fc(self.avg_pool(x).view(B, C))
            max_out = self.fc(self.max_pool(x).view(B, C))
            channel_attn = (avg_out + max_out).view(B, C, 1, 1)
            return x * channel_attn

        elif self.mode == 0:
            # ---- 串联式融合：ECA + SE，再做一次 Sigmoid ----
            # ECA 分支
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            y = avg_out + max_out
            y = y.squeeze(-1).transpose(-1, -2)                      # (B, 1, C)
            y = self.eca(y).transpose(-1, -2).unsqueeze(-1)          # (B, C, 1, 1)

            # SE 分支
            avg_se = self.fc(self.avg_pool(x).view(B, C))
            max_se = self.fc(self.max_pool(x).view(B, C))
            se = (avg_se + max_se).view(B, C, 1, 1)

            # 融合（与你原逻辑保持一致）
            out = self.final_act(x * y + x * se)
            return out

        else:
            # ---- 并联式融合：门控加权 ----
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            y = avg_out + max_out

            # SE
            se = self.fc(y.view(B, C)).view(B, C, 1, 1)

            # ECA（走 1xC 的 1D 卷积）
            eca_in = y.squeeze(-1).transpose(-1, -2)                 # (B, 1, C)
            eca = self.eca(eca_in).transpose(-1, -2).unsqueeze(-1)   # (B, C, 1, 1)

            # 门控（逐样本 2 路权重）
            combined = torch.cat([se, eca], dim=1)                   # (B, 2C, 1, 1)
            gates = self.fusion_gate(combined.view(B, 2 * C)).view(B, 2, 1, 1)
            final = gates[:, 0:1] * se + gates[:, 1:2] * eca         # (B, 1, 1, 1)按通道广播

            return x * final