"""注意力模块: CBAM (Woo et al. 2018) + SE (Hu et al. 2018)
用于 TWAU-Net 混合注意力版本 (v12-hybrid).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018).

    全局平均池化 → FC 压缩 → ReLU → FC 扩张 → Sigmoid → 通道加权.
    用于 s4 全局语义特征 (H/32=20, 局部窗口 attention 收益低).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018).

    Channel Attention (avg+max pool → MLP → sigmoid) +
    Spatial Attention (avg+max along channel → conv7×7 → sigmoid).
    用于 s3 中层语义特征 (H/16=40).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_p = F.adaptive_avg_pool2d(x, 1)
        max_p = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.channel_mlp(avg_p) + self.channel_mlp(max_p))
        x = x * ch
        # Spatial attention
        sp = torch.sigmoid(
            self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1))
        )
        x = x * sp
        return x


class CBAMResBlock(nn.Module):
    """CBAM 残差块: x + CBAM(x), 用 GroupNorm 稳定.

    比裸 CBAM 更稳定,适合做跳跃连接处的混合注意力单元.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.cbam = CBAMBlock(channels, reduction=reduction)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.cbam(x))


class SEResBlock(nn.Module):
    """SE 残差块: x + SE(x), 用 GroupNorm 稳定."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = SEBlock(channels, reduction=reduction)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.se(x))


if __name__ == "__main__":
    print("=== attn_modules 自检 ===")
    for ch in (64, 128, 256, 512):
        se = SEResBlock(ch)
        cbam = CBAMResBlock(ch)
        x = torch.randn(2, ch, 32, 32)
        y_se = se(x)
        y_cb = cbam(x)
        n_se = sum(p.numel() for p in se.parameters()) / 1e6
        n_cb = sum(p.numel() for p in cbam.parameters()) / 1e6
        assert y_se.shape == x.shape
        assert y_cb.shape == x.shape
        print(f"ch={ch:4d}  SE: out {tuple(y_se.shape)}  params {n_se:.3f}M  | "
              f"CBAM: out {tuple(y_cb.shape)}  params {n_cb:.3f}M")
    print("=== attn_modules 自检通过 ===")