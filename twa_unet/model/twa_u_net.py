"""TWAU-Net: U-Net with Triangular Window Attention on skip connections.

架构:
  encoder: torchvision ResNet34 (ImageNet 预训练)
  bottleneck: 512 channels, H/32 × W/32
  4 个 skip connection 处插入 TWABlock (TWA)
  decoder: Conv-BN-ReLU 双卷积块

通道: 64 → 128 → 256 → 512 (encoder 4 个尺度)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# 让 from twa_block import TWABlock 能找到 (model/ 与 twa_block.py 同级)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from twa_block import TWABlock
from attn_modules import CBAMResBlock, SEResBlock


class ConvBlock(nn.Module):
    """Conv-BN-ReLU 双卷积块 (U-Net decoder 基本单元)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TWAUNet(nn.Module):
    """U-Net + Triangular Window Attention.

    Args:
        num_classes: 输出类别数 (默认 6)
        encoder: 编码器骨干, 默认 resnet34 (ImageNet 预训练)
        pretrained: 是否加载 ImageNet 预训练权重
        window_size: TWA 窗口大小, 默认 8
        num_heads: TWA 多头数, 默认 8
        twa_direction: 'both' / 'lower' / 'upper'
    """

    def __init__(self, num_classes: int = 6, encoder: str = "resnet34", pretrained: bool = True,
                 window_size: int = 8, num_heads: int = 8, twa_direction: str = "both",
                 use_twa: bool = True, attn_mode: str = "twa",
                 use_shift: bool = False, deepsup: bool = False,
                 pre_norm: bool = False, use_swiglu: bool = False,
                 gated_skip: bool = False):
        """attn_mode:
          - 'twa': 全部 4 层 TWA (原 v11 行为)
          - 'hybrid': s1=TWA, s2=TWA, s3=CBAM, s4=SE
          - 'hybrid_v2': s1=CBAM, s2=TWA, s3=CBAM, s4=SE
          - 'none': 无任何注意力
        use_shift: Swin-style shifted window for s1/s2 TWA
        deepsup: 启用 deep supervision (aux heads on d2/d3)
        pre_norm (v_solve-A): LLM 风格 Pre-Norm (替代 Post-Norm)
        use_swiglu (v_solve-A): SwiGLU 替代 GELU MLP
        gated_skip (v_solve-B): decoder 加 gated skip connection
        """
        super().__init__()
        if encoder != "resnet34":
            raise NotImplementedError(f"encoder={encoder}, 仅支持 resnet34")
        assert attn_mode in ("twa", "hybrid", "hybrid_v2", "none"), f"attn_mode={attn_mode}"
        # encoder
        if pretrained:
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None
        backbone = tvm.resnet34(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64, H/4, W/4
        self.layer2 = backbone.layer2  # 128, H/8, W/8
        self.layer3 = backbone.layer3  # 256, H/16, W/16
        self.layer4 = backbone.layer4  # 512, H/32, W/32
        # 跳跃连接注意力模块 (按 attn_mode 选择)
        self.use_twa = use_twa
        self.attn_mode = attn_mode
        shift = window_size // 2 if use_shift else 0
        if use_twa and attn_mode != "none":
            if attn_mode == "twa":
                # v_solve-A: Pre-Norm + SwiGLU 改造 TWA 内部
                self.attn1 = TWABlock(64, window_size, num_heads, direction=twa_direction,
                                      shift_size=shift, pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn2 = TWABlock(128, window_size, num_heads, direction=twa_direction,
                                      shift_size=shift, pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn3 = TWABlock(256, window_size, num_heads, direction=twa_direction,
                                      pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn4 = TWABlock(512, window_size, num_heads, direction=twa_direction,
                                      pre_norm=pre_norm, use_swiglu=use_swiglu)
            elif attn_mode == "hybrid":
                self.attn1 = TWABlock(64, window_size, num_heads, direction=twa_direction,
                                      shift_size=shift, pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn2 = TWABlock(128, window_size, num_heads, direction=twa_direction,
                                      shift_size=shift, pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn3 = CBAMResBlock(256, reduction=16)
                self.attn4 = SEResBlock(512, reduction=16)
            elif attn_mode == "hybrid_v2":
                self.attn1 = CBAMResBlock(64, reduction=16)
                self.attn2 = TWABlock(128, window_size, num_heads, direction=twa_direction,
                                      shift_size=shift, pre_norm=pre_norm, use_swiglu=use_swiglu)
                self.attn3 = CBAMResBlock(256, reduction=16)
                self.attn4 = SEResBlock(512, reduction=16)
        # decoder: up + concat(skip) + ConvBlock
        # d4: bottleneck 上采样 → concat with TWA(l3) → ConvBlock
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)  # cat(256+256) → 256
        # d3: d4 上采样 → concat with TWA(l2) → ConvBlock
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)
        # d2: d3 上采样 → concat with TWA(l1) → ConvBlock
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        # d1: d2 上采样 → ConvBlock (无 skip, 因为 stem 浅层无方向性收益)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 64)
        # 最后一个上采样把 H/2 恢复到 H/1 (ResNet34 stem 含 maxpool, 总共下采样 4 倍到 layer1)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # segmentation head
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        # v14: deep supervision aux heads (训练时输出,推理时丢弃)
        self.deepsup = deepsup
        if deepsup:
            self.aux_head_d3 = nn.Conv2d(128, num_classes, kernel_size=1)  # 在 d3 后 (H/8)
            self.aux_head_d2 = nn.Conv2d(64, num_classes, kernel_size=1)   # 在 d2 后 (H/4)
        # v_solve-B: Gated Skip Connection (LLM-style 门控机制)
        # 在 decoder 的 3 个跳跃连接处 (d2/d3/d4) 各加一个 gate, 让网络自己学 skip 重要性
        self.gated_skip = gated_skip
        if gated_skip:
            # gate 输出 (B, skip_ch, H, W) sigmoid, 与 skip 逐元素相乘
            self.skip_gate3 = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=False),
                nn.Sigmoid(),
            )
            self.skip_gate2 = nn.Sequential(
                nn.Conv2d(128, 128, 1, bias=False),
                nn.Sigmoid(),
            )
            self.skip_gate1 = nn.Sequential(
                nn.Conv2d(64, 64, 1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # encoder
        x0 = self.stem(x)        # 64, H/4, W/4
        e1 = self.layer1(x0)     # 64, H/4, W/4
        e2 = self.layer2(e1)     # 128, H/8, W/8
        e3 = self.layer3(e2)     # 256, H/16, W/16
        e4 = self.layer4(e3)     # 512, H/32, W/32
        # 跳跃连接注意力 (use_twa=False 即 vanilla U-Net)
        if self.use_twa and self.attn_mode != "none":
            s1 = self.attn1(e1)
            s2 = self.attn2(e2)
            s3 = self.attn3(e3)
            s4 = self.attn4(e4)
        else:
            s1, s2, s3, s4 = e1, e2, e3, e4
        # decoder (v_solve-B: 可选 gated skip)
        d4 = self.up4(s4)
        if self.gated_skip:
            s3 = s3 * self.skip_gate3(s3)
        d4 = self.dec4(torch.cat([d4, s3], dim=1))
        d3 = self.up3(d4)
        if self.gated_skip:
            s2 = s2 * self.skip_gate2(s2)
        d3 = self.dec3(torch.cat([d3, s2], dim=1))
        # v14: aux head at d3 (H/8)
        if self.deepsup and self.training:
            aux_d3 = self.aux_head_d3(d3)
        d2 = self.up2(d3)
        if self.gated_skip:
            s1 = s1 * self.skip_gate1(s1)
        d2 = self.dec2(torch.cat([d2, s1], dim=1))
        # v14: aux head at d2 (H/4)
        if self.deepsup and self.training:
            aux_d2 = self.aux_head_d2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        d0 = self.up0(d1)
        out = self.head(d0)
        if self.deepsup and self.training:
            return out, aux_d3, aux_d2
        return out


if __name__ == "__main__":
    print("=== TWAU-Net 自检 (含 hybrid 模式) ===")
    for attn_mode in ("twa", "hybrid", "hybrid_v2", "none"):
        for direction in ("both", "lower", "upper") if attn_mode == "twa" else ("both",):
            m = TWAUNet(num_classes=6, pretrained=False, twa_direction=direction,
                        attn_mode=attn_mode)
            x = torch.randn(2, 3, 256, 256)
            y = m(x)
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            tag = f"attn={attn_mode:10s} dir={direction:5s}"
            print(f"{tag}  输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}  参数量 {n_params:.2f}M")
            assert y.shape == (2, 6, 256, 256)
    print("=== TWAU-Net 自检通过 ===")