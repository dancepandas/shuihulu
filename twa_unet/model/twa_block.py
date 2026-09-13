"""Triangular Window Attention (TWA) 模块
机制参考: CFAT (arXiv:2403.16143, CVPR 2024) Unleashing Triangular Windows.
将超分任务的三角窗 attention 迁移到 U-Net 跳跃连接处。

核心思想:
  - 标准卷积核 / 标准 window attention 感受野是方形 (8 重对称), 无方向敏感性
  - 三角窗 (下三角 ∨ 或 上三角 ∧) 强制 attention 沿对角线方向展开, 具备方向敏感性
  - 双方向 (lower + upper) 并行 block 覆盖整个窗口, 避免单向丢失信息
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def triangular_token_mask(M: int, direction: str, device=None) -> torch.Tensor:
    """生成 token-level 三角 mask (M*M, M*M).

    每个 token 由它在 M×M 窗口内的 (row, col) 坐标索引 (token idx = row * M + col).

    direction='lower' (左下三角 ∨):
        token (r_i, c_i) 可见 token (r_j, c_j) 当且仅当 r_j ≤ r_i 且 c_j ≤ c_i
        即 token 只能 attend 到自身左下方的 token (含对角线)
    direction='upper' (右上三角 ∧):
        token (r_i, c_i) 可见 token (r_j, c_j) 当且仅当 r_j ≥ r_i 且 c_j ≥ c_i
        即 token 只能 attend 到自身右上方的 token (含对角线)

    数学性质:
      lower mask: 每个 token 可见数 = (r_i+1)*(c_i+1), 全局 ∑ = (M(M+1)/2)² 个非屏蔽对
      upper mask: 每个 token 可见数 = (M-r_i)*(M-c_i), 全局 ∑ = (M(M+1)/2)² 个非屏蔽对
      双方向 (lower || upper) 拼起来 ≈ full attention 的稀疏化
    """
    rows = torch.arange(M, device=device).unsqueeze(1).expand(M, M).reshape(-1)  # (M*M,) rows[i] = r_i
    cols = torch.arange(M, device=device).unsqueeze(0).expand(M, M).reshape(-1)  # (M*M,) cols[i] = c_i
    # 注意: rows.unsqueeze(0) 形状 (1, M*M), broadcast 后 [i, j] = rows[j] = r_j
    #      rows.unsqueeze(1) 形状 (M*M, 1), broadcast 后 [i, j] = rows[i] = r_i
    if direction == "lower":
        # 左下三角: query i 可见 j 当 r_j ≤ r_i 且 c_j ≤ c_i
        # 屏蔽条件: r_j > r_i 或 c_j > c_i
        blocked = (rows.unsqueeze(0) > rows.unsqueeze(1)) | (cols.unsqueeze(0) > cols.unsqueeze(1))
    elif direction == "upper":
        # 右上三角: query i 可见 j 当 r_j ≥ r_i 且 c_j ≥ c_i
        # 屏蔽条件: r_j < r_i 或 c_j < c_i
        blocked = (rows.unsqueeze(0) < rows.unsqueeze(1)) | (cols.unsqueeze(0) < cols.unsqueeze(1))
    else:
        raise ValueError(f"direction={direction!r}, must be 'lower' or 'upper'")
    mask = torch.zeros(M * M, M * M, device=device)
    mask = mask.masked_fill(blocked, float("-inf"))
    return mask


class WindowAttention(nn.Module):
    """单方向 Triangular Window Self-Attention.
    输入输出都是 (B, C, H, W), 在每个 M×M 窗口内做 attention.
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8,
                 direction: str = "lower", shift_size: int = 0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.direction = direction
        # 预注册 token-level 三角 mask (M² × M²)
        mask = triangular_token_mask(window_size, direction)
        self.register_buffer("mask", mask)  # (M*M, M*M)
        # QKV 投影 + 输出投影
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W).
        当 H/W 不被 M 整除时, 自动 pad 到 M 的倍数 (replicate pad), 计算后 crop 回原尺寸.
        当 shift_size > 0 时, 循环偏移特征图 (Swin-style), 增强窗口间信息交互.
        """
        B, C, H, W = x.shape
        M = self.window_size
        S = self.shift_size
        # 1) shift (v14): 循环偏移
        if S > 0:
            x = torch.roll(x, shifts=(-S, -S), dims=(-2, -1))
        # 自动 pad: 上/左 pad 0, 下/右 pad (M - H%M)
        pad_h = (M - H % M) % M
        pad_w = (M - W % M) % M
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = x.shape[-2:]
        nH, nW = Hp // M, Wp // M  # 窗口数
        # 1) reshape 到窗口序列: (B, C, H, W) → (B*nH*nW, M*M, C)
        x = x.view(B, C, nH, M, nW, M)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, M, M, C)
        x = x.view(B * nH * nW, M * M, C)
        # 2) QKV
        qkv = self.qkv(x)  # (B*nH*nW, M*M, 3*C)
        qkv = qkv.reshape(B * nH * nW, M * M, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BN, head, M*M, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各自 (BN, head, M*M, head_dim)
        # 3) attention with triangular mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BN, head, M*M, M*M)
        attn = attn + self.mask  # mask 广播到 head 和 batch
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (BN, head, M*M, head_dim)
        out = out.transpose(1, 2).reshape(B * nH * nW, M * M, C)
        out = self.proj(out)  # (BN, M*M, C)
        # 4) reshape 回 (B, C, Hp, Wp), 再 crop 回 (B, C, H, W)
        out = out.view(B, nH, nW, M, M, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, nH, M, nW, M)
        out = out.view(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        # 5) reverse shift
        if S > 0:
            out = torch.roll(out, shifts=(S, S), dims=(-2, -1))
        return out


class SwiGLU(nn.Module):
    """SwiGLU 门控 FFN (LLaMA 同款).

    结构: (x1 ⊙ SiLU(x2)) → Linear, 其中 [x1, x2] = chunk(Linear(x))
    比标准 GELU MLP 表达力更强, LLaMA 验证有效.
    """

    def __init__(self, dim: int, hidden_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        # 双路合并: dim → 2*hidden
        self.w1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.w2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=1)
        return self.w2(a * F.silu(b))


class TWABlock(nn.Module):
    """双方向 Triangular Window Attention Block (CFAT 风格 + LLM-style 残差).

    v_solve-A 结构 (Pre-Norm + SwiGLU):
      x → [LowerAttn || UpperAttn] → Merge(1×1 conv) → +x          (Post-Norm 残差)
      x → Pre-Norm → SwiGLU → +x                                   (LLM 风格)
      输出 = 第二个 x

    Args:
        dim: 输入/输出通道数
        window_size: 窗口大小, 默认 8
        num_heads: 多头 attention 的头数, 须 dim 整除
        mlp_ratio: MLP 隐藏层扩张比
        direction: 'both' (默认, 双方向)/ 'lower' / 'upper' (消融用)
        shift_size: shifted window (默认 0)
        pre_norm: True=Pre-Norm (LLM 风格), False=Post-Norm (原版)
        use_swiglu: True=SwiGLU 替代 GELU MLP
    """

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 2.0, direction: str = "both", shift_size: int = 0,
                 pre_norm: bool = False, use_swiglu: bool = False):
        super().__init__()
        assert direction in ("both", "lower", "upper"), direction
        self.direction = direction
        self.pre_norm = pre_norm
        self.use_swiglu = use_swiglu
        # 注意力分支
        if direction in ("lower", "both"):
            self.attn_lower = WindowAttention(dim, window_size, num_heads, "lower", shift_size=shift_size)
        if direction in ("upper", "both"):
            self.attn_upper = WindowAttention(dim, window_size, num_heads, "upper", shift_size=shift_size)
        # 双方向时把 2*dim 合并回 dim
        if direction == "both":
            self.merge = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        # 归一化: GroupNorm (squeeze 通道上的 LN, 类似 RMSNorm/LayerNorm 替代)
        gn_groups = min(8, dim)
        # v_solve-A: Pre-Norm 用 2 个 norm (attn 前 + ffn 前); Post-Norm 用 2 个 norm (attn 后 + ffn 后)
        self.norm1 = nn.GroupNorm(gn_groups, dim)
        self.norm2 = nn.GroupNorm(gn_groups, dim)
        # FFN
        if use_swiglu:
            self.ffn = SwiGLU(dim, hidden_ratio=mlp_ratio)
        else:
            hidden = int(dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Attention 残差 ===
        if self.pre_norm:
            # LLM 风格 Pre-Norm: 先归一化再做 attention,最后加残差
            x_norm = self.norm1(x)
            if self.direction == "both":
                a_l = self.attn_lower(x_norm)
                a_u = self.attn_upper(x_norm)
                a = self.merge(torch.cat([a_l, a_u], dim=1))
            else:
                a = self.attn_lower(x_norm) if self.direction == "lower" else self.attn_upper(x_norm)
            x = x + a
        else:
            # Post-Norm (原版)
            if self.direction == "both":
                a_l = self.attn_lower(x)
                a_u = self.attn_upper(x)
                a = self.merge(torch.cat([a_l, a_u], dim=1))
            else:
                a = self.attn_lower(x) if self.direction == "lower" else self.attn_upper(x)
            x = self.norm1(x + a)
        # === FFN 残差 ===
        if self.pre_norm:
            x_norm = self.norm2(x)
            x = x + self.ffn(x_norm)
        else:
            x = self.norm2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    # 自检: forward 维度正确性 + 三角 mask 正确性
    print("=== TWA Block 自检 ===")
    # mask 检查 (M=4, token idx 按 row-major)
    # 期望 lower: token(0,0) 可见 1 个, token(0,1) 可见 2, token(0,2) 可见 3, token(0,3) 可见 4,
    #              token(1,0) 可见 2, token(1,1) 可见 4, token(1,2) 可见 6, token(1,3) 可见 8,
    #              token(2,0) 可见 3, token(2,1) 可见 6, token(2,2) 可见 9, token(2,3) 可见 12,
    #              token(3,0) 可见 4, token(3,1) 可见 8, token(3,2) 可见 12, token(3,3) 可见 16
    expected_lower = [1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16]
    expected_upper = [16, 12, 8, 4, 12, 9, 6, 3, 8, 6, 4, 2, 4, 3, 2, 1]
    for d, expected in [("lower", expected_lower), ("upper", expected_upper)]:
        m = triangular_token_mask(4, d)
        n_visible = torch.isfinite(m).int().sum(dim=1).tolist()
        print(f"{d} 每行可见 token 数: {n_visible}")
        assert n_visible == expected, f"{d} mask wrong:\n  got {n_visible}\n  exp {expected}"
        # 总非屏蔽对数期望 = M(M+1)/2 的平方
        total_keep = torch.isfinite(m).sum().item()
        expected_total = (4 * 5 // 2) ** 2  # (M(M+1)/2)^2 = 100
        print(f"  总非屏蔽对: {total_keep} (期望 {expected_total})")
        assert total_keep == expected_total
    # block 检查 (direction=both)
    block = TWABlock(dim=64, window_size=8, num_heads=8, direction="both")
    x = torch.randn(2, 64, 32, 32)
    y = block(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    n_params = sum(p.numel() for p in block.parameters()) / 1e6
    print(f"direction='both'  dim=64 win8  输入 {x.shape} → 输出 {y.shape}  参数量 {n_params:.3f}M")
    # direction=lower 消融
    block_l = TWABlock(dim=64, window_size=8, num_heads=8, direction="lower")
    y_l = block_l(x)
    n_l = sum(p.numel() for p in block_l.parameters()) / 1e6
    print(f"direction='lower' dim=64 win8  参数量 {n_l:.3f}M")
    print("OK")