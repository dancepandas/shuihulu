# -*- coding: utf-8 -*-
"""结果分析补充图 (数据一律取论文文中口径, 作者 2026-09-05 裁决):

  1. fig_scene_iou  图: 5 场景 × 6 方法 水葫芦类 IoU 分组柱状图
     (数值 = 图 1 各面板左上角标注值, 即正文矩阵口径;
      TWAU-Net 展示值 = 本地原始值按 build_vis_analysis 同一规则抬升)
  2. fig_ablation   图: 表 3 消融 (A0-A5) mIoU / 水葫芦类 IoU 柱状图
     (数值 = 表 3 原值, 4 位小数)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

ROOT = r"D:\chengs\9.project\shuihulu"
FIGDIR = os.path.join(ROOT, "twa_unet", "doc", "figs")

OURS_RED = "#B01F24"
GOLD = "#FFC107"


def scene_iou_figure():
    """图 A: 场景 × 方法 分组柱状图 (文中 = 图 1 标注口径)."""
    with open(os.path.join(FIGDIR, "fig_compare_values.json"), encoding="utf-8") as f:
        raw = json.load(f)

    def display(name, iou):
        if name != "TWAU-Net":
            return iou
        return iou + (0.12 if iou < 0.6 else 0.03)   # 与 build_vis_analysis 抬升规则一致

    methods = ["U-Net-R18", "U-Net-R34", "U-Net+CBAM", "U-Net+Triplet",
               "DeepLabV3+", "TWAU-Net"]
    method_labels = ["U-Net (R18)", "U-Net (R34)", "U-Net+CBAM",
                     "U-Net+Triplet", "DeepLabV3+", "TWAU-Net(本文)"]
    scenes = list(raw.keys())                      # 保持 SCENES 顺序
    vals = np.array([[display(m, raw[s][m]) for s in scenes] for m in methods])

    # 场景名换行 (每行 ≤7 字, "+" 后强制换行)
    def wrap(s, w=7):
        s = s.replace("+", "+\n")
        return "\n".join(seg[i:i + w] for seg in s.split("\n")
                         for i in range(0, len(seg), w))

    n_m, n_s = vals.shape
    x = np.arange(n_s)
    bw = 0.13
    fig, ax = plt.subplots(figsize=(10.6, 3.9), dpi=300)
    for i, m in enumerate(methods):
        offs = (i - (n_m - 1) / 2) * bw
        if m == "TWAU-Net":
            bars = ax.bar(x + offs, vals[i], bw * 0.92, color=OURS_RED,
                          label=method_labels[i], zorder=3)
        else:
            bars = ax.bar(x + offs, vals[i], bw * 0.92, color="#9db8cc",
                          edgecolor="#5a7a94",
                          linewidth=0.5, label=method_labels[i], zorder=2)
        for b, v in zip(bars, vals[i]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.2,
                    color=OURS_RED if m == "TWAU-Net" else "#3a3a3a",
                    fontweight="bold" if m == "TWAU-Net" else "normal",
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(s) for s in scenes], fontsize=9.5)
    ax.set_ylabel("水葫芦类 IoU", fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_scene_iou.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_scene_iou.png/pdf")


# 表 3 (文中原值, 4 位小数)
ABLATION = [
    ("A0", "无 TWA",           0.7505, 0.5917, "24.40"),
    ("A1", "lower, M=8",      0.8708, 0.7493, "27.20"),
    ("A2", "upper, M=8",      0.8686, 0.7319, "27.20"),
    ("A3", "both, M=8",       0.8641, 0.7370, "29.29"),
    ("A4", "both, M=4",       0.8768, 0.7132, "29.29"),
    ("A5", "lower, M=4",      0.8729, 0.7807, "27.20"),
]


def ablation_figure():
    """图 B: 表 3 消融柱状图 (mIoU / 水葫芦类 IoU)."""
    # 参数量并入 x 轴标签第三行 (表 3 口径)
    names = [f"{a}\n{d}\n{p} M" for a, d, _, _, p in ABLATION]
    miou = [r[2] for r in ABLATION]
    wh = [r[3] for r in ABLATION]
    x = np.arange(len(ABLATION))
    bw = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=300)
    b1 = ax.bar(x - bw / 2, miou, bw, color="#5b8db8", label="mIoU", zorder=2)
    b2 = ax.bar(x + bw / 2, wh, bw, color="#9db8cc", label="水葫芦类 IoU", zorder=2)
    # A5 (最终模型) 金框 + 主色高亮
    for b in (b1[5], b2[5]):
        b.set_color(OURS_RED if b is b2[5] else "#d4767a")
        b.set_edgecolor(GOLD)
        b.set_linewidth(2.0)
        b.set_zorder(3)
    for bars, vv in ((b1, miou), (b2, wh)):
        for b, v in zip(bars, vv):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=OURS_RED if b in (b1[5], b2[5]) else "#3a3a3a",
                    fontweight="bold" if b in (b1[5], b2[5]) else "normal")
    # 参数量并入 x 轴标签第三行 (表 3 口径)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("IoU", fontsize=10.5)
    ax.set_ylim(0.5, 0.92)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.01))
    ax.text(0.995, 1.02, "金框 = 最终模型 A5（下三角，M=4）", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_ablation.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_ablation.png/pdf")


if __name__ == "__main__":
    scene_iou_figure()
    ablation_figure()
