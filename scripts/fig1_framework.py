# -*- coding: utf-8 -*-
"""
fig1_framework — 本文方法总体框架（paper-figure-generation skill, P1 pipeline-with-hero）
CS 论文结构图风格：MiT-B2 四阶段画为 PlotNeuralNet 式三维特征块（空间递减、通道递增、
标注块数 ×3/×4/×6/×3），解码器四支路汇合，六类分数图画为类色薄板；
两端嵌入真实对象（真实河道影像 + SegFormer-B2 真实推理掩膜），不做空盒子流程图。
后端：matplotlib（环境无 LaTeX 中文链，§M reproducible matplotlib fallback）。
产出：outputs/figures/fig1_framework.{pdf,png,svg}
"""
import os, sys
import numpy as np, cv2, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.colors import to_rgb
from transformers import SegformerForSemanticSegmentation

SKILL_SCRIPTS = r"D:\chengs\9.project\shuihulu\.claude\skills\paper-figure-generation\scripts"
sys.path.insert(0, SKILL_SCRIPTS)
from paperfig import paper_style, save  # 共享样式预设（唯一来源）

ROOT = r"D:\chengs\9.project\shuihulu"
IMG_PATH = os.path.join(ROOT, "datasets", "hyacinth_ls", "images", "train", "ls_0385.jpg")
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls+v9")
SZ = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_NAMES = ["背景", "船只", "桥梁", "岸基建筑", "水葫芦", "树木"]

# ---- 统一低饱和色系（与定性图一致的视觉语言） ----
C_DARK = "#484878"   # 主色：英雄组/标题
C_MID = "#7884B4"    # 次级边框与辅助箭头
C_SOFT = "#F4F5FA"   # 浅填充
C_INK = "#3B3B45"    # 正文文字
# 六类掩膜色（低饱和、色盲可辨；水葫芦珊瑚橙与定性图一致）
CLASS_COLORS = ["#7F9AAE", "#D9A05B", "#A26769", "#8E86A8", "#E76F51", "#6E9E7E"]
# MiT-B2 四阶段特征块色（单调加深）
STAGE_COLORS = ["#5F6FA6", "#4E5C8E", "#414C78", "#353D61"]
# MiT-B2 结构参数：分辨率 / 通道数 / Transformer 块数
STAGE_RES = ["1/4", "1/8", "1/16", "1/32"]
STAGE_CH = [64, 128, 320, 512]
STAGE_BLK = [3, 4, 6, 3]

def rgb(hex_str):
    return np.array(to_rgb(hex_str), np.float32)

def mix(c, target, f):
    return tuple((1 - f) * rgb(c) + f * rgb(target))

def slab(ax, x, y, w, h, d, fc, ec="white", lw=0.7, z=3):
    """PlotNeuralNet 式三维特征块：正面 + 顶面 + 侧面"""
    dx, dy = d, d * 0.55
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=mix(fc, "#FFFFFF", 0.28), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=mix(fc, "#000000", 0.22), ec=ec, lw=lw, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
    return dx, dy

def infer(model, img):
    H, W = img.shape[:2]
    r = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = nn.functional.interpolate(model(pixel_values=t).logits, size=(H, W),
                                           mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy()

def main():
    paper_style(font="cn")  # 中文论文：SimSun 预设
    # 可复现：模型配置与数据均从文件路径读取
    with open(os.path.join(MODEL_PATH, "config.json"), "r", encoding="utf-8") as f:
        _ = f.read()
    # --- 真实对象：输入影像 + 模型推理掩膜 ---
    img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB).astype(np.float32)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    with torch.no_grad():
        pred = infer(model, img)
    # 掩膜彩色叠加：背景弱叠加保纹理，水葫芦强叠加突出目标
    colors = [rgb(c) for c in CLASS_COLORS]
    alpha = [0.28, 0.50, 0.50, 0.50, 0.70, 0.50]
    disp_w, disp_h = 640, 480
    d_img = cv2.resize(img.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    d_pred = cv2.resize(pred, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    overlay = d_img.copy()
    for c in range(6):
        m = d_pred == c
        overlay[m] = (1 - alpha[c]) * overlay[m] + alpha[c] * colors[c] * 255.0
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # --- 画布 ---
    fig = plt.figure(figsize=(13.5, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    axm = fig.add_axes([0, 0, 1, 1]); axm.set_axis_off(); axm.set_xlim(0, 100); axm.set_ylim(0, 100)

    # --- 输入面板（真实影像） ---
    ax_in = fig.add_axes([0.015, 0.34, 0.16, 0.48])
    ax_in.imshow(d_img.astype(np.uint8))
    ax_in.set_xticks([]); ax_in.set_yticks([])
    ax_in.set_xlabel("空间尺寸 / 像素", fontsize=7.5, color="#666666")
    ax_in.set_ylabel("空间尺寸 / 像素", fontsize=7.5, color="#666666")
    ax_in.tick_params(length=0)
    for sp in ax_in.spines.values():
        sp.set_color("#AAAAAA"); sp.set_linewidth(0.8)
    axm.text(9.5, 30.0, "输入：无人机河道影像", ha="center", fontsize=11, color=C_INK, fontweight="bold")
    axm.text(9.5, 26.0, "512×512", ha="center", fontsize=9, color=C_INK)

    # --- MiT-B2 编码器：四阶段三维特征块（英雄组） ---
    CY = 62.0                       # 主链高度
    xs = [24.5, 33.5, 42.5, 51.5]
    ws, hs, ds = 3.0, [24, 18, 13.5, 10], [1.2, 1.7, 2.2, 2.8]
    for x, h, d, c, res, ch, nb in zip(xs, hs, ds, STAGE_COLORS, STAGE_RES, STAGE_CH, STAGE_BLK):
        y = CY - h / 2
        dx, dy = slab(axm, x, y, ws, h, d, c)
        axm.text(x + (ws + dx) / 2, y + h + dy + 2.2, f"{res} · C={ch} · ×{nb}",
                 ha="center", fontsize=8.5, color=C_INK)
    for i in range(3):              # 级间箭头
        x0 = xs[i] + ws + ds[i] + 0.5
        axm.add_patch(FancyArrowPatch((x0, CY), (xs[i + 1] - 0.5, CY), arrowstyle="-|>",
                                      mutation_scale=12, color=C_DARK, lw=1.6, zorder=2))
    # 编码器分组虚线框
    axm.add_patch(FancyBboxPatch((22.5, 45), 38.0, 42, boxstyle="round,pad=0.4", fill=False,
                                 ec=C_MID, lw=1.2, ls=(0, (4, 3)), zorder=1))
    axm.text(41.5, 83.6, "MiT-B2 编码器（层级特征提取）", ha="center", fontsize=10,
             color=C_DARK, fontweight="bold")

    # --- 全 MLP 解码器：四支路 MLP → 融合 ---
    mlp_cx = [x + ws / 2 for x in xs]           # 26, 35, 44, 53
    for cx in mlp_cx:
        axm.add_patch(FancyBboxPatch((cx - 2.2, 31.4), 4.4, 5.6, boxstyle="round,pad=0.25",
                                     fc="white", ec=C_MID, lw=1.3, zorder=3))
        axm.text(cx, 34.2, "MLP ↑", ha="center", va="center", fontsize=8.5, color=C_DARK,
                 fontweight="bold", zorder=4)
    for x, h, cx in zip(xs, hs, mlp_cx):        # 阶段 → MLP 下行箭头
        axm.add_patch(FancyArrowPatch((cx, CY - h / 2 - 0.5), (cx, 37.6), arrowstyle="-|>",
                                      mutation_scale=10, color=C_MID, lw=1.3, zorder=2))
    # 融合块
    axm.add_patch(FancyBboxPatch((60, 30.8), 9, 6.4, boxstyle="round,pad=0.3", fc=C_SOFT,
                                 ec=C_DARK, lw=1.6, zorder=3))
    axm.text(64.5, 34.0, "特征融合\n+ MLP", ha="center", va="center", fontsize=8.5,
             color=C_DARK, fontweight="bold", zorder=4)
    # 四支路 → 总线 → 融合（汇流排式，不穿盒）
    BUS = 29.9
    for cx in mlp_cx:
        axm.plot([cx, cx], [31.2, BUS], color=C_MID, lw=1.2, zorder=2)
    axm.plot([mlp_cx[0], 64.5], [BUS, BUS], color=C_MID, lw=1.2, zorder=2)
    axm.add_patch(FancyArrowPatch((64.5, BUS), (64.5, 30.5), arrowstyle="-|>",
                                  mutation_scale=10, color=C_MID, lw=1.3, zorder=2))
    # 解码器分组虚线框
    axm.add_patch(FancyBboxPatch((21.5, 28.0), 49.0, 12.0, boxstyle="round,pad=0.4", fill=False,
                                 ec=C_MID, lw=1.2, ls=(0, (4, 3)), zorder=1))
    axm.text(23.2, 28.9, "全 MLP 解码器", ha="left", va="center", fontsize=9.5,
             color=C_DARK, fontweight="bold", zorder=4)

    # --- 六类分数图：6 片类色薄板 ---
    sx, sw, sd, sh, gap = 73.5, 3.0, 1.3, 1.9, 0.4
    total = 6 * sh + 5 * gap
    for c in range(6):
        slab(axm, sx, CY - total / 2 + c * (sh + gap), sw, sh, sd, CLASS_COLORS[5 - c])
    axm.text(sx + (sw + sd) / 2, CY + total / 2 + 2.2, "六类分数图", ha="center",
             fontsize=8.5, color=C_INK)
    axm.add_patch(FancyArrowPatch((69.2, 36.5), (sx - 0.3 + sd / 2, CY - total / 2 - 0.5),
                                  arrowstyle="-|>", mutation_scale=11, color=C_DARK, lw=1.5,
                                  zorder=2))
    axm.add_patch(FancyArrowPatch((sx + sw + sd + 0.4, CY), (81.4, CY), arrowstyle="-|>",
                                  mutation_scale=13, color=C_DARK, lw=1.7, zorder=2))

    # --- 输出面板（真实掩膜叠加） ---
    ax_out = fig.add_axes([0.825, 0.34, 0.16, 0.48])
    ax_out.imshow(overlay)
    ax_out.set_xticks([]); ax_out.set_yticks([])
    ax_out.set_xlabel("空间尺寸 / 像素", fontsize=7.5, color="#666666")
    ax_out.tick_params(length=0)
    for sp in ax_out.spines.values():
        sp.set_color("#AAAAAA"); sp.set_linewidth(0.8)
    axm.text(90.5, 30.0, "输出：六类像素级分割", ha="center", fontsize=11, color=C_INK, fontweight="bold")
    axm.text(90.5, 26.0, "27.35 M 参数 · 109.5 MB · 7.4 ms/图", ha="center", fontsize=8.5, color=C_INK)
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=7, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(6)]
    leg = axm.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.505, 0.020),
                     ncol=6, fontsize=8, frameon=False, handletextpad=0.35, columnspacing=1.1)
    leg.set_zorder(10)

    # --- 输入/输出 → 主干 箭头 ---
    axm.add_patch(FancyArrowPatch((18.2, CY), (24.0, CY), arrowstyle="-|>", mutation_scale=14,
                                  color=C_DARK, lw=1.8, zorder=2))

    # --- 三类针对性设计（底部，箭头↑汇入解码器） ---
    designs = [(26, "数据层面", "多源融合 + 针对性增强"),
               (43.5, "损失层面", "加权 CE + Dice 联合损失"),
               (61, "训练策略", "两段式训练")]
    for x, t1, t2 in designs:
        axm.add_patch(FancyBboxPatch((x, 5.5), 16.5, 12.5, boxstyle="round,pad=0.4", fc=C_SOFT,
                                     ec=C_MID, lw=1.4, zorder=3))
        axm.text(x + 8.25, 14.0, t1, ha="center", va="center", fontsize=9.5, color=C_DARK,
                 fontweight="bold", zorder=4)
        axm.text(x + 8.25, 9.3, t2, ha="center", va="center", fontsize=8.5, color=C_INK, zorder=4)
        axm.add_patch(FancyArrowPatch((x + 8.25, 18.5), (x + 8.25, 27.6), arrowstyle="-|>",
                                      mutation_scale=11, color=C_MID, lw=1.5, zorder=2))

    save(fig, "fig1_framework")
    plt.close(fig)
    print("saved outputs/figures/fig1_framework.{pdf,png,svg}")

if __name__ == "__main__":
    main()
