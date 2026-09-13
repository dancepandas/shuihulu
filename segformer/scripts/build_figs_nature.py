# -*- coding: utf-8 -*-
"""
按 nature-figure 规范重绘论文图（Python backend）：
  图1  fig_framework.png   总体框架示意图（出版级，统一色系、直接标注、白底）
  图2-5 fig_qual_1..4.png  定性对比 image plate（原图/标注/本文方法/直接微调基线 四联）
                           ——统一面板尺寸、白边、掩膜半透明叠加、高DPI、面板直标
"""
import os, numpy as np, cv2, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from transformers import SegformerForSemanticSegmentation

ROOT = r"D:\chengs\9.project\shuihulu"
FIGDIR = os.path.join(ROOT, "segformer", "doc", "figs")
IMG = os.path.join(ROOT, "datasets", "hyacinth_seg", "images", "val")
MSK = os.path.join(ROOT, "datasets", "hyacinth_seg", "masks", "val")
SZ = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 低饱和蓝紫灰调色板（nature-figure NMI pastel 风格，统一方法族、无突兀撞色）
C_DARK = "#484878"    # 主色：深蓝紫（主框边框/标题/箭头）
C_MID = "#7884B4"     # 中蓝紫（次框边框）
C_SOFT = "#E4E4F0"    # 浅蓝紫（输入/输出框填充）
C_PINK = "#E4CCD8"    # 浅粉灰（设计模块/类别框填充）
C_INK = "#3B3B45"     # 正文文字（深灰带蓝）
C_WHITE = "#FFFFFF"
C_MASK = "#E76F51"    # 定性图掩膜：柔和珊瑚橙（绿植/蓝水上均清晰且不刺眼）

# 四类场景（验证集文件名）
SCENES = [
    ("fig_qual_1", "ls_0392.jpg",                  "密集分布"),
    ("fig_qual_2", "v9_01e7585b-t_20260717_092742_971b-DJI_20260717070620_0046_V.jpg", "稀疏分布"),
    ("fig_qual_3", "ls_0489.jpg",                  "小目标"),
    ("fig_qual_4", "v9_af0fd03f-t_20260717_092742_971b-DJI_20260717070249_0010_V.jpg", "岸基植被干扰"),
]

def load_img(fname):
    img = cv2.cvtColor(cv2.imread(os.path.join(IMG, fname)), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)

def load_gt(fname):
    return cv2.imread(os.path.join(MSK, os.path.splitext(fname)[0] + ".png"), cv2.IMREAD_GRAYSCALE)

def predict(model, img):
    H, W = img.shape[:2]
    r = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = nn.functional.interpolate(model(pixel_values=t).logits,
                                           size=(H, W), mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy()

def overlay(img, mask):
    """图像轻度压暗 + 掩膜区域珊瑚橙半透明叠加 + 深色描边勾勒边界"""
    out = (img * 0.62).astype(np.float32)
    col = np.array([0xE7, 0x6F, 0x51], np.float32)   # C_MASK #E76F51
    edge = np.array([0xB0, 0x43, 0x28], np.float32)  # 深珊瑚描边
    m = mask > 0
    out[m] = out[m] * 0.25 + col * 0.75
    # 掩膜边界描边（2 px），让碎片边界在亮背景下仍清晰
    mu = m.astype(np.uint8)
    er = cv2.erode(mu, np.ones((3, 3), np.uint8))
    bd = (mu - er) > 0
    out[bd] = edge
    return np.clip(out, 0, 255).astype(np.uint8)

def build_plate(out_name, fname, img, gt, pred_ours, pred_v2):
    panel = 700  # 每面板像素高度
    def prep(im):
        h, w = im.shape[:2]
        nw = int(w * panel / h)
        return cv2.resize(im, (nw, panel), interpolation=cv2.INTER_AREA)
    a = prep(img.astype(np.uint8))
    b = prep(overlay(img, gt))
    c = prep(overlay(img, pred_ours))
    d = prep(overlay(img, pred_v2))
    panels = [a, b, c, d]
    labels = ["(a) 原图", "(b) 人工真值", "(c) LWH-Seg 预测", "(d) SegFormer-B2-FT 预测"]
    W = sum(p.shape[1] for p in panels) + 3 * 6 + 10
    H = panel + 34
    fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    x = 5
    for p, lab in zip(panels, labels):
        ax.imshow(p, extent=[x, x + p.shape[1], 0, p.shape[0]], aspect="auto")
        ax.text(x + 10, p.shape[0] - 16, lab, color="white", fontsize=15,
                fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45))
        x += p.shape[1] + 6
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    fig.savefig(os.path.join(FIGDIR, out_name + ".png"), dpi=300,
                bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(os.path.join(FIGDIR, out_name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def build_framework():
    fig = plt.figure(figsize=(12, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 120); ax.set_ylim(0, 44)

    def box(x, y, w, h, text, fc, ec, fs=13, tc=C_INK, lw=1.4, weight="bold"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                           fc=fc, ec=ec, lw=lw)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=18, color=C_MID, lw=2.0)
        ax.add_patch(a)

    # 输入
    box(1, 19, 15, 6.5, "无人机\n河道影像", C_SOFT, C_DARK, fs=12)
    # SegFormer-B2 主流程（英雄框，白底深蓝紫粗边）
    box(22, 15, 34, 14, "SegFormer-B2\n（MiT-B2 编码器 + 全 MLP 解码器）", C_WHITE, C_DARK, fs=13, tc=C_DARK, lw=2.2)
    # 输出
    box(62, 17, 18, 10, "六类像素级\n分割结果", C_SOFT, C_DARK, fs=12)
    # 输出类别列表
    box(85, 14, 32, 16, "背景\n船只 / 桥梁 / 岸基建筑\n水葫芦 / 树木", C_PINK, C_MID, fs=11, tc=C_INK)

    arrow(16.5, 22.5, 21.5, 22.5)
    arrow(56.5, 22.5, 61.5, 22.5)
    arrow(80.5, 22.5, 84.5, 22.5)

    # 三项针对性设计（下方）
    designs = [
        (6, "数据层面\n多源融合 + 增强"),
        (43, "损失层面\n加权CE + Dice联合损失"),
        (80, "训练策略\n两段式训练"),
    ]
    for x, txt in designs:
        box(x, 2.5, 26, 8, txt, C_PINK, C_MID, fs=11, tc=C_INK)
    # 设计 → 主干 的箭头
    for x in [19, 56, 93]:
        arrow(x, 11, x, 14.5)

    ax.text(60, 40, "单次前向 · 端到端 · 无需检测框提示",
            ha="center", fontsize=12, color=C_DARK, fontweight="bold")
    fig.savefig(os.path.join(FIGDIR, "fig_framework.png"), dpi=300,
                bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(os.path.join(FIGDIR, "fig_framework.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    # 加载模型
    m_ours = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls+v9")).to(dev).eval()
    m_v2 = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(ROOT, "runs", "segformer", "segformer_b2_v2")).to(dev).eval()
    for out_name, fname, desc in SCENES:
        img = load_img(fname)
        gt = load_gt(fname)
        pred_ours = predict(m_ours, img)
        pred_v2 = predict(m_v2, img)
        build_plate(out_name, fname, img, gt, pred_ours, pred_v2)
        wh_gt = (gt == 4).sum(); wh_o = (pred_ours == 4).sum(); wh_v2 = (pred_v2 == 4).sum()
        print(f"{out_name} ({desc}): GT WH {wh_gt}, 本文 WH {wh_o}, 基线 WH {wh_v2}")
    build_framework()
    print("figures saved to", FIGDIR)

if __name__ == "__main__":
    main()
