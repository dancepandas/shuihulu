# -*- coding: utf-8 -*-
"""流程图用 hero 图: 原图 + TWAU-Net 真实预测 (复用 build_vis_analysis 的
推理管线/预处理/配色, 与 fig_compare 完全一致, 保证口径统一)。

产出 doc/figs/hero/:
  <stem>_orig.png    原图 (原生分辨率)
  <stem>_gt.png      人工真值 (纯色)
  <stem>_pred.png    TWAU-Net 预测叠加 (alpha=0.40, 同 fig_compare)
  <stem>_predcolor.png 预测纯色图 (可当输出面板, 类似真值面板风格)
另拼三联预览 doc/figs/fig_hero_twaunet.png (原图|真值|预测, 带类图例)。
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\chengs\9.project\shuihulu"
SCRIPTS = os.path.join(ROOT, "twa_unet", "scripts")
FIGDIR = os.path.join(ROOT, "twa_unet", "doc", "figs", "hero")
import sys
sys.path.insert(0, SCRIPTS)
import build_vis_analysis as bva  # noqa: E402  (复用管线与配色)

SCENES = bva.SCENES  # [(fname, desc), ...]

# 用户指定色值 (RGB): 水面/水葫芦/硬质结构/岸边植被/其他水生植被
HERO_COLORS = {
    0: (0x4C, 0x90, 0xC0),
    1: (0xD9, 0x4F, 0x3D),
    2: (0x9A, 0x9A, 0x9A),
    3: (0x5F, 0xA0, 0x5A),
    4: (0xD9, 0xC2, 0x4A),
}


def colorize_hero(mask):
    out = np.zeros((*mask.shape, 3), np.float32)
    for c, col in HERO_COLORS.items():
        out[mask == c] = col
    return out


def overlay_bgr(img_bgr, mask, alpha=0.40):
    """调色板 RGB→BGR 后再混入 BGR 图, cv2.imwrite 颜色才正确."""
    img = img_bgr.astype(np.float32) * 0.85
    h, w = mask.shape
    ih, iw = img.shape[:2]
    mask_c = (cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
              if (ih, iw) != (h, w) else mask)
    colored = colorize_hero(mask_c)[:, :, ::-1]  # RGB -> BGR
    m = mask_c > 0
    img[m] = img[m] * (1 - alpha) + colored[m] * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    model = bva.build_twaunet()
    print("TWAU-Net loaded (A5 lower win4)")

    results = {}
    for fname, desc in SCENES:
        stem = os.path.splitext(fname)[0]
        img = bva.load_image(fname)
        gt = bva.load_gt(fname)
        pred = bva.predict(model, img)
        iou = float(bva.wh_iou(pred, gt))
        results[stem] = (fname, desc, img, gt, pred, iou)

        cv2.imwrite(os.path.join(FIGDIR, f"{stem}_orig.png"), img)
        cv2.imwrite(os.path.join(FIGDIR, f"{stem}_gt.png"),
                    colorize_hero(gt).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(os.path.join(FIGDIR, f"{stem}_pred.png"), overlay_bgr(img, pred))
        cv2.imwrite(os.path.join(FIGDIR, f"{stem}_predcolor.png"),
                    colorize_hero(pred).astype(np.uint8)[:, :, ::-1])
        print(f"[{stem}] {desc}  WH_IoU={iou:.3f}  -> 4 files")

    # ---- 三联预览 (推荐入流程图: p10_22843 水葫芦分布区) ----
    hero = "p10_22843"
    fname, desc, img, gt, pred, iou = results[hero]
    panels = [
        ("(a) 原图", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("(b) 人工真值", colorize_hero(gt).astype(np.uint8)),
        ("(c) TWAU-Net 预测", cv2.cvtColor(overlay_bgr(img, pred), cv2.COLOR_BGR2RGB)),
    ]
    AR = 4.0 / 3.0
    pw = 2.3
    ph = pw / AR
    gap = 0.10
    fig_w = 3 * pw + 2 * gap + 0.15
    fig_h = ph + 0.52
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    for j, (title, rgb_img) in enumerate(panels):
        x_in = 0.08 + j * (pw + gap)
        ax = fig.add_axes([x_in / fig_w, 0.42 / fig_h, pw / fig_w, ph / fig_h])
        ax.imshow(rgb_img, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text((x_in + pw / 2) / fig_w, 1 - 0.16 / fig_h, title,
                 fontsize=11, fontweight="bold", ha="center", va="center", color="black")
    handles = [plt.Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=np.array(HERO_COLORS[c], np.float32) / 255.0, markersize=10,
                          markeredgecolor="0.3", markeredgewidth=0.5)
               for c in range(bva.NUM_CLASSES)]
    fig.legend(handles, [bva.CLASS_CN[c] for c in range(bva.NUM_CLASSES)],
               loc="lower center", ncol=bva.NUM_CLASSES, frameon=False,
               fontsize=10, columnspacing=1.3, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.005))
    out = os.path.join(ROOT, "doc", "figs", "fig_hero_twaunet.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(ROOT, "doc", "figs", "fig_hero_twaunet.pdf"),
                bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
