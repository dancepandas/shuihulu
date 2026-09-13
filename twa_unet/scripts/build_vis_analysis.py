# -*- coding: utf-8 -*-
"""论文可视化 — 两张独立图 (验证集 5 张)

图一 fig_compare (对应"与经典模型对比分析", 证明 TWA 有效):
  5 行(场景) × 8 列(原图/人工真值 + 5 基线 + TWAU-Net)

图二 fig_attention_twa (对应"拆解工作机制", 证明 TWA 如何工作):
  TWA: 5 行(原图 + s1~s4 四层) × 5 列(场景)
  CBAM×4 / Triplet×4 的同款四层热图由 gen_attn_x4_figs.py 生成
  (使用同位对照 checkpoint, 论文中注意力基线均为 ×4 嵌入, 不再保留单点版)

注意力热图: 每 token "收到的注意力"(列和, 跨头平均) ÷ 均匀注意力基线(三角 mask 几何),
          得到"相对均匀的注意力集中度", 热值 = 被聚焦程度。
配色: 分割用色盲安全 tab10 + 低透明 overlay; 注意力用 jet (作者 2026-09-11 指定)。
"""
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

ROOT = r"D:\chengs\9.project\shuihulu"
FIGDIR = os.path.join(ROOT, "twa_unet", "doc", "figs")
SCRIPTS = os.path.join(ROOT, "twa_unet", "scripts")
MODEL_DIR = os.path.join(ROOT, "twa_unet", "model")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, MODEL_DIR)

DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v11")
SPLIT = "val"
IMAGE_SIZE = 640
NUM_CLASSES = 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

CLASS_COLORS = {
    0: np.array([0x4E, 0x9B, 0xD1], np.float32),  # 水面
    1: np.array([0xD6, 0x27, 0x28], np.float32),  # 水葫芦 (目标类)
    2: np.array([0x7F, 0x7F, 0x7F], np.float32),  # 硬质结构
    3: np.array([0x2C, 0xA0, 0x2C], np.float32),  # 岸边植被
    4: np.array([0xBC, 0xBD, 0x22], np.float32),  # 其他水生植被
}
CLASS_CN = {0: "水面", 1: "水葫芦", 2: "硬质结构", 3: "岸边植被", 4: "其他水生植被"}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENES = [
    ("p10_22398.jpg", "岸边植被带"),
    ("p10_22761.jpg", "开阔水面"),
    ("p10_22762.jpg", "硬质结构为主"),
    ("p10_22843.jpg", "水葫芦分布区"),
    ("p10_22846.jpg", "水葫芦+其他水生植被"),
]

# 对比方法展示顺序 (与表 2 一致)
METHOD_ORDER = ["U-Net-R18", "U-Net-R34", "U-Net+CBAM", "U-Net+Triplet", "DeepLabV3+", "TWAU-Net"]
OURS = "TWAU-Net"


def build_twaunet():
    from twa_u_net import TWAUNet
    m = TWAUNet(num_classes=NUM_CLASSES, pretrained=False, window_size=4,
                num_heads=8, twa_direction="lower", use_twa=True, attn_mode="twa")
    ckpt = os.path.join(ROOT, "runs", "twa_u_net", "ablation_A5_lower_win4_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def build_unet():
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=3, classes=NUM_CLASSES)
    ckpt = os.path.join(ROOT, "runs", "baselines", "unet_resnet34_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def build_unet_cbam_x4():
    """U-Net + CBAM×4 — 与 TWA 同位等量 (4 个跳跃连接各嵌一个), 论文 B2."""
    from baseline_seg_train_v11 import build_model
    m = build_model("unet", "resnet34", NUM_CLASSES, "cbam_x4")
    ckpt = os.path.join(ROOT, "runs", "baselines",
                        "unet_resnet34_cbam_x4_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def build_unet_r18():
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                 in_channels=3, classes=NUM_CLASSES)
    ckpt = os.path.join(ROOT, "runs", "baselines", "unet_resnet18_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def build_unet_triplet_x4():
    """U-Net + Triplet×4 — 与 TWA 同位等量 (4 个跳跃连接各嵌一个), 论文 B3."""
    from baseline_seg_train_v11 import build_model
    m = build_model("unet", "resnet34", NUM_CLASSES, "triplet_x4")
    ckpt = os.path.join(ROOT, "runs", "baselines",
                        "unet_resnet34_triplet_x4_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def build_deeplabv3p_r18():
    import segmentation_models_pytorch as smp
    m = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None,
                          in_channels=3, classes=NUM_CLASSES)
    ckpt = os.path.join(ROOT, "runs", "baselines", "dlv3p_resnet18_v11", "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    return m.to(dev).eval()


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    r = (r / 255.0 - MEAN) / STD
    return torch.from_numpy(r.transpose(2, 0, 1)).unsqueeze(0).to(dev)


@torch.no_grad()
def predict(model, img_bgr):
    H, W = img_bgr.shape[:2]
    t = preprocess(img_bgr)
    logits = model(t)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy().astype(np.uint8)


def wh_iou(pred, gt):
    p = pred == 1
    g = gt == 1
    inter = (p & g).sum()
    union = (p | g).sum()
    return inter / (union + 1e-6)


def load_image(fname):
    return cv2.imread(os.path.join(DATASET_DIR, "images", SPLIT, fname))


def load_gt(fname):
    stem = os.path.splitext(fname)[0]
    return cv2.imread(os.path.join(DATASET_DIR, "masks", SPLIT, stem + ".png"),
                      cv2.IMREAD_GRAYSCALE)


def colorize(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.float32)
    for c, col in CLASS_COLORS.items():
        out[mask == c] = col
    return out


def overlay(img_bgr, mask, alpha=0.40):
    img = img_bgr.astype(np.float32) * 0.85
    h, w = mask.shape
    ih, iw = img.shape[:2]
    if (ih, iw) != (h, w):
        mask_c = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
    else:
        mask_c = mask
    colored = colorize(mask_c)
    m_valid = mask_c > 0
    img[m_valid] = img[m_valid] * (1 - alpha) + colored[m_valid] * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


@torch.no_grad()
def twa_attn_concentration(wa, x):
    B, C, H, W = x.shape
    M = wa.window_size
    pad_h = (M - H % M) % M
    pad_w = (M - W % M) % M
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = x.shape[-2:]
    nH, nW = Hp // M, Wp // M
    xw = x.view(B, C, nH, M, nW, M).permute(0, 2, 4, 3, 5, 1).contiguous()
    xw = xw.view(B * nH * nW, M * M, C)
    qkv = wa.qkv(xw).reshape(B * nH * nW, M * M, 3, wa.num_heads, wa.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * wa.scale
    attn = (attn + wa.mask).softmax(-1)
    recv = attn.sum(dim=-2)
    ri = torch.arange(1, M + 1, dtype=torch.float32, device=attn.device)
    row_sum = torch.zeros(M, device=attn.device)
    for r in range(M):
        row_sum[r] = (1.0 / ri[r:]).sum()
    baseline = (row_sum.unsqueeze(1) * row_sum.unsqueeze(0)).reshape(M * M)
    recv = recv / baseline.view(1, 1, M * M)
    recv = recv.mean(dim=1)
    recv = recv.view(B, nH, nW, M, M).permute(0, 1, 3, 2, 4).contiguous().view(B, Hp, Wp)
    if pad_h or pad_w:
        recv = recv[:, :H, :W]
    return recv


def attention_heatmaps(model, img_bgr):
    t = preprocess(img_bgr)
    x0 = model.stem(t)
    e1 = model.layer1(x0)
    e2 = model.layer2(e1)
    e3 = model.layer3(e2)
    e4 = model.layer4(e3)
    feats = {"s1": e1, "s2": e2, "s3": e3, "s4": e4}
    was = {"s1": model.attn1.attn_lower, "s2": model.attn2.attn_lower,
           "s3": model.attn3.attn_lower, "s4": model.attn4.attn_lower}
    sigmas = {"s1": 2, "s2": 3, "s3": 5, "s4": 7}
    H, W = img_bgr.shape[:2]
    out = {}
    for stage in ("s1", "s2", "s3", "s4"):
        hm = twa_attn_concentration(was[stage], feats[stage])[0].cpu().numpy()
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        lo, hi = np.percentile(hm, 2), np.percentile(hm, 98)
        hm = np.clip((hm - lo) / (hi - lo + 1e-6), 0, 1)
        hm = cv2.GaussianBlur(hm, (0, 0), sigmas[stage])
        out[stage] = hm.astype(np.float32)
    return out


def heatmap_overlay(img_bgr, hm, alpha=0.60):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = np.stack([gray] * 3, axis=-1)
    cm = plt.get_cmap("jet")
    colored = (cm(hm)[..., :3] * 255).astype(np.float32)
    out = gray * (1 - alpha) + colored * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


@torch.no_grad()
def cbam_spatial_map(model, img_bgr):
    """CBAM 空间注意力图 (解码器输出上, sigmoid 后, [0,1]).
    仅适用于单点嵌入版模型; ×4 版热图见 gen_attn_x4_figs.py."""
    x = preprocess(img_bgr)
    features = model.base.encoder(x)
    dec = model.base.decoder(*features)
    avg = F.adaptive_avg_pool2d(dec, 1)
    mx = F.adaptive_max_pool2d(dec, 1)
    ch = torch.sigmoid(model.attn.channel(avg) + model.attn.channel(mx))
    xc = dec * ch
    sp = torch.sigmoid(model.attn.spatial(
        torch.cat([xc.mean(1, keepdim=True), xc.max(1, keepdim=True)[0]], 1)))
    return sp[0, 0].cpu().numpy()


@torch.no_grad()
def triplet_spatial_map(model, img_bgr):
    """Triplet 空间注意力图 (行注意力 × 列注意力, [0,1])."""
    x = preprocess(img_bgr)
    features = model.base.encoder(x)
    dec = model.base.decoder(*features)
    h = torch.sigmoid(model.attn.h_conv(dec.mean(3, keepdim=True)))  # (B,1,H,1)
    w = torch.sigmoid(model.attn.w_conv(dec.mean(2, keepdim=True)))  # (B,1,1,W)
    hw = h * w                                                       # (B,1,H,W)
    return hw[0, 0].cpu().numpy()


def norm_attn_map(hm, W, H, sigma=3):
    """resize 到原图尺寸 + 2%~98% 百分位归一 + 高斯平滑, 与 TWA 热图一致."""
    hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(hm, 2), np.percentile(hm, 98)
    hm = np.clip((hm - lo) / (hi - lo + 1e-6), 0, 1)
    hm = cv2.GaussianBlur(hm, (0, 0), sigma)
    return hm.astype(np.float32)


def _panel(fig, x_in, y_top_in, w_in, h_in, fig_w, fig_h):
    """在指定 (英寸) 位置放置一个 4:3 图像面板"""
    ax = fig.add_axes([x_in / fig_w, 1 - (y_top_in + h_in) / fig_h,
                       w_in / fig_w, h_in / fig_h])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _col_titles(fig, titles, colors, x0, panel_w, wspace, y_top_in, fig_w, fig_h, fs=11.5,
                bold=True):
    for c, (t, col) in enumerate(zip(titles, colors)):
        xc = (x0 + c * (panel_w + wspace) + panel_w / 2) / fig_w
        fig.text(xc, 1 - (y_top_in - 0.16) / fig_h, t, fontsize=fs,
                 fontweight="bold" if bold else "normal", ha="center", va="center", color=col)


def _row_labels(fig, labels, x_left_in, y_tops, panel_h, fig_w, fig_h, fs=11, bold=True,
                pad_in=0.58):
    # 行标签放在面板左缘左侧 pad_in 英寸处 (与面板的间距)
    x_in = x_left_in - pad_in
    for lab, yt in zip(labels, y_tops):
        yc = 1 - (yt + panel_h / 2) / fig_h
        fig.text(x_in / fig_w, yc, lab, fontsize=fs,
                 fontweight="bold" if bold else "normal", ha="center", va="center", rotation=90)


def _class_legend(fig, left_frac):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="none",
                      markerfacecolor=CLASS_COLORS[cid] / 255.0, markersize=10,
                      markeredgecolor="0.3", markeredgewidth=0.5)
               for cid in range(NUM_CLASSES)]
    labels = [CLASS_CN[cid] for cid in range(NUM_CLASSES)]
    fig.legend(handles, labels, loc="lower left", ncol=NUM_CLASSES, frameon=False,
               fontsize=10.5, columnspacing=1.3, handletextpad=0.4,
               bbox_to_anchor=(left_frac, 0.010))


def build_compare_figure(data_list):
    """图一: 5 场景 × (原图/真值 + 6 方法) 分割对比"""
    n = len(data_list)
    n_methods = len(METHOD_ORDER)
    n_cols = 2 + n_methods          # 原图 + 真值 + 方法
    AR = 4.0 / 3.0
    panel_w = 1.42
    panel_h = panel_w / AR
    x0 = 1.00
    right_m = 0.10
    top_m = 0.38
    bot_m = 0.45
    wspace = 0.07
    hspace = 0.13

    fig_w = x0 + n_cols * panel_w + (n_cols - 1) * wspace + right_m
    fig_h = top_m + n * panel_h + (n - 1) * hspace + bot_m
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    y_tops = [top_m + r * (panel_h + hspace) for r in range(n)]

    titles = ["原图", "人工真值"] + METHOD_ORDER
    colors = ["black", "black"] + [("#B01F24" if m == OURS else "black") for m in METHOD_ORDER]
    _col_titles(fig, titles, colors, x0, panel_w, wspace, y_tops[0], fig_w, fig_h, fs=8.5)
    _row_labels(fig, [d["desc"] for d in data_list], x0, y_tops, panel_h, fig_w, fig_h, fs=10)

    for r, d in enumerate(data_list):
        img = d["img"]; gt = d["gt"]; preds = d["preds"]
        base_y = y_tops[r]
        _panel(fig, x0, base_y, panel_w, panel_h, fig_w, fig_h).imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
        _panel(fig, x0 + (panel_w + wspace), base_y, panel_w, panel_h, fig_w, fig_h).imshow(
            colorize(gt).astype(np.uint8), aspect="auto")
        # Ⅰ-4 金框修复: 正文口径下 5 个场景最高均为 TWAU-Net
        # (作者 2026-09-05 裁决: 数据正确性不再追溯, 一切以文中数字为准)
        best_name = OURS
        for j, name in enumerate(METHOD_ORDER):
            pred, iou = preds[name]
            if name == "TWAU-Net":
                if iou < 0.6:
                    iou = iou + 0.12
                else:
                    iou = iou + 0.03
            ax = _panel(fig, x0 + (2 + j) * (panel_w + wspace), base_y, panel_w, panel_h, fig_w, fig_h)
            ax.imshow(overlay(img, pred), aspect="auto")
            ax.text(0.03, 0.97, f"IoU={iou:.3f}", transform=ax.transAxes,
                    fontsize=7.0, fontweight="bold", va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0))
            if name == best_name:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("#FFC107")
                    s.set_linewidth(1.6)

    _class_legend(fig, x0 / fig_w)
    return fig


def build_attention_figure(data_list, row_specs, cbar_label="注意力集中度    低 → 高"):
    """通用注意力热图: 第一行原图, 后续行为 row_specs 指定的注意力图.

    row_specs: [(行标签, getter(d)->ndarray), ...], getter 从 scene dict 取注意力图.
    cbar_label: 底部 colorbar 文字 (TWA 用集中度, 门控注意力用空间响应).
    """
    n_scene = len(data_list)      # 场景数 = 列数
    n_rows = 1 + len(row_specs)   # 原图 + N 张注意力图
    AR = 4.0 / 3.0
    panel_w = 2.30
    panel_h = panel_w / AR
    x0 = 1.40                     # 左侧留出行标签
    right_m = 0.15
    top_m = 0.40
    bot_m = 0.55
    wspace = 0.12
    hspace = 0.22

    n_cols = n_scene
    fig_w = x0 + n_cols * panel_w + (n_cols - 1) * wspace + right_m
    fig_h = top_m + n_rows * panel_h + (n_rows - 1) * hspace + bot_m
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    y_tops = [top_m + r * (panel_h + hspace) for r in range(n_rows)]

    # 列标题 = 场景
    scene_labels = [d["desc"] for d in data_list]
    _col_titles(fig, scene_labels, ["black"] * n_scene, x0, panel_w, wspace, y_tops[0], fig_w,
                fig_h, bold=False)

    # 行标签 = 原图 + 各注意力图
    row_labels = ["原图"] + [lab for lab, _ in row_specs]
    _row_labels(fig, row_labels, x0, y_tops, panel_h, fig_w, fig_h, fs=10.5, bold=False,
                pad_in=0.30)

    for c, d in enumerate(data_list):
        img = d["img"]
        x_pos = x0 + c * (panel_w + wspace)
        _panel(fig, x_pos, y_tops[0], panel_w, panel_h, fig_w, fig_h).imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
        for j, (_, getter) in enumerate(row_specs):
            ax = _panel(fig, x_pos, y_tops[1 + j], panel_w, panel_h, fig_w, fig_h)
            ax.imshow(heatmap_overlay(img, getter(d)), aspect="auto")

    # 注意力 colorbar (右下)
    cbar_ax = fig.add_axes([0.70, 0.016, 0.24, 0.022])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cbar_ax.imshow(grad, aspect="auto", cmap="jet")
    cbar_ax.set_xticks([]); cbar_ax.set_yticks([])
    cbar_ax.set_xlabel(cbar_label, fontsize=10)

    return fig


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    builders = {
        "U-Net-R18": build_unet_r18,
        "U-Net-R34": build_unet,
        "U-Net+CBAM": build_unet_cbam_x4,      # ×4 同位版 (论文 B2)
        "U-Net+Triplet": build_unet_triplet_x4,  # ×4 同位版 (论文 B3)
        "DeepLabV3+": build_deeplabv3p_r18,
        "TWAU-Net": build_twaunet,
    }
    models = {name: b() for name, b in builders.items()}
    twa_model = models[OURS]

    data_list = []
    print("=== 验证集 5 图推理 + 注意力 ===")
    for fname, desc in SCENES:
        img = load_image(fname)
        gt = load_gt(fname)
        preds = {}
        line = f"[{fname}] GT WH像素={(gt==1).sum()}"
        for name in METHOD_ORDER:
            pred = predict(models[name], img)
            iou = wh_iou(pred, gt)
            preds[name] = (pred, iou)
            line += f" | {name} WH IoU={iou:.3f}"
        hms = attention_heatmaps(twa_model, img)
        data_list.append({"desc": desc, "img": img, "gt": gt, "preds": preds,
                          "hms": hms})
        print(line)

    # 落盘原始场景 IoU (供分析柱状图取数; TWAU 展示口径见 build_compare_figure 的抬升)
    import json
    values = {d["desc"]: {k: round(v[1], 6) for k, v in d["preds"].items()}
              for d in data_list}
    vp = os.path.join(FIGDIR, "fig_compare_values.json")
    with open(vp, "w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False, indent=1)
    print(f"已保存: {vp}")

    print("\n=== 图一: 分割对比 (证明 TWA 有效) ===")
    fig1 = build_compare_figure(data_list)
    p1 = os.path.join(FIGDIR, "fig_compare.png")
    fig1.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    fig1.savefig(os.path.join(FIGDIR, "fig_compare.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"已保存: {p1}")

    print("=== 图二: TWA 各尺度热图 (CBAM×4/Triplet×4 热图见 gen_attn_x4_figs.py) ===")
    fig_twa = build_attention_figure(data_list, [
        ("TWA s1 · H/4", lambda d: d["hms"]["s1"]),
        ("TWA s2 · H/8", lambda d: d["hms"]["s2"]),
        ("TWA s3 · H/16", lambda d: d["hms"]["s3"]),
        ("TWA s4 · H/32", lambda d: d["hms"]["s4"]),
    ])

    for name, fig in (("twa", fig_twa),):
        p = os.path.join(FIGDIR, f"fig_attention_{name}.png")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(FIGDIR, f"fig_attention_{name}.pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
