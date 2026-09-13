# -*- coding: utf-8 -*-
"""批量复测：逐类 IoU (B4/B1/A5)、逐图水葫芦 IoU (A5/A1/A2/B1)、A5 混淆矩阵、配对 Wilcoxon。
内部全部使用本地 checkpoint 原始推理值；论文口径由调用方决定。"""
import os, sys, json
ROOT = r"D:\chengs\9.project\shuihulu"
sys.path.insert(0, os.path.join(ROOT, "twa_unet", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "twa_unet"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import baseline_seg_data as _bsd
_bsd.DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v11")
_bsd.IMAGE_SIZE = 640
_bsd.NUM_CLASSES = 5
_bsd.CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
                    3: "shore_vegetation", 4: "other_aquatic_vegetation"}
from baseline_seg_data import SegDataset, get_transform
from model.twa_u_net import TWAUNet
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(ROOT, "twa_unet", "scripts"))
from baseline_seg_train_v11 import build_model  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_ds = SegDataset("val", transform=get_transform("val"))
val_ld = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
N = len(val_ds)
print("val =", N, flush=True)


@torch.no_grad()
def run_model(model):
    """返回: per_class_iou[5], per_image_wh[135], cm[5,5]"""
    model.eval()
    inter = np.zeros(5); union = np.zeros(5)
    cm = np.zeros((5, 5), np.int64)
    per_img_wh = np.zeros(N)
    for idx, (img, mask) in enumerate(val_ld):
        img = img.to(dev)
        logits = model(img)
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()[0]; m = mask.numpy()[0]
        for c in range(5):
            p = (pred == c); t = (m == c)
            inter[c] += (p & t).sum(); union[c] += (p | t).sum()
        for g in range(5):
            for p in range(5):
                cm[g, p] += ((m == g) & (pred == p)).sum()
        wp = (pred == 1); wt = (m == 1)
        u = (wp | wt).sum()
        per_img_wh[idx] = (wp & wt).sum() / u if u > 0 else np.nan
    ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return ious, per_img_wh, cm


def load_twa(ckpt, direction, window):
    m = TWAUNet(num_classes=5, pretrained=False, window_size=window, num_heads=8,
                twa_direction=direction, use_twa=True, attn_mode="twa")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev)


def load_base(arch, backbone, attn, ckpt):
    m = build_model(arch, backbone, 5, attn)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev)


TWA = os.path.join(ROOT, "runs", "twa_u_net")
BAS = os.path.join(ROOT, "runs", "baselines")

res = {}
specs = {
    "A5": load_twa(os.path.join(TWA, "ablation_A5_lower_win4_v11", "best.pt"), "lower", 4),
    "A1": load_twa(os.path.join(TWA, "ablation_A1_lower_v11", "best.pt"), "lower", 8),
    "A2": load_twa(os.path.join(TWA, "ablation_A2_upper_v11", "best.pt"), "upper", 8),
    "B1": load_base("unet", "resnet34", "none", os.path.join(BAS, "unet_resnet34_v11", "best.pt")),
    "B4": load_base("deeplabv3p", "resnet18", "none", os.path.join(BAS, "dlv3p_resnet18_v11", "best.pt")),
}
out = {}
for name, model in specs.items():
    ious, per_img, cm = run_model(model)
    out[name] = {"per_class_iou": [round(float(x), 4) for x in ious],
                 "miou": round(float(np.nanmean(ious)), 4),
                 "per_img_wh": [None if np.isnan(x) else round(float(x), 4) for x in per_img]}
    if name == "A5":
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        out["A5_cm_row_pct"] = [[round(float(x), 2) for x in row] for row in cmn]
        r1 = cm[1].sum(); c1 = cm[:, 1].sum()
        out["wh_recall_pct"] = round(float(cm[1, 1]) / r1 * 100, 2)
        out["wh_prec_pct"] = round(float(cm[1, 1]) / c1 * 100, 2)
        out["wh_to_other_pct_of_row"] = round(float(cm[1, 4]) / r1 * 100, 2)
        out["other_to_wh_pct_of_row"] = round(float(cm[4, 1]) / cm[4].sum() * 100, 2)
        out["wh_err_share_to_other_pct"] = round(float(cm[1, 4]) / max(r1 - cm[1, 1], 1) * 100, 2)
    print(name, "done", out[name]["miou"], flush=True)

from scipy.stats import wilcoxon
def paired(a, b):
    x = np.array(out[a]["per_img_wh"], float); y = np.array(out[b]["per_img_wh"], float)
    ok = ~(np.isnan(x) | np.isnan(y))
    stat, p = wilcoxon(x[ok], y[ok])
    win = float((x[ok] > y[ok]).mean())
    return {"n": int(ok.sum()), "p": float(p), "stat": float(stat),
            "mean_diff": float(np.mean(x[ok] - y[ok])), "win_rate": win}

out["wilcoxon_A5_vs_B1"] = paired("A5", "B1")
out["wilcoxon_A1_vs_A2"] = paired("A1", "A2")

with open(os.path.join(ROOT, "scripts", "_perclass_eval_out.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED")
