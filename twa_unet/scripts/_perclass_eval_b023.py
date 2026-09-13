# -*- coding: utf-8 -*-
"""补测 B0/B2/B3 的逐类 IoU，追加到 _perclass_eval_out.json。"""
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
from baseline_seg_train_v11 import build_model

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_ld = DataLoader(SegDataset("val", transform=get_transform("val")), 1,
                    shuffle=False, num_workers=0, pin_memory=True)
print("val =", len(val_ld.dataset), flush=True)


@torch.no_grad()
def per_class_iou(model):
    model.eval()
    inter = np.zeros(5); union = np.zeros(5)
    for img, mask in val_ld:
        logits = model(img.to(dev))
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()[0]; m = mask.numpy()[0]
        for c in range(5):
            p = (pred == c); t = (m == c)
            inter[c] += (p & t).sum(); union[c] += (p | t).sum()
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


BAS = os.path.join(ROOT, "runs", "baselines")
specs = {
    "B0": ("unet", "resnet18", "none", os.path.join(BAS, "unet_resnet18_v11", "best.pt")),
    "B2": ("unet", "resnet34", "cbam_x4", os.path.join(BAS, "unet_resnet34_cbam_x4_v11", "best.pt")),
    "B3": ("unet", "resnet34", "triplet_x4", os.path.join(BAS, "unet_resnet34_triplet_x4_v11", "best.pt")),
}
out_path = os.path.join(ROOT, "scripts", "_perclass_eval_out.json")
out = json.load(open(out_path, encoding="utf-8"))
for name, (arch, bb, attn, ckpt) in specs.items():
    m = build_model(arch, bb, 5, attn)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = m.to(dev)
    ious = per_class_iou(m)
    out[name] = {"per_class_iou": [round(float(x), 4) for x in ious],
                 "miou": round(float(np.nanmean(ious)), 4)}
    print(name, out[name]["miou"], out[name]["per_class_iou"], flush=True)
    del m; torch.cuda.empty_cache()

json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("SAVED")
