# -*- coding: utf-8 -*-
"""复测 A5 (TWAU-Net) 在 hyacinth_ls_v11 验证集上的指标，核对 metrics.txt 与论文表3。"""
import os, sys
ROOT = r"D:\chengs\9.project\shuihulu"
SCRIPTS_DIR = os.path.join(ROOT, "twa_unet", "scripts")
TWA_DIR = os.path.join(ROOT, "runs", "twa_u_net")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(ROOT, "twa_unet"))  # model/ 包所在

import numpy as np
import torch
from torch.utils.data import DataLoader
import baseline_seg_data as _bsd
_bsd.DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v11")
_bsd.IMAGE_SIZE = 640
_bsd.NUM_CLASSES = 5
_bsd.CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
                    3: "shore_vegetation", 4: "other_aquatic_vegetation"}
import baseline_seg_eval as _bse
_bse.NUM_CLASSES = _bsd.NUM_CLASSES
_bse.CLASS_NAMES = _bsd.CLASS_NAMES
from baseline_seg_data import SegDataset, get_transform
from baseline_seg_eval import evaluate_all
from model.twa_u_net import TWAUNet

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TWA_DIR, "ablation_A5_lower_win4_v11", "best.pt")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_ds = SegDataset("val", transform=get_transform("val"))
val_ld = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
print(f"val={len(val_ds)}")

model = TWAUNet(num_classes=5, pretrained=False, window_size=4, num_heads=8,
                twa_direction="lower", use_twa=True, attn_mode="twa").to(dev)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd)
model.eval()

r = evaluate_all(model, val_ld, dev)
names = [_bsd.CLASS_NAMES[i] for i in range(5)]
print("== 复测结果 ==")
print(f"mIoU={r['mIoU']:.4f} mAcc={r['mAcc']:.4f} WH_IoU(idx1 水葫芦)={r['per_class_iou'][1]:.4f} other_aq(idx4)={r['per_class_iou'][4]:.4f}")
print("per_class_iou:", dict(zip(names, [round(x, 4) for x in r["per_class_iou"]])))
print("params_M=%.3f" % r["params_m"])
