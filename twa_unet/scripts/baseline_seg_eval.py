"""统一评估 — 6 类水葫芦语义分割基线

指标口径 (与论文一致):
  mIoU     = 各类别 IoU 的算术平均
  mAcc     = 各类别像素精度 (recall = TP/(TP+FN)) 的算术平均
  WH IoU   = 类别 4 (水葫芦) 的 IoU
  Params   = 模型参数量 (M)
  Size     = 模型权重文件体积 (MB, 实测 torch.save 结果)
  Latency  = 单图推理时间 (ms, batch=1, FP32, 含 forward+upsample)
"""
import os
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from baseline_seg_data import SegDataset, get_transform, NUM_CLASSES, CLASS_NAMES


@torch.no_grad()
def evaluate_all(model, val_loader, device, num_classes=NUM_CLASSES):
    model.eval()
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    tp = np.zeros(num_classes); fp = np.zeros(num_classes); fn = np.zeros(num_classes)
    lat = []
    for img, mask in val_loader:
        img = img.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(img)
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        pred = logits.argmax(1).cpu().numpy(); m = mask.numpy()
        for c in range(num_classes):
            p = (pred == c); t = (m == c)
            inter[c] += (p & t).sum(); union[c] += (p | t).sum()
            tp[c] += (p & t).sum(); fp[c] += (p & ~t).sum(); fn[c] += (~p & t).sum()
    ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    recall = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1), np.nan)
    prec = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), np.nan)
    return {
        "mIoU": float(np.nanmean(ious)),
        "mAcc": float(np.nanmean(recall)),
        "wh_iou": float(ious[4]),
        "per_class_iou": ious.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_prec": prec.tolist(),
        "latency_ms": float(np.mean(lat)) if lat else None,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }


def measure_size(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def report(result, tag=""):
    names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    print(f"[{tag}] mIoU={result['mIoU']:.4f}  mAcc={result['mAcc']:.4f}  "
          f"WH IoU={result['wh_iou']:.4f}  Params={result['params_m']:.2f}M  "
          f"Latency={result['latency_ms']:.1f}ms")
    print("  IoU    :", " ".join(f"{names[i]}={result['per_class_iou'][i]:.3f}" for i in range(NUM_CLASSES)))
    print("  Recall :", " ".join(f"{names[i]}={result['per_class_recall'][i]:.3f}" for i in range(NUM_CLASSES)))
    print("  Prec   :", " ".join(f"{names[i]}={result['per_class_prec'][i]:.3f}" for i in range(NUM_CLASSES)))


def make_val_loader(batch=1):
    ds = SegDataset("val", get_transform("val"))
    return DataLoader(ds, batch, shuffle=False, num_workers=0, pin_memory=True)
