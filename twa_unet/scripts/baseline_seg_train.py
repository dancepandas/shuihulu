"""U-Net / DeepLabV3+ 单阶段语义分割基线训练

配置与论文 LWH-Seg 对齐: 512x512, AdamW(WD=0.01), 余弦退火, 输入分辨率 512,
增强与 baseline_seg_data.get_transform("train") 一致, 损失 = 类别加权CE + Dice (α=1, β=0.5).

用法:
  python scripts/baseline_seg_train.py --arch unet       --backbone resnet34 --epochs 80
  python scripts/baseline_seg_train.py --arch deeplabv3p --backbone resnet34 --epochs 80
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from baseline_seg_data import (
    SegDataset, get_transform, class_weights_from_masks,
    DATASET_DIR, IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, ROOT,
)
from baseline_seg_eval import evaluate_all, report, measure_size

import segmentation_models_pytorch as smp


def dice_loss(pred_logits, mask, num_classes, smooth=1.0):
    """论文公式(3): L_Dice = (1/C) Σ_c (1 - (2 Σ z*·m + ε)/(Σ z* + Σ m + ε))"""
    pred = F.softmax(pred_logits, dim=1)          # B,C,H,W
    m = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()  # B,C,H,W
    C = num_classes
    loss = 0.0
    for c in range(C):
        z = pred[:, c].contiguous().view(-1)
        mm = m[:, c].contiguous().view(-1)
        num = 2.0 * (z * mm).sum() + smooth
        den = z.sum() + mm.sum() + smooth
        loss += 1.0 - num / den
    return loss / C


def build_model(arch, backbone, num_classes):
    if arch == "unet":
        model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet",
                         in_channels=3, classes=num_classes)
    elif arch == "deeplabv3p":
        model = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights="imagenet",
                                  in_channels=3, classes=num_classes)
    else:
        raise ValueError(arch)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["unet", "deeplabv3p"], default="unet")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2, help="梯度累积步数 (等价 batch=batch*accum)")
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--resume", default=None, help="恢复的 state_dict 路径")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.arch}_{args.backbone}"
    out_dir = os.path.join(ROOT, "runs", "baselines", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{tag}] 训练配置: epochs={args.epochs} batch={args.batch} "
          f"accum={args.accum} lr={args.lr} size={args.size} device={device}")

    w_c, p_c = class_weights_from_masks("train")
    print("类别权重:", np.round(w_c, 3), " priors:", np.round(p_c, 5))
    w_t = torch.tensor(w_c, dtype=torch.float32, device=device)

    train_ds = SegDataset("train", get_transform("train"))
    val_ds = SegDataset("val", get_transform("val"))
    # drop_last=True: 209张/bs2=104批+1张，末批为1会使 DeepLabV3+ 的 ASPP 1x1池化分支 BN 报错
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = build_model(args.arch, args.backbone, NUM_CLASSES).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(weight=w_t, ignore_index=255)

    best = {"mIoU": 0.0}
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run_loss = 0.0; n_b = 0
        opt.zero_grad()
        for bi, (img, mask) in enumerate(tqdm(train_ld, desc=f"E{ep}/{args.epochs}", leave=False)):
            img = img.to(device); mask = mask.to(device)
            logits = model(img)
            loss = ce(logits, mask) + 0.5 * dice_loss(logits, mask, NUM_CLASSES)
            loss = loss / args.accum
            loss.backward()
            if (bi + 1) % args.accum == 0:
                opt.step(); opt.zero_grad()
            run_loss += loss.item() * args.accum; n_b += 1
        if (bi + 1) % args.accum != 0:
            opt.step(); opt.zero_grad()
        sched.step()
        r = evaluate_all(model, val_ld, device)
        report(r, f"{tag} E{ep}")
        if r["mIoU"] > best["mIoU"]:
            best = r; best["epoch"] = ep
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        print(f"  E{ep} loss={run_loss/max(n_b,1):.4f} 耗时 {time.time()-t0:.1f}s  "
              f"best_mIoU={best['mIoU']:.4f}@{best.get('epoch', '-')}")

    print(f"\n[{tag}] 最终结果 (best mIoU={best['mIoU']:.4f} @E{best.get('epoch')}):")
    report(best, f"{tag} FINAL")
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"arch={args.arch} backbone={args.backbone}\n")
        f.write(f"epochs={args.epochs} batch={args.batch} accum={args.accum} lr={args.lr}\n")
        f.write(f"mIoU={best['mIoU']:.4f} mAcc={best['mAcc']:.4f} WH_IoU={best['wh_iou']:.4f}\n")
        f.write(f"params_M={best['params_m']:.3f} latency_ms={best['latency_ms']:.2f} "
                f"size_MB={measure_size(model):.2f}\n")
        f.write(f"per_class_iou={best['per_class_iou']}\n")
        f.write(f"per_class_recall={best['per_class_recall']}\n")
        f.write(f"per_class_prec={best['per_class_prec']}\n")
    print("metrics 写入:", os.path.join(out_dir, "metrics.txt"))


if __name__ == "__main__":
    main()
