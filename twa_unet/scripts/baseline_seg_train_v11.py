"""U-Net / DeepLabV3+ / U-Net+CBAM / U-Net+Triplet 训练 (v11 数据集)

配置: 640×640 (v11), AdamW(WD=0.01), 余弦退火, 类别加权CE + Dice (α=1, β=0.5)
通过 monkey-patch baseline_seg_data 的常量切换到 v11 数据集 (5 类, 546 train / 135 val)

用法:
  python scripts/baseline_seg_train_v11.py --arch unet --backbone resnet34 --epochs 80
  python scripts/baseline_seg_train_v11.py --arch unet --backbone resnet18 --attn cbam --epochs 80
  python scripts/baseline_seg_train_v11.py --arch unet --backbone resnet18 --attn triplet --epochs 80
  python scripts/baseline_seg_train_v11.py --arch deeplabv3p --backbone resnet18 --epochs 80
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# === v11 数据集 monkey-patch ===
import baseline_seg_data as _bsd
from baseline_seg_data import ROOT as _ROOT
_bsd.DATASET_DIR = os.path.join(_ROOT, "datasets", "hyacinth_ls_v11")
_bsd.IMAGE_SIZE = 640
_bsd.NUM_CLASSES = 5
_bsd.CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
                    3: "shore_vegetation", 4: "other_aquatic_vegetation"}

from baseline_seg_data import (
    SegDataset, get_transform, class_weights_from_masks,
    DATASET_DIR, IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, ROOT,
)
import baseline_seg_eval as _bse
_bse.NUM_CLASSES = NUM_CLASSES
_bse.CLASS_NAMES = CLASS_NAMES
from baseline_seg_eval import evaluate_all, report, measure_size

import segmentation_models_pytorch as smp


# ============= CBAM 实现 (Channel + Spatial Attention) =============
class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
        )

    def forward(self, x):
        # Channel attention: avg & max pool
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.channel(avg_pool) + self.channel(max_pool))
        x = x * ch
        # Spatial attention: avg & max along channel
        sp = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))
        x = x * sp
        return x


# ============= Triplet Attention 实现 (Misra et al., 2021 简化版) =============
class TripletAttention(nn.Module):
    """Triplet Attention 简化实现: 3 路 (channel / H / W) sigmoid attention 串联."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # channel 分支: GAP → MLP → sigmoid
        self.c_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        # H 分支: 在 W 维 pool 后用 1x1 conv 学每行权重
        self.h_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )
        # W 分支: 在 H 维 pool 后用 1x1 conv 学每列权重
        self.w_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # channel attention
        z = F.adaptive_avg_pool2d(x, 1).flatten(1)
        c = torch.sigmoid(self.c_mlp(z)).unsqueeze(-1).unsqueeze(-1)
        # H attention
        h = torch.sigmoid(self.h_conv(x.mean(3, keepdim=True)))  # (B, 1, H, 1)
        # W attention
        w = torch.sigmoid(self.w_conv(x.mean(2, keepdim=True)))  # (B, 1, 1, W)
        return x * c * h * w


class AttentionWrapper(nn.Module):
    """在 smp.Unet 的 decoder 输出上接 attention (CBAM / Triplet)."""

    def __init__(self, base_model, attn_type):
        super().__init__()
        self.base = base_model
        self.attn_type = attn_type
        out_ch = base_model.segmentation_head[0].in_channels  # decoder 输出通道
        if attn_type == "cbam":
            self.attn = CBAM(out_ch)
        elif attn_type == "triplet":
            self.attn = TripletAttention(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        features = self.base.encoder(x)
        # smp UnetDecoder.forward(*features) 用变长参数,需要解包
        decoder_output = self.base.decoder(*features)
        decoder_output = self.attn(decoder_output)
        logits = self.base.segmentation_head(decoder_output)
        return logits


class MultiPlaceAttention(nn.Module):
    """在 encoder 的 4 个跳跃特征 (H/4,H/8,H/16,H/32) 上各嵌一个注意力模块.

    与 TWAU-Net 的 s1-s4 嵌入位置逐一对齐, 用于公平对照 (C-7: CBAM×4).
    """

    def __init__(self, base_model, attn_type):
        super().__init__()
        self.base = base_model
        chs = base_model.encoder.out_channels   # (3, 64, 64, 128, 256, 512)
        make = CBAM if attn_type == "cbam" else TripletAttention
        self.attns = nn.ModuleList([make(chs[i]) for i in range(1, 5)])

    def forward(self, x):
        features = list(self.base.encoder(x))
        for i, attn in enumerate(self.attns, start=1):
            features[i] = attn(features[i])
        decoder_output = self.base.decoder(*features)
        return self.base.segmentation_head(decoder_output)


def build_model(arch, backbone, num_classes, attn):
    if arch == "unet":
        base = smp.Unet(encoder_name=backbone, encoder_weights="imagenet",
                        in_channels=3, classes=num_classes)
    elif arch == "deeplabv3p":
        base = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights="imagenet",
                                 in_channels=3, classes=num_classes)
    else:
        raise ValueError(arch)
    if attn in ("cbam", "triplet"):
        model = AttentionWrapper(base, attn)
    elif attn in ("cbam_x4", "triplet_x4"):
        model = MultiPlaceAttention(base, attn.replace("_x4", ""))
    else:
        model = base
    return model


def dice_loss(pred_logits, mask, num_classes, smooth=1.0):
    pred = F.softmax(pred_logits, dim=1)
    m = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
    C = num_classes
    loss = 0.0
    for c in range(C):
        z = pred[:, c].contiguous().view(-1)
        mm = m[:, c].contiguous().view(-1)
        num = 2.0 * (z * mm).sum() + smooth
        den = z.sum() + mm.sum() + smooth
        loss += 1.0 - num / den
    return loss / C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["unet", "deeplabv3p"], default="unet")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--attn", choices=["none", "cbam", "triplet", "cbam_x4", "triplet_x4"],
                    default="none")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{args.arch}_{args.backbone}_{args.attn}_v11"
    out_dir = os.path.join(ROOT, "runs", "baselines", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{tag}] 训练配置: epochs={args.epochs} batch={args.batch} "
          f"accum={args.accum} lr={args.lr} size={args.size} "
          f"arch={args.arch} backbone={args.backbone} attn={args.attn} device={device}")

    w_c, p_c = class_weights_from_masks("train")
    print("类别权重:", np.round(w_c, 3), " priors:", np.round(p_c, 5))
    w_t = torch.tensor(w_c, dtype=torch.float32, device=device)

    train_ds = SegDataset("train", get_transform("train"))
    val_ds = SegDataset("val", get_transform("val"))
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = build_model(args.arch, args.backbone, NUM_CLASSES, args.attn).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n_params:.2f} M")

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
        f.write(f"arch={args.arch} backbone={args.backbone} attn={args.attn}\n")
        f.write(f"epochs={args.epochs} batch={args.batch} accum={args.accum} lr={args.lr}\n")
        f.write(f"mIoU={best['mIoU']:.4f} mAcc={best['mAcc']:.4f} WH_IoU={best['wh_iou']:.4f}\n")
        f.write(f"params_M={n_params:.3f} latency_ms={best['latency_ms']:.2f} "
                f"size_MB={measure_size(model):.2f}\n")
        f.write(f"per_class_iou={best['per_class_iou']}\n")
        f.write(f"per_class_recall={best['per_class_recall']}\n")
        f.write(f"per_class_prec={best['per_class_prec']}\n")
    print("metrics 写入:", os.path.join(out_dir, "metrics.txt"))


if __name__ == "__main__":
    main()