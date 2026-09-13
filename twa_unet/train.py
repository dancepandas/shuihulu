"""TWAU-Net 训练脚本
与 baseline_seg_train.py 训练流程对齐 (CE+Dice, AdamW, 余弦退火, 512x512)
但模型用自定义的 TWAU-Net 而非 smp.Unet.

用法:
  cd twa_unet
  PYTHONPATH=scripts python train.py --direction both --epochs 80
  (训练产物输出到 runs/twa_u_net/<tag>)
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

# 把 twa_unet/scripts/ 加入 path 以便导入 baseline_seg_data / baseline_seg_eval
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(__file__))  # 让 train.py 能找到 model.twa_u_net

from baseline_seg_data import ROOT as _ROOT
import baseline_seg_data as _bsd
# v11 数据集 (5 类, 546 train / 135 val, 640×640)
# 强制覆盖 baseline_seg_data 的 hyacinth_seg 默认值
_bsd.DATASET_DIR = os.path.join(_ROOT, "datasets", "hyacinth_ls_v11")
_bsd.IMAGE_SIZE = 640
_bsd.NUM_CLASSES = 5
_bsd.CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
                    3: "shore_vegetation", 4: "other_aquatic_vegetation"}
DATASET_DIR = _bsd.DATASET_DIR
IMAGE_SIZE = _bsd.IMAGE_SIZE
NUM_CLASSES = _bsd.NUM_CLASSES
CLASS_NAMES = _bsd.CLASS_NAMES
print(f"数据集: {DATASET_DIR}")
print(f"  IMAGE_SIZE={IMAGE_SIZE}  NUM_CLASSES={NUM_CLASSES}  CLASS_NAMES={CLASS_NAMES}")

from baseline_seg_data import SegDataset, get_transform, class_weights_from_masks
import baseline_seg_data as _bsd2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import baseline_seg_eval as _bse
# 同步覆盖 eval 模块的 hard-code 引用
_bse.NUM_CLASSES = NUM_CLASSES
_bse.CLASS_NAMES = CLASS_NAMES
from baseline_seg_eval import evaluate_all, report, measure_size

from model.twa_u_net import TWAUNet


def dice_loss(pred_logits, mask, num_classes, smooth=1.0):
    """论文公式: L_Dice = (1/C) Σ_c (1 - (2 Σ z·m + ε)/(Σ z + Σ m + ε))"""
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


def boundary_bce_loss(pred_logits, mask, num_classes):
    """v15: 边界 BCE 损失. 提取 mask 边界像素, 在边界上算 BCE.
    pred_logits: (B, C, H, W)
    mask: (B, H, W)
    """
    # 边界提取: 用 3x3 maxpool(mask) - mask, 取差集即为边界
    m = mask.unsqueeze(1).float()  # (B, 1, H, W)
    dil = F.max_pool2d(m, 3, stride=1, padding=1)
    boundary = (dil - m).gt(0.5).float()  # (B, 1, H, W) 0/1
    n_boundary = boundary.sum().clamp(min=1.0)
    # 把 pred_logits 插值到 mask 尺寸 (一般相同)
    if pred_logits.shape[-2:] != mask.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    # 在 boundary 像素上算 BCE (按通道求和)
    pred_prob = F.softmax(pred_logits, dim=1)  # (B, C, H, W)
    # one-hot mask
    m_one = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    # 仅在 boundary 像素上算
    boundary_4d = boundary.expand_as(pred_prob)  # (B, C, H, W)
    bce = -(m_one * torch.log(pred_prob.clamp(min=1e-7))) * boundary_4d
    return bce.sum() / n_boundary / num_classes


def focal_loss_with_weight(logits, target, weight, gamma=2.0, ignore_index=255):
    """v_solve-C: Focal Loss with class weight. 自动聚焦难例 (γ=2)."""
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()
    # nll_loss reduction='none' → 每个像素的 CE
    ce = F.nll_loss(logits, target, weight=weight, ignore_index=ignore_index, reduction="none")
    pt = prob.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, H, W)
    pt = pt * (target != ignore_index)  # 屏蔽 ignore_index
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()


def class_balanced_weights(p_c, alpha=0.999):
    """Class-balanced weights (Cui et al. 2019). 有效样本数加权, 比 inverse prior 更平滑."""
    n_classes = len(p_c)
    eff_num = (1.0 - alpha ** (1.0 / p_c.clip(min=1e-6))) / (1.0 - alpha)
    weights = (1.0 / eff_num)
    weights = weights / weights.sum() * n_classes  # 归一化
    return weights.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["both", "lower", "upper"], default="both")
    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--attn_mode", choices=["twa", "hybrid", "hybrid_v2", "none"],
                    default="twa", help="跳跃连接注意力模式: twa=全TWA, hybrid=TWA+CBAM+SE, none=vanilla")
    ap.add_argument("--use_ema", action="store_true", help="v13+: EMA 权重评估")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--use_shift", action="store_true", help="v14+: shifted window (s1/s2)")
    ap.add_argument("--use_deepsup", action="store_true", help="v14+: deep supervision (aux heads)")
    ap.add_argument("--use_boundary", action="store_true", help="v15+: boundary loss")
    ap.add_argument("--use_scale", action="store_true", help="v15+: RandomScale (0.75-1.5) 增强")
    ap.add_argument("--pre_norm", action="store_true", help="v_solve-A: LLM 风格 Pre-Norm")
    ap.add_argument("--use_swiglu", action="store_true", help="v_solve-A: SwiGLU 替代 GELU MLP")
    ap.add_argument("--gated_skip", action="store_true", help="v_solve-B: decoder gated skip")
    ap.add_argument("--use_focal", action="store_true", help="v_solve-C: Focal Loss (γ=2)")
    ap.add_argument("--use_balanced_sampler", action="store_true", help="v_solve-C: 类平衡采样器")
    ap.add_argument("--deepsup_weight", type=float, default=0.3)
    ap.add_argument("--boundary_weight", type=float, default=0.1)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--no_twa", action="store_true",
                    help="消融: 不使用 TWA 模块 (退化为 vanilla U-Net)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"twa_u_net_{args.direction}_w{args.window_size}_h{args.num_heads}_v11"
    out_dir = os.path.join(_ROOT, "runs", "twa_u_net", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{tag}] 训练配置: epochs={args.epochs} batch={args.batch} "
          f"accum={args.accum} lr={args.lr} size={args.size} "
          f"direction={args.direction} window={args.window_size} heads={args.num_heads} "
          f"pretrained={not args.no_pretrained} device={device}")

    w_c, p_c = class_weights_from_masks("train")
    print("类别权重:", np.round(w_c, 3), " priors:", np.round(p_c, 5))
    w_t = torch.tensor(w_c, dtype=torch.float32, device=device)

    # 初始化 transforms (避免 UnboundLocalError)
    train_transform = get_transform("train")
    val_transform = get_transform("val")
    # v15: 用更强 RandomScale 替换 train transform
    if args.use_scale:
        def get_transform_v15(split):
            if split == "train":
                return A.Compose([
                    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
                    A.Affine(translate_percent=0.08, scale=(0.75, 1.5), rotate=(-20, 20), p=0.7),  # 更强
                    A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
                    A.HueSaturationValue(15, 25, 15, p=0.4),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.Normalize(_bsd2.MEAN, _bsd2.STD), ToTensorV2(),
                ])
            return get_transform("val")
        train_transform = get_transform_v15("train")
        val_transform = get_transform("val")

    train_ds = SegDataset("train", train_transform)
    val_ds = SegDataset("val", val_transform)
    # v_solve-C: 类平衡采样器 — 每张图的采样权重 ∝ 1/(各类像素数+ε), 让水葫芦+其他水生植被样本被更频繁采样
    if args.use_balanced_sampler:
        from torch.utils.data import WeightedRandomSampler
        import cv2
        sample_weights = []
        for idx in range(len(train_ds)):
            img_id = train_ds.images[idx]
            stem = os.path.splitext(img_id)[0]
            mask_path = os.path.join(_bsd2.DATASET_DIR, "masks", "train", stem + ".png")
            if os.path.exists(mask_path):
                m = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
                wh_ratio = ((m == 1).sum() + (m == 4).sum()) / max(m.size, 1)
            else:
                wh_ratio = 0.05
            sample_weights.append(1.0 + 5.0 * wh_ratio)
        sample_weights = np.array(sample_weights, dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_ld = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
        print(f"  [balanced sampler] {len(sample_weights)} samples, weight range [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    else:
        train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = TWAUNet(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained,
                    window_size=args.window_size, num_heads=args.num_heads,
                    twa_direction=args.direction, use_twa=not args.no_twa,
                    attn_mode=args.attn_mode,
                    use_shift=args.use_shift, deepsup=args.use_deepsup,
                    pre_norm=args.pre_norm, use_swiglu=args.use_swiglu,
                    gated_skip=args.gated_skip).to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n_params:.2f} M")

    # EMA (Exponential Moving Average) 权重 — v13
    ema_model = None
    if args.use_ema:
        from torch.optim.swa_utils import AveragedModel, SWALR
        # 用 AveragedModel 实现 EMA
        class EMAModel(AveragedModel):
            def __init__(self, model, decay=0.999):
                super().__init__(model, multi_avg_fn=self._ema_avg(decay))
            @staticmethod
            def _ema_avg(decay):
                def avg_fn(avg, current, n):
                    return decay * avg + (1 - decay) * current
                return avg_fn
        ema_model = EMAModel(model, decay=args.ema_decay).to(device)
        print(f"  [EMA] enabled, decay={args.ema_decay}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(weight=w_t, ignore_index=255)
    print(f"[{tag}] flags: use_ema={args.use_ema} use_shift={args.use_shift} "
          f"use_deepsup={args.use_deepsup} use_boundary={args.use_boundary} "
          f"pre_norm={args.pre_norm} swiglu={args.use_swiglu} gated_skip={args.gated_skip} "
          f"focal={args.use_focal} balanced_sampler={args.use_balanced_sampler} "
          f"deepsup_w={args.deepsup_weight} boundary_w={args.boundary_weight}")

    best = {"mIoU": 0.0}
    best_is_ema = False
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run_loss = 0.0; n_b = 0
        opt.zero_grad()
        for bi, (img, mask) in enumerate(tqdm(train_ld, desc=f"E{ep}/{args.epochs}", leave=False)):
            img = img.to(device); mask = mask.to(device)
            out = model(img)
            # v_solve-C: Focal Loss 替代 CE
            if args.use_focal:
                loss_ce = focal_loss_with_weight(out if not isinstance(out, tuple) else out[0],
                                                  mask, w_t, gamma=args.focal_gamma)
            # v14: deep supervision → (main, aux_d3, aux_d2)
            if isinstance(out, tuple):
                logits, aux_d3, aux_d2 = out
                if args.use_focal:
                    main_loss = loss_ce + 0.5 * dice_loss(logits, mask, NUM_CLASSES)
                    aux3_loss = focal_loss_with_weight(aux_d3, mask, w_t, gamma=args.focal_gamma) \
                                + 0.5 * dice_loss(aux_d3, mask, NUM_CLASSES)
                    aux2_loss = focal_loss_with_weight(aux_d2, mask, w_t, gamma=args.focal_gamma) \
                                + 0.5 * dice_loss(aux_d2, mask, NUM_CLASSES)
                else:
                    main_loss = ce(logits, mask) + 0.5 * dice_loss(logits, mask, NUM_CLASSES)
                    aux3_loss = ce(aux_d3, mask) + 0.5 * dice_loss(aux_d3, mask, NUM_CLASSES)
                    aux2_loss = ce(aux_d2, mask) + 0.5 * dice_loss(aux_d2, mask, NUM_CLASSES)
                loss = main_loss + args.deepsup_weight * (aux3_loss + aux2_loss)
            else:
                logits = out
                if args.use_focal:
                    loss = loss_ce + 0.5 * dice_loss(logits, mask, NUM_CLASSES)
                else:
                    loss = ce(logits, mask) + 0.5 * dice_loss(logits, mask, NUM_CLASSES)
            # v15: boundary loss
            if args.use_boundary:
                loss = loss + args.boundary_weight * boundary_bce_loss(logits, mask, NUM_CLASSES)
            loss = loss / args.accum
            loss.backward()
            if (bi + 1) % args.accum == 0:
                opt.step(); opt.zero_grad()
                # EMA 更新: 每个 opt.step() 后同步权重
                if ema_model is not None:
                    ema_model.update_parameters(model)
            run_loss += loss.item() * args.accum; n_b += 1
        if (bi + 1) % args.accum != 0:
            opt.step(); opt.zero_grad()
            if ema_model is not None:
                ema_model.update_parameters(model)
        sched.step()
        # 评估: 优先用 EMA (v13+)
        eval_target = ema_model if ema_model is not None else model
        r = evaluate_all(eval_target, val_ld, device)
        report(r, f"{tag} E{ep}{' (EMA)' if ema_model is not None else ''}")
        if r["mIoU"] > best["mIoU"]:
            best = r; best["epoch"] = ep
            best_is_ema = ema_model is not None
            torch.save(eval_target.state_dict(), os.path.join(out_dir, "best.pt"))
        print(f"  E{ep} loss={run_loss/max(n_b,1):.4f} 耗时 {time.time()-t0:.1f}s  "
              f"best_mIoU={best['mIoU']:.4f}@{best.get('epoch', '-')}{' [EMA]' if best_is_ema else ''}")

    print(f"\n[{tag}] 最终结果 (best mIoU={best['mIoU']:.4f} @E{best.get('epoch')}):")
    report(best, f"{tag} FINAL")
    torch.save((ema_model if ema_model is not None else model).state_dict(),
               os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"arch=twa_u_net attn_mode={args.attn_mode} direction={args.direction} window={args.window_size} heads={args.num_heads}\n")
        f.write(f"use_ema={args.use_ema} ema_decay={args.ema_decay}\n")
        f.write(f"use_shift={args.use_shift} use_deepsup={args.use_deepsup} "
                f"use_boundary={args.use_boundary} deepsup_w={args.deepsup_weight} "
                f"boundary_w={args.boundary_weight}\n")
        f.write(f"pretrained={not args.no_pretrained}\n")
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