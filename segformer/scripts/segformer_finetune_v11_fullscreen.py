"""SegFormer-B2 v11 续跑微调 — 补充"几乎全屏水葫芦"难例 (7 张截图)。
从 v11b 权重继续训练, 数据 = v11 (546 train + 135 val, 含新增 7 张全屏水葫芦),
低学习率微调, 全量解冻 (不冻结编码器)。
"""
import os, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import evaluate

ROOT = r"D:\chengs\9.project\shuihulu"
DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v11")   # 546 train + 135 val
OUTPUT_DIR = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11_fullscreen")
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11b")  # 续跑起点

IMAGE_SIZE = 640
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 30            # 续跑微调, 不需要太多 epoch
PHASE1_EPOCHS = 0          # 全量微调 (不冻结编码器)
LR, WD = 2e-5, 0.01        # 低学习率, 避免破坏已有知识
NUM_CLASSES = 5
LOSS_ALPHA, LOSS_BETA = 1.0, 0.5
DICE_EPS = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
               3: "shore_vegetation", 4: "other_aquatic_vegetation"}


class SegDataset(Dataset):
    def __init__(self, split, transform=None):
        self.img_dir = os.path.join(DATASET_DIR, "images", split)
        self.mask_dir = os.path.join(DATASET_DIR, "masks", split)
        self.transform = transform
        self.images = sorted(f for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg", ".png", ".jpeg")))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]; stem = os.path.splitext(fname)[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            a = self.transform(image=img, mask=mask)
            return {"pixel_values": a["image"], "labels": a["mask"].long()}
        return {"pixel_values": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                "labels": torch.from_numpy(mask).long()}


def get_transform(split):
    if split == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.3),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def compute_class_weights(split="train"):
    mask_dir = os.path.join(DATASET_DIR, "masks", split)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for f in os.listdir(mask_dir):
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        for c in range(NUM_CLASSES):
            counts[c] += (m == c).sum()
    p = counts / counts.sum()
    inv = 1.0 / np.clip(p, 1e-6, None)
    w = inv / inv.sum() * NUM_CLASSES
    print(f"[class weight] train pixel: {dict(zip(CLASS_NAMES.values(), (p*100).round(2)))}")
    print(f"[class weight] weights:     {dict(zip(CLASS_NAMES.values(), w.round(2)))}")
    return torch.from_numpy(w).float().to(DEVICE)


def wce_dice_loss(logits, labels, weights, alpha=1.0, beta=0.5, eps=1.0):
    logp = F.log_softmax(logits, dim=1)
    loss_ce = F.nll_loss(logp, labels, weight=weights, ignore_index=255)
    pred = logp.exp()
    loss_dice = torch.zeros(1, device=logits.device)
    valid = labels != 255
    C = logits.shape[1]
    for c in range(C):
        m = (labels == c).float()
        z = pred[:, c, :, :].float()
        if not valid.any():
            continue
        inter = (z * m * valid.unsqueeze(0)).sum(dim=(1, 2))
        denom = z.sum(dim=(1, 2)) + m.sum(dim=(1, 2)) + eps
        loss_dice += (1 - inter / denom).mean()
    loss_dice = loss_dice / C
    return alpha * loss_ce + beta * loss_dice


def compute_metrics(preds, labels):
    m = evaluate.load("mean_iou")
    r = m.compute(predictions=preds.argmax(1), references=labels,
                  num_labels=NUM_CLASSES, ignore_index=255)
    return {"mIoU": r["mean_iou"], "per_class_iou": r["per_category_iou"]}


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ds = SegDataset("train", get_transform("train"))
    val_ds = SegDataset("val", get_transform("val"))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Classes: {NUM_CLASSES}  Size: {IMAGE_SIZE}")
    train_ld = DataLoader(train_ds, BATCH_SIZE, True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, 1, False, num_workers=0)

    class_weights = compute_class_weights("train")

    # 从 v11b 权重续跑 (已 5 类, 直接加载)
    print(f"[weights] 续跑起点: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(DEVICE)

    # 全量微调 (不冻结)
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  [phase] UNFREEZE all -> trainable {n_train}")

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, NUM_EPOCHS, eta_min=1e-6)

    best = 0.0; best_wh = 0.0
    for ep in range(NUM_EPOCHS):
        model.train(); total = 0.0; n_batches = 0
        opt.zero_grad()
        for b in tqdm(train_ld, desc=f"E{ep+1}/{NUM_EPOCHS} train"):
            pv = b["pixel_values"].to(DEVICE); lb = b["labels"].to(DEVICE)
            out = model(pixel_values=pv)
            logits = out.logits
            if logits.shape[-2:] != lb.shape[-2:]:
                logits = F.interpolate(logits, size=lb.shape[-2:], mode="bilinear", align_corners=False)
            loss = wce_dice_loss(logits, lb, class_weights, LOSS_ALPHA, LOSS_BETA, DICE_EPS)
            (loss / GRAD_ACCUM).backward()
            total += loss.item()
            n_batches += 1
            if n_batches % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        if n_batches % GRAD_ACCUM != 0:
            opt.step(); opt.zero_grad()
        sched.step()

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for b in tqdm(val_ld, desc=f"E{ep+1} val"):
                out = model(pixel_values=b["pixel_values"].to(DEVICE))
                logits = F.interpolate(out.logits, b["labels"].shape[-2:],
                                       mode="bilinear", align_corners=False)
                preds.append(logits.cpu()); gts.append(b["labels"])
        m = compute_metrics(torch.cat(preds), torch.cat(gts))
        ious = m["per_class_iou"]
        print(f"E{ep+1}: loss={total/n_batches:.4f}  mIoU={m['mIoU']:.4f}  "
              + "  ".join(f"{CLASS_NAMES[c]}={ious[c]:.3f}" for c in range(NUM_CLASSES)))
        if m["mIoU"] > best:
            best = m["mIoU"]; model.save_pretrained(OUTPUT_DIR)
            print(f"  => best mIoU ({best:.4f})")
        if ious[1] > best_wh:
            best_wh = ious[1]
    print(f"Done. best mIoU={best:.4f}  best WH IoU={best_wh:.4f}  saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
