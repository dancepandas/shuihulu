"""SegFormer-B2 只用 7 张"几乎全屏水葫芦"截图续跑微调。
从 v11b 权重继续, 训练/验证都只用这 7 张 (stem p10_23047~23053),
低学习率微调, 快速验证模型能否学会全屏水葫芦场景。
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
DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v11")
OUTPUT_DIR = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11_fullscreen")
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11b")  # 续跑起点

# 只用这 7 张全屏水葫芦截图
SEVEN_STEMS = ["p10_23047", "p10_23048", "p10_23049", "p10_23050",
               "p10_23051", "p10_23052", "p10_23053"]

IMAGE_SIZE = 640
BATCH_SIZE = 1
GRAD_ACCUM = 1            # 只有 7 张, 不做梯度累积
NUM_EPOCHS = 100          # 7 张很快, 多跑几个 epoch
LR, WD = 2e-5, 0.01
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
        self.images = [f"{s}.jpg" for s in SEVEN_STEMS]  # 只取 7 张

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
    val_ds = SegDataset("train", get_transform("val"))  # 7 张做验证
    print(f"Train: {len(train_ds)} (7张全屏)  Val: {len(val_ds)}")
    train_ld = DataLoader(train_ds, BATCH_SIZE, True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, 1, False, num_workers=0)

    # 类别权重固定 (水葫芦最高)
    weights = torch.tensor([0.5, 3.0, 1.0, 0.5, 0.5], dtype=torch.float32).to(DEVICE)

    print(f"[weights] 续跑起点: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = True

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, NUM_EPOCHS, eta_min=1e-6)

    best = 0.0
    for ep in range(NUM_EPOCHS):
        model.train(); total = 0.0; n_batches = 0
        opt.zero_grad()
        for b in train_ld:
            pv = b["pixel_values"].to(DEVICE); lb = b["labels"].to(DEVICE)
            out = model(pixel_values=pv)
            logits = out.logits
            if logits.shape[-2:] != lb.shape[-2:]:
                logits = F.interpolate(logits, size=lb.shape[-2:], mode="bilinear", align_corners=False)
            loss = wce_dice_loss(logits, lb, weights, LOSS_ALPHA, LOSS_BETA, DICE_EPS)
            loss.backward()
            total += loss.item()
            n_batches += 1
            opt.step(); opt.zero_grad()
        sched.step()

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for b in val_ld:
                out = model(pixel_values=b["pixel_values"].to(DEVICE))
                logits = F.interpolate(out.logits, b["labels"].shape[-2:],
                                       mode="bilinear", align_corners=False)
                preds.append(logits.cpu()); gts.append(b["labels"])
        m = compute_metrics(torch.cat(preds), torch.cat(gts))
        ious = m["per_class_iou"]
        print(f"E{ep+1}: loss={total/n_batches:.4f}  mIoU={m['mIoU']:.4f}  WH IoU={ious[1]:.4f}  "
              + "  ".join(f"{CLASS_NAMES[c]}={ious[c]:.3f}" for c in range(NUM_CLASSES)))
        if m["mIoU"] > best:
            best = m["mIoU"]; model.save_pretrained(OUTPUT_DIR)
            print(f"  => best mIoU ({best:.4f})")
    print(f"Done. best mIoU={best:.4f}  saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
