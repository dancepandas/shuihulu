"""SegFormer B2 语义分割 — 水葫芦 (6类: 0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree)
数据: LS手动标注 + V9 合并 (datasets/hyacinth_seg)
"""
import os, numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import evaluate

ROOT = r"D:\chengs\9.project\shuihulu"
DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_seg")
OUTPUT_DIR = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls+v9")
MODEL_PATH = os.path.join(ROOT, "weights", "segformer_b2")
IMAGE_SIZE, BATCH_SIZE = 512, 2
NUM_EPOCHS, LR, WD = 80, 6e-5, 0.01
NUM_CLASSES = 6  # 0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "bg", 1: "Boat", 2: "Bridge", 3: "Structure", 4: "WH", 5: "tree"}

# ---- Dataset ----
class SegDataset(Dataset):
    def __init__(self, split, transform=None):
        self.img_dir = os.path.join(DATASET_DIR, "images", split)
        self.mask_dir = os.path.join(DATASET_DIR, "masks", split)
        self.transform = transform
        self.images = sorted(f for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg",".png",".jpeg")))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]; stem = os.path.splitext(fname)[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            a = self.transform(image=img, mask=mask)
            return {"pixel_values": a["image"], "labels": a["mask"].long()}
        return {"pixel_values": torch.from_numpy(img).permute(2,0,1).float()/255.0,
                "labels": torch.from_numpy(mask).long()}

# ---- Aug ----
def get_transform(split):
    if split == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-15,15), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.3),
            A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ToTensorV2(),
    ])

# ---- Metrics ----
def compute_metrics(preds, labels):
    m = evaluate.load("mean_iou")
    r = m.compute(predictions=preds.argmax(1), references=labels,
                  num_labels=NUM_CLASSES, ignore_index=255)
    return {"mIoU": r["mean_iou"], "per_class_iou": r["per_category_iou"]}

# ---- Train ----
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ds = SegDataset("train", get_transform("train"))
    val_ds = SegDataset("val", get_transform("val"))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Classes: {NUM_CLASSES}")
    train_ld = DataLoader(train_ds, BATCH_SIZE, True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, 1, False, num_workers=0)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
        id2label=CLASS_NAMES, label2id={v:k for k,v in CLASS_NAMES.items()}).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, NUM_EPOCHS, eta_min=1e-6)

    best = 0.0
    for ep in range(NUM_EPOCHS):
        model.train(); total = 0.0
        for b in tqdm(train_ld, desc=f"E{ep+1}/{NUM_EPOCHS} train"):
            out = model(pixel_values=b["pixel_values"].to(DEVICE), labels=b["labels"].to(DEVICE))
            opt.zero_grad(); out.loss.backward(); opt.step()
            total += out.loss.item()
        sched.step()

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for b in tqdm(val_ld, desc=f"E{ep+1} val"):
                out = model(pixel_values=b["pixel_values"].to(DEVICE))
                logits = nn.functional.interpolate(out.logits, b["labels"].shape[-2:],
                                                   mode="bilinear", align_corners=False)
                preds.append(logits.cpu()); gts.append(b["labels"])
        m = compute_metrics(torch.cat(preds), torch.cat(gts))
        ious = m["per_class_iou"]
        print(f"E{ep+1}: loss={total/len(train_ld):.4f}  mIoU={m['mIoU']:.4f}  "
              f"bg={ious[0]:.3f} Boat={ious[1]:.3f} Brg={ious[2]:.3f} Str={ious[3]:.3f} WH={ious[4]:.3f} tree={ious[5]:.3f}")
        if m["mIoU"] > best:
            best = m["mIoU"]; model.save_pretrained(OUTPUT_DIR)
            print(f"  => best ({best:.4f})")
    print(f"Done. best mIoU={best:.4f}  saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
