"""共享数据加载与增强 — 6 类水葫芦语义分割基线

类号: 0=背景, 1=船只(Boat), 2=桥梁(Bridge), 3=岸基建筑(Structure),
      4=水葫芦(WH), 5=树木(tree)
数据: datasets/hyacinth_seg  (train 209 / val 71)
口径: 与 scripts/segformer_train.py 完全一致 (512x512, ImageNet 归一化)
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = r"D:\chengs\9.project\shuihulu"
DATASET_DIR = os.path.join(ROOT, "datasets", "hyacinth_seg")
IMAGE_SIZE = 512
NUM_CLASSES = 6
CLASS_NAMES = {0: "bg", 1: "Boat", 2: "Bridge", 3: "Structure", 4: "WH", 5: "tree"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class SegDataset(Dataset):
    """与 segformer_train.py 相同的目录约定: images/<split>/<fname>, masks/<split>/<stem>.png"""

    def __init__(self, split, transform=None):
        self.img_dir = os.path.join(DATASET_DIR, "images", split)
        self.mask_dir = os.path.join(DATASET_DIR, "masks", split)
        self.transform = transform
        self.images = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        stem = os.path.splitext(fname)[0]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            a = self.transform(image=img, mask=mask)
            return a["image"], a["mask"].long()
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t, torch.from_numpy(mask).long()


def get_transform(split):
    if split == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.3),
            A.Normalize(MEAN, STD), ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(MEAN, STD), ToTensorV2(),
    ])


def class_weights_from_masks(split="train", eps=1e-3):
    """按训练集像素占比倒数的归一化类别权重 (与论文公式一致), 含 p_c 下界防爆炸."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    mask_dir = os.path.join(DATASET_DIR, "masks", split)
    for f in os.listdir(mask_dir):
        if not f.lower().endswith(".png"):
            continue
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        for c in range(NUM_CLASSES):
            counts[c] += int((m == c).sum())
    total = counts.sum()
    p_c = np.maximum(counts / max(total, 1), eps)
    w = (1.0 / p_c)
    return w / w.sum(), p_c


# 预置类别权重（训练时若需快速取值可 import 本常量; 也可运行时用上面函数重算）
PRECOMPUTED_W, PRECOMPUTED_P = class_weights_from_masks("train")
print("class priors  :", np.round(PRECOMPUTED_P, 5))
print("class weights :", np.round(PRECOMPUTED_W, 3))
