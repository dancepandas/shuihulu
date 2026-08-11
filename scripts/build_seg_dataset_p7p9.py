"""
project-7 + project-9 Brush PNG export → SegFormer 语义分割训练集
只使用这两个 zip 的数据，不混入 p5 / v9 / 新服务器 LS 数据。
输出: datasets/hyacinth_seg_p7p9/{images,masks}/{train,val}
类别: 0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree
"""
import os, re, random, shutil
import numpy as np
import cv2
from collections import defaultdict

ROOT = r"D:\chengs\9.project\shuihulu"
OUT = os.path.join(ROOT, "datasets", "hyacinth_seg_p7p9")
TMP = os.path.join(ROOT, "datasets", "_p7p9_tmp")

LABEL2ID = {"Boat": 1, "Bridge": 2, "Structure": 3, "Water Hyacinth": 4, "tree": 5}

SOURCES = [
    ("p7", os.path.join(ROOT, "data", "ls_export_p7o"), os.path.join(ROOT, "data", "ls_images_p7o")),
    ("p9", os.path.join(ROOT, "data", "ls_export_p9"),  os.path.join(ROOT, "data", "ls_images_p9")),
]

if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
if os.path.exists(OUT):
    shutil.rmtree(OUT)

def save_pair(stem, img, mask):
    cv2.imwrite(os.path.join(TMP, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(TMP, stem + ".png"), mask)

def build_mask(items, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    for label_name, png_path in items:
        png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if png is None:
            continue
        if png.shape[:2] != (H, W):
            png = cv2.resize(png, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[png > 0] = LABEL2ID.get(label_name, 0)
    return mask

total = 0
for tag, mask_dir, img_dir in SOURCES:
    task_files = defaultdict(list)
    for fname in os.listdir(mask_dir):
        m = re.match(r"task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+\.png", fname)
        if not m:
            continue
        tid = int(m.group(1))
        label = m.group(2)
        task_files[tid].append((label, os.path.join(mask_dir, fname)))

    cnt = 0
    for tid, items in sorted(task_files.items()):
        img_path = os.path.join(img_dir, f"{tid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {tag} task {tid}: image missing, skip")
            continue
        H, W = img.shape[:2]
        mask = build_mask(items, H, W)
        save_pair(f"{tag}_{tid}", img, mask)
        cnt += 1
    total += cnt
    print(f"{tag}: {cnt} pairs")

jpgs = sorted(f for f in os.listdir(TMP) if f.endswith(".jpg"))
random.seed(42)
random.shuffle(jpgs)
n_val = max(1, len(jpgs) // 5)
val_set = set(jpgs[:n_val])

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks", split), exist_ok=True)

for f in jpgs:
    split = "val" if f in val_set else "train"
    stem = f[:-4]
    shutil.move(os.path.join(TMP, f), os.path.join(OUT, "images", split, f))
    shutil.move(os.path.join(TMP, stem + ".png"), os.path.join(OUT, "masks", split, stem + ".png"))
shutil.rmtree(TMP)

train_n = len(os.listdir(os.path.join(OUT, "images", "train")))
val_n = len(os.listdir(os.path.join(OUT, "images", "val")))
print(f"\nDone. Total: {len(jpgs)} ({train_n} train / {val_n} val)")

for cid, name in [(1, "Boat"), (2, "Bridge"), (3, "Structure"), (4, "WH"), (5, "tree")]:
    cnt = 0
    for split in ["train", "val"]:
        mask_dir = os.path.join(OUT, "masks", split)
        for f in os.listdir(mask_dir):
            m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
            if m is not None and cid in m:
                cnt += 1
    print(f"  {name}: {cnt} images")
