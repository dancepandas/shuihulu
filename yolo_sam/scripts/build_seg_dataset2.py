"""
合并 4 个来源 → 语义分割训练集 hyacinth_seg2 (6类: 0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree)
  A. 现有 hyacinth_seg 中的 ls_* (新服务器手动标注 383-509, 126对)
  B. p5 YOLO 导出 (edge-push 项目, 104对; 替代 v9_ —— V9 的103张是 p5 子集)
  C. p7o 笔刷导出 (老服务器项目7, 275对)
  D. p9  笔刷导出 (老服务器项目9, 61对)
输出: datasets/hyacinth_seg2/{images,masks}/{train,val} (80/20, seed 42)
"""
import os, re, random, shutil, hashlib
import numpy as np
import cv2
from collections import defaultdict

ROOT = r"D:\chengs\9.project\shuihulu"
OUT = os.path.join(ROOT, "datasets", "hyacinth_seg2")
TMP = os.path.join(ROOT, "datasets", "_seg2_tmp")

LABEL2ID = {"Boat": 1, "Bridge": 2, "Structure": 3, "Water Hyacinth": 4, "tree": 5}
P5_MAP = {0: 5, 1: 2, 2: 4, 3: 3, 4: 1}  # p5 classes.txt: 0树 1桥 2水葫芦 3结构 4船

if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
if os.path.exists(OUT):
    shutil.rmtree(OUT)

def save_pair(stem, img, mask):
    cv2.imwrite(os.path.join(TMP, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(TMP, stem + ".png"), mask)

def brush_to_mask(items, H, W):
    """items: [(label, png_path)] → 合并 mask"""
    mask = np.zeros((H, W), dtype=np.uint8)
    for lbl, pth in items:
        png = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        if png is None:
            continue
        if png.shape[:2] != (H, W):
            png = cv2.resize(png, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[png > 0] = LABEL2ID.get(lbl, 0)
    return mask

# ============ A. ls_* from hyacinth_seg ============
cnt_a = 0
SRC = os.path.join(ROOT, "datasets", "hyacinth_seg")
for split in ["train", "val"]:
    idir, mdir = os.path.join(SRC, "images", split), os.path.join(SRC, "masks", split)
    for f in os.listdir(idir):
        if not f.startswith("ls_"):
            continue
        stem = f[:-4]
        img = cv2.imread(os.path.join(idir, f))
        mask = cv2.imread(os.path.join(mdir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        save_pair(stem, img, mask)
        cnt_a += 1
print(f"A. ls_* (新服务器): {cnt_a}")

# ============ B. p5 YOLO ============
cnt_b = 0
P5 = os.path.join(ROOT, "data", "p5_yolo")
for f in sorted(os.listdir(os.path.join(P5, "images"))):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    stem = os.path.splitext(f)[0]
    img = cv2.imread(os.path.join(P5, "images", f))
    if img is None:
        continue
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    lbl_path = os.path.join(P5, "labels", stem + ".txt")
    if os.path.exists(lbl_path):
        with open(lbl_path) as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                cls = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if len(vals) == 4:  # bbox → 矩形
                    xc, yc, w, h = vals
                    x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                    x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), P5_MAP[cls], -1)
                else:  # polygon
                    pts = np.array(vals).reshape(-1, 2)
                    pts[:, 0] *= W
                    pts[:, 1] *= H
                    cv2.fillPoly(mask, [pts.astype(np.int32)], P5_MAP[cls])
    save_pair(f"p5_{stem}", img, mask)
    cnt_b += 1
print(f"B. p5 (YOLO): {cnt_b}")

# ============ C/D. p7o / p9 笔刷 ============
for tag, mask_dir, img_dir in [
    ("o7", os.path.join(ROOT, "data", "ls_export_p7o"), os.path.join(ROOT, "data", "ls_images_p7o")),
    ("p9", os.path.join(ROOT, "data", "ls_export_p9"),  os.path.join(ROOT, "data", "ls_images_p9")),
]:
    task_files = defaultdict(list)
    for fname in os.listdir(mask_dir):
        m = re.match(r"task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+\.png", fname)
        if m:
            task_files[int(m.group(1))].append((m.group(2), os.path.join(mask_dir, fname)))
    cnt = 0
    for tid, items in sorted(task_files.items()):
        img_path = os.path.join(img_dir, f"{tid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {tag} task {tid}: image missing, skip")
            continue
        H, W = img.shape[:2]
        mask = brush_to_mask(items, H, W)
        save_pair(f"{tag}_{tid}", img, mask)
        cnt += 1
    print(f"{'C' if tag=='o7' else 'D'}. {tag} (笔刷): {cnt}")

# ============ 去重 (64x64 签名) ============
def sig(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return hashlib.md5(cv2.resize(img, (64, 64)).tobytes()).hexdigest()

jpgs = sorted(f for f in os.listdir(TMP) if f.endswith(".jpg"))
seen, dups = set(), []
for f in jpgs:
    s = sig(os.path.join(TMP, f))
    if s in seen:
        dups.append(f)
    else:
        seen.add(s)
for f in dups:
    os.remove(os.path.join(TMP, f))
    os.remove(os.path.join(TMP, f[:-4] + ".png"))
jpgs = [f for f in jpgs if f not in dups]
print(f"\n去重: 移除 {len(dups)} 对" + (f" → {dups}" if dups else ""))

# ============ 划分 train/val (80/20) ============
random.seed(42)
random.shuffle(jpgs)
n_val = max(1, len(jpgs) // 5)
val_set = set(jpgs[:n_val])

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks", split), exist_ok=True)
for f in jpgs:
    s = "val" if f in val_set else "train"
    stem = f[:-4]
    shutil.move(os.path.join(TMP, f), os.path.join(OUT, "images", s, f))
    shutil.move(os.path.join(TMP, stem + ".png"), os.path.join(OUT, "masks", s, stem + ".png"))
shutil.rmtree(TMP)

n_tr = len(os.listdir(os.path.join(OUT, "images", "train")))
n_va = len(os.listdir(os.path.join(OUT, "images", "val")))
print(f"\nDone. Total: {len(jpgs)} ({n_tr} train / {n_va} val)")

# 类别统计
for cid, name in [(1, "Boat"), (2, "Bridge"), (3, "Structure"), (4, "WH"), (5, "tree")]:
    cnt = 0
    for split in ["train", "val"]:
        for f in os.listdir(os.path.join(OUT, "masks", split)):
            m = cv2.imread(os.path.join(OUT, "masks", split, f), cv2.IMREAD_GRAYSCALE)
            if cid in m:
                cnt += 1
    print(f"  {name}: {cnt} images")
