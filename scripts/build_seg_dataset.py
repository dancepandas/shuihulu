"""
合并 V9 + LS 数据集 → 语义分割训练集 (6类: 0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree)
V9: YOLO polygon (class 0-4) → +1 offset → mask 1-5
LS: Brush PNG export (mask 0-5, bg already 0) → 直接复用
"""
import os, re, json, base64, random, shutil
import numpy as np
import cv2
from collections import defaultdict

ROOT = r"D:\chengs\9.project\shuihulu"
OUT = os.path.join(ROOT, "datasets", "hyacinth_seg")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# A. LS 数据 (Brush PNG export, 手动标注 383-496)
# ============================================================
LS_EXPORT = os.path.join(ROOT, "data", "ls_export_p7")
LS_JSONL = os.path.join(ROOT, "data", "v9_500.jsonl")
LABEL2ID = {"Boat": 1, "Bridge": 2, "Structure": 3, "Water Hyacinth": 4, "tree": 5}
TASK_BASE = 383
TASK_MIN, TASK_MAX = 383, 509

# 读 JSONL (按顺序 = task_id - BASE)
with open(LS_JSONL, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# 收集 PNG
task_files = defaultdict(list)
for fname in os.listdir(LS_EXPORT):
    m = re.match(r'task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+\.png', fname)
    if not m: continue
    tid = int(m.group(1))
    if TASK_MIN <= tid <= TASK_MAX:
        task_files[tid].append((m.group(2), os.path.join(LS_EXPORT, fname)))

cnt_ls = 0
for tid, items in sorted(task_files.items()):
    idx = tid - TASK_BASE
    if idx < 0 or idx >= len(records): continue
    # 解图片
    b64 = records[idx]["data"]["image"]
    _, enc = b64.split(",", 1) if "," in b64 else ("", b64)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(enc), np.uint8), cv2.IMREAD_COLOR)
    if img is None: continue
    H, W = img.shape[:2]
    # 合并 mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for lbl, pth in items:
        png = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
        if png is None: continue
        if png.shape[:2] != (H, W):
            png = cv2.resize(png, (W, H), interpolation=cv2.INTER_NEAREST)
        cid = LABEL2ID.get(lbl, 0)
        mask[png > 0] = cid
    stem = f"ls_{tid:04d}"
    cv2.imwrite(os.path.join(OUT, f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(OUT, f"{stem}.png"), mask)
    cnt_ls += 1
print(f"LS: {cnt_ls} pairs")

# ============================================================
# B. V9 数据 (YOLO polygon, class 0-4 → +1 → mask 1-5)
# ============================================================
V9_DIR = os.path.join(ROOT, "datasets", "hyacinth9")
V9_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}  # YOLO cls → mask value (0=bg)

cnt_v9 = 0
for split in ["train", "val"]:
    img_dir = os.path.join(V9_DIR, "images", split)
    lbl_dir = os.path.join(V9_DIR, "labels", split)
    if not os.path.isdir(img_dir): continue
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")): continue
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None: continue
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7: continue
                    cls = int(parts[0])
                    pts = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                    pts[:, 0] *= W; pts[:, 1] *= H
                    cv2.fillPoly(mask, [pts.astype(np.int32)], V9_MAP[cls])
        stem_out = f"v9_{stem}"
        cv2.imwrite(os.path.join(OUT, f"{stem_out}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(OUT, f"{stem_out}.png"), mask)
        cnt_v9 += 1
print(f"V9: {cnt_v9} pairs")

# ============================================================
# C. 划分 train/val (80/20)
# ============================================================
jpgs = sorted([f for f in os.listdir(OUT) if f.endswith(".jpg")])
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
    shutil.move(os.path.join(OUT, f), os.path.join(OUT, "images", s, f))
    shutil.move(os.path.join(OUT, f"{stem}.png"), os.path.join(OUT, "masks", s, f"{stem}.png"))

# 验证
train_m = len([x for x in os.listdir(os.path.join(OUT, "masks", "train")) if x.endswith('.png')])
val_m = len([x for x in os.listdir(os.path.join(OUT, "masks", "val")) if x.endswith('.png')])
print(f"\nDone. Total: {len(jpgs)} ({train_m} train / {val_m} val)")
print(f"LS: {cnt_ls}  V9: {cnt_v9}")

# 统计类别
for cls_id, name in [(1,"Boat"),(2,"Bridge"),(3,"Structure"),(4,"WH"),(5,"tree")]:
    count = 0
    for f in os.listdir(os.path.join(OUT, "masks", "train")):
        m = cv2.imread(os.path.join(OUT, "masks", "train", f), cv2.IMREAD_GRAYSCALE)
        if cls_id in m: count += 1
    for f in os.listdir(os.path.join(OUT, "masks", "val")):
        m = cv2.imread(os.path.join(OUT, "masks", "val", f), cv2.IMREAD_GRAYSCALE)
        if cls_id in m: count += 1
    print(f"  {name}: {count} images")
