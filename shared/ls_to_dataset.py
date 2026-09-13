"""
LS Brush PNG export → 语义分割训练集
- 输入: data/ls_export_p7/ (task-{id}-annotation-*-label-{name}-*.png)
- 输出: datasets/hyacinth_ls/
  - images/  → 原始 JPEG (从 JSONL 按 task_id 匹配)
  - masks/   → 单张灰度 PNG (0=bg, 1=Boat, 2=Bridge, 3=Structure, 4=WH, 5=tree)
"""
import os, re, json, base64, shutil
import numpy as np
import cv2
from collections import defaultdict

ROOT = r"D:\chengs\9.project\shuihulu"
EXPORT_DIR = os.path.join(ROOT, "data", "ls_export_p7")
JSONL_PATH = os.path.join(ROOT, "data", "v9_500.jsonl")
OUT_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls")
TASK_ID_BASE = 383  # LS project 7 当前这批的第一个 task id
TASK_ID_MIN = 383   # 用户手动改过的范围
TASK_ID_MAX = 509

# 标签名 → 类别 ID
LABEL2ID = {
    "Boat": 1,
    "Bridge": 2,
    "Structure": 3,
    "Water Hyacinth": 4,
    "tree": 5,
}
ID2NAME = {v: k for k, v in LABEL2ID.items()}

# --- 1. 加载 JSONL (按索引 = task_id - BASE) ---
with open(JSONL_PATH, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]
print(f"JSONL: {len(records)} records")

# --- 2. 收集 mask PNG, 按 task 分组 ---
task_masks = defaultdict(list)  # task_id → [(label_name, png_path)]
for fname in os.listdir(EXPORT_DIR):
    m = re.match(r'task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+\.png', fname)
    if not m:
        continue
    task_id = int(m.group(1))
    label = m.group(2)
    task_masks[task_id].append((label, os.path.join(EXPORT_DIR, fname)))

print(f"tasks with masks: {len(task_masks)}")

# --- 3. 合并 mask + 匹配图片 ---
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

done = 0
for task_id, mask_items in sorted(task_masks.items()):
    if task_id < TASK_ID_MIN or task_id > TASK_ID_MAX:
        continue  # 只取用户手动改过的范围
    idx = task_id - TASK_ID_BASE
    if idx < 0 or idx >= len(records):
        print(f"  task {task_id}: idx={idx} out of range, skip")
        continue

    # 解码图片
    rec = records[idx]
    b64_str = rec["data"]["image"]
    try:
        _, encoded = b64_str.split(",", 1)
    except ValueError:
        encoded = b64_str
    img_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  task {task_id}: image decode failed, skip")
        continue
    H, W = img.shape[:2]

    # 创建合并 mask: 0=bg, 然后按 LABEL2ID 填
    merged_mask = np.zeros((H, W), dtype=np.uint8)
    for label_name, png_path in mask_items:
        png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if png is None:
            continue
        # 处理尺寸不一致 (PNG 可能有 alpha 裁剪)
        pH, pW = png.shape[:2]
        if pH != H or pW != W:
            png = cv2.resize(png, (W, H), interpolation=cv2.INTER_NEAREST)
        cls_id = LABEL2ID.get(label_name, 0)
        merged_mask[png > 0] = cls_id

    # 保存
    stem = f"ls_{task_id:04d}"
    cv2.imwrite(os.path.join(OUT_DIR, "images", f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(OUT_DIR, "masks", f"{stem}.png"), merged_mask)
    done += 1

print(f"\nDone: {done} image/mask pairs saved to {OUT_DIR}")

# 统计
print(f"\nClass distribution:")
for cls_id in sorted(LABEL2ID.values()):
    name = ID2NAME[cls_id]
    count = sum(1 for f in os.listdir(os.path.join(OUT_DIR, "masks"))
                if f.endswith(".png") and cls_id in np.unique(cv2.imread(os.path.join(OUT_DIR, "masks", f), cv2.IMREAD_GRAYSCALE)))
    print(f"  {cls_id} {name}: {count} images")
