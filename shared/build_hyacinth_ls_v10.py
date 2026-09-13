"""
合并 LS project 10 笔刷 PNG → 单通道类别索引 mask + 落原图。
类别编号：
  0 = water (水体背景)
  1 = water_hyacinth (核心目标)
  2 = hard_structure (硬结构：桥墩/堤坝/护岸)
  3 = shore_vegetation (岸基植被)
  4 = other_aquatic_vegetation (其他水生植物)

输入：data/ls_export_p10/{images, task-*-label-*.png, meta.jsonl}
输出：datasets/hyacinth_ls_v10/{images, masks}/
"""
import os, glob, re, json
import numpy as np, cv2

ROOT = r"D:\chengs\9.project\shuihulu"
EXP = os.path.join(ROOT, "data", "ls_export_p10")
OUT = os.path.join(ROOT, "datasets", "hyacinth_ls_v10")
IMG_DIR = os.path.join(OUT, "images")
MSK_DIR = os.path.join(OUT, "masks")

LABEL2ID = {
    "water": 0,
    "water_hyacinth": 1,
    "hard_structure": 2,
    "shore_vegetation": 3,
    "other_aquatic_vegetation": 4,
}
ID2NAME = {v: k for k, v in LABEL2ID.items()}

# task_id → image basename
task_image = {}
with open(os.path.join(EXP, "meta.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        task_image[t["id"]] = os.path.basename(t["data"]["image"])

# task_id → [(label_name, png_path)]
task_masks = {}
for f in sorted(glob.glob(os.path.join(EXP, "task-*-label-*.png"))):
    base = os.path.basename(f).replace(".png", "")
    m = re.match(r"task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+$", base)
    if not m: continue
    tid = int(m.group(1)); label = m.group(2)
    if label not in LABEL2ID: continue
    task_masks.setdefault(tid, []).append((label, f))

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MSK_DIR, exist_ok=True)

done = skipped = 0
for tid in sorted(task_masks):
    if tid not in task_image:
        skipped += 1; continue
    img_name = task_image[tid]
    img_src = os.path.join(EXP, "images", img_name)
    if not os.path.exists(img_src):
        skipped += 1; continue
    img = cv2.imread(img_src, cv2.IMREAD_COLOR)
    if img is None:
        skipped += 1; continue
    H, W = img.shape[:2]

    merged = np.zeros((H, W), dtype=np.uint8)
    for label, png in task_masks[tid]:
        m = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        cls_id = LABEL2ID[label]
        # 标注 255 = 该类像素；同类多笔刷取或（掩膜 union）
        merged[m > 0] = cls_id

    stem = f"p10_{tid:05d}"
    cv2.imwrite(os.path.join(IMG_DIR, f"{stem}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(MSK_DIR, f"{stem}.png"), merged)
    done += 1

# 统计每类像素占比
total_px = 0
counts = np.zeros(5, dtype=np.int64)
for f in sorted(glob.glob(os.path.join(MSK_DIR, "*.png"))):
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if m is None: continue
    total_px += m.size
    for c in range(5):
        counts[c] += int((m == c).sum())

print(f"done={done}  skipped={skipped}")
print(f"\nclass pixel distribution (out of {total_px:,} total px):")
for c in range(5):
    pct = counts[c] / total_px * 100 if total_px else 0
    print(f"  {c} {ID2NAME[c]:28s}  {counts[c]:>12,} px  {pct:5.2f}%")
print(f"\nout: {OUT}")