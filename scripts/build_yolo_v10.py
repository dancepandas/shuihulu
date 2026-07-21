"""
hyacinth_seg (LS+V9 合并, 217张) → YOLO polygon 格式
mask → contour → YOLO label txt
"""
import os, shutil, random, numpy as np, cv2

ROOT = r"D:\chengs\9.project\shuihulu"
SRC = os.path.join(ROOT, "datasets", "hyacinth_seg")
TEMP = os.path.join(ROOT, "datasets", "_v10_temp")
OUT = os.path.join(ROOT, "datasets", "hyacinth_v10")

MASK2YOLO = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

if os.path.exists(TEMP):
    shutil.rmtree(TEMP)
os.makedirs(TEMP)

count = 0
for split in ["train", "val"]:
    md = os.path.join(SRC, "masks", split)
    for f in os.listdir(md):
        if not f.endswith(".png"): continue
        stem = f[:-4]
        img = None
        for e in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(SRC, "images", split, stem + e)
            if os.path.exists(p): img = p; break
        if img is None: continue
        mask = cv2.imread(os.path.join(md, f), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        H, W = mask.shape

        # mask → YOLO polygon
        lines = []
        for mv, yc in MASK2YOLO.items():
            binary = (mask == mv).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if len(c) < 3 or cv2.contourArea(c) < 50: continue
                pts = c.reshape(-1, 2).astype(float)
                pts[:, 0] /= W; pts[:, 1] /= H
                lines.append(f"{yc} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))

        name = f"img_{count:04d}"
        shutil.copy2(img, os.path.join(TEMP, name + os.path.splitext(img)[1]))
        with open(os.path.join(TEMP, name + ".txt"), "w") as lf:
            lf.write("\n".join(lines) + "\n" if lines else "")
        count += 1

print(f"Total: {count} pairs")

# 划分 train/val (8:2)
pairs = []
for f in os.listdir(TEMP):
    if f.endswith((".jpg", ".jpeg", ".png")):
        stem = os.path.splitext(f)[0]
        lbl = os.path.join(TEMP, stem + ".txt")
        if os.path.exists(lbl):
            pairs.append((os.path.join(TEMP, f), lbl))

random.seed(42)
random.shuffle(pairs)
n_val = max(1, len(pairs) // 5)

for s in ["train", "val"]:
    for d in ["images", "labels"]:
        os.makedirs(os.path.join(OUT, d, s), exist_ok=True)

for i, (img, lbl) in enumerate(pairs):
    s = "val" if i < n_val else "train"
    shutil.copy2(img, os.path.join(OUT, "images", s, os.path.basename(img)))
    shutil.copy2(lbl, os.path.join(OUT, "labels", s, os.path.splitext(os.path.basename(img))[0] + ".txt"))

print(f"Done: {len(pairs)} total ({len(pairs)-n_val} train / {n_val} val)")
shutil.rmtree(TEMP)
