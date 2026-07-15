"""
ExG + V6 双路融合预标注流水线

规则:
1. V6 WH mask  ∩ ExG → A级水葫芦 polygon
2. V6 tree mask ∩ ExG → A级树 polygon
3. V6 船/桥/建筑 box → 直接保留
4. ExG - V6 - 船区 → 两侧15%=树, 中间70%=水葫芦 (B级), 船框范围=丢弃
"""
import cv2, json, sys, warnings
from pathlib import Path
from collections import Counter

import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────
V6_PATH = "runs/segment/runs/segment/hyacinth6_yolo_sam2/weights/best.pt"
DATA_DIR = Path("data")
OUT_IMAGES = Path("datasets/hyacinth7/images/train")
OUT_LABELS = Path("datasets/hyacinth7/labels/train")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]
SIDE_RATIO = 0.05       # 两侧各5%判为树
CONF = 0.10              # V6置信度阈值
EXG_OTSU_SCALE = 1.5     # Otsu阈值缩放(>1更激进)

# skip non-image files
SKIP_EXT = {".mrk", ".nav", ".obs", ".rtk", ".zip"}
# ───────────────────────────────────────────────────

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

model = YOLO(V6_PATH)

images = []
for root, _, files in DATA_DIR.walk():
    for f in files:
        p = Path(root) / f
        if p.suffix.lower() in SKIP_EXT:
            continue
        if p.suffix.lower() in IMG_EXT:
            images.append(p)
images = sorted(images)
print(f"Images: {len(images)}")

cnt_a = Counter()   # A级标签
cnt_b = Counter()   # B级标签
tagged = 0

for idx, img_path in enumerate(images):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # ── 1. ExG 植被掩膜 ──────────────────────────
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    exg = 2 * G - R - B
    exg_u8 = ((exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, veg_mask = cv2.threshold(exg_u8, int(otsu_thr * EXG_OTSU_SCALE), 255, cv2.THRESH_BINARY)

    # ── 2. V6 推理 ───────────────────────────────
    results = model(str(img_path), conf=CONF, iou=0.5, imgsz=640, verbose=False,
                    retina_masks=True)
    boxes = results[0].boxes
    v6_cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) > 0 else np.array([])
    v6_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) > 0 else np.zeros((0, 4))
    v6_xy = results[0].masks.xy if results[0].masks and hasattr(results[0].masks, "xy") else []

    # ── 3. 构建船框掩膜 ──────────────────────────
    boat_mask = np.zeros((h, w), dtype=np.uint8)
    for j in range(len(v6_cls)):
        if v6_cls[j] == 0:  # Boat
            x1, y1, x2, y2 = v6_xyxy[j].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boat_mask[y1:y2, x1:x2] = 255

    # 两侧/中间分界
    left_cut = int(w * SIDE_RATIO)
    right_cut = int(w * (1 - SIDE_RATIO))

    yolo_lines = []

    # ── 4. 规则 1 & 2: V6 mask ∩ ExG → A 级 ─────
    for j in range(len(v6_cls)):
        c = v6_cls[j]
        if c >= 5:
            continue
        cls_name = NAMES[c]

        # 船/桥/建筑: 规则 3, 直接保留
        if c in (0, 1, 2):  # Boat, Bridge, Structure
            x1, y1, x2, y2 = v6_xyxy[j]
            x1n, y1n = x1 / w, y1 / h
            x2n, y2n = x2 / w, y2 / h
            poly = f"{x1n:.6f} {y1n:.6f} {x2n:.6f} {y1n:.6f} {x2n:.6f} {y2n:.6f} {x1n:.6f} {y2n:.6f}"
            yolo_lines.append(f"{c} {poly}")
            cnt_a[cls_name] += 1
            continue

        # 水葫芦/树: 需 ExG 交集确认
        if c in (3, 4):  # Water Hyacinth, tree
            if j < len(v6_xy) and len(v6_xy[j]) >= 3:
                pts = v6_xy[j]
                # 计算与ExG的交集比例
                mask_j = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_j, [pts.astype(np.int32)], 255)
                inter = cv2.countNonZero(cv2.bitwise_and(mask_j, veg_mask))
                area = cv2.countNonZero(mask_j)
                ratio = inter / (area + 1)

                if ratio > 0.3:  # 交集 > 30%, A级
                    # Clip to ExG mask for finer boundary
                    masked = cv2.bitwise_and(mask_j, veg_mask)
                    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) < 50:
                            continue
                        pts_norm = [f"{p[0][0] / w:.6f} {p[0][1] / h:.6f}"
                                    for p in contour]
                        if len(pts_norm) >= 3:
                            yolo_lines.append(f"{c} {' '.join(pts_norm)}")
                            cnt_a[cls_name] += 1
                else:
                    # 交集不足, 丢弃
                    pass

    # ── 5. 规则 4: ExG - V6所有框 - 船区 → B级 ───
    # 构建 V6 全框掩膜(所有类别)
    v6_all_mask = np.zeros((h, w), dtype=np.uint8)
    for j in range(len(v6_cls)):
        x1, y1, x2, y2 = v6_xyxy[j].astype(int)
        cv2.rectangle(v6_all_mask, (max(0, x1), max(0, y1)),
                      (min(w, x2), min(h, y2)), 255, -1)

    # ExG 独有 = 植被掩膜 - V6全框 - 船区
    exg_only = cv2.bitwise_and(veg_mask, cv2.bitwise_not(v6_all_mask))
    exg_only = cv2.bitwise_and(exg_only, cv2.bitwise_not(boat_mask))

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exg_only = cv2.morphologyEx(exg_only, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(exg_only, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        # 判断质心位置
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        if cx < left_cut or cx > right_cut:
            cls_id = 4  # tree
            cls_name = "tree"
        else:
            cls_id = 3  # Water Hyacinth
            cls_name = "Water Hyacinth"

        pts_norm = [f"{p[0][0] / w:.6f} {p[0][1] / h:.6f}" for p in contour]
        if len(pts_norm) >= 3:
            yolo_lines.append(f"{cls_id} {' '.join(pts_norm)}")
            cnt_b[cls_name] += 1

    # ── 6. 写入标签 ───────────────────────────────
    if yolo_lines:
        stem = img_path.stem
        with open(OUT_LABELS / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        # 拷贝图片符号链接(节省空间)
        dst = OUT_IMAGES / img_path.name
        if not dst.exists():
            import shutil
            shutil.copy2(img_path, dst)
        tagged += 1
    else:
        # 即使无标签也复制图片(保持数据集完整)
        dst = OUT_IMAGES / img_path.name
        if not dst.exists():
            import shutil
            shutil.copy2(img_path, dst)

    if (idx + 1) % 100 == 0:
        print(f"  [{idx + 1}/{len(images)}] tagged {tagged}, "
              f"A: {sum(cnt_a.values())}, B: {sum(cnt_b.values())}")

# ── 7. 输出统计 ───────────────────────────────────
print(f"\n{'='*50}")
print(f"Total images: {len(images)}")
print(f"With labels:  {tagged}")
print(f"A级标签 ({sum(cnt_a.values())}):")
for c, n in cnt_a.most_common():
    print(f"  {c}: {n}")
print(f"B级标签 ({sum(cnt_b.values())}):")
for c, n in cnt_b.most_common():
    print(f"  {c}: {n}")

# 生成数据集配置文件
config = f"""# 水葫芦 5 类 ExG+V6 融合数据集
path: datasets/hyacinth7
train: images/train
val: images/train

names:
  0: Boat
  1: Bridge
  2: Structure
  3: Water Hyacinth
  4: tree
"""
cfg_path = Path("configs/dataset_hyacinth7_seg.yaml")
cfg_path.write_text(config, encoding="utf-8")
print(f"\nConfig: {cfg_path}")
print("Done!")
