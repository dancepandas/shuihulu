"""
GLI(>0.12) + V7 水葫芦 V8 伪标签生成流水线

规则:
1. GLI 固定阈值 > 0.12 → 全图植被掩膜（不归一化，不 Otsu）
2. V7 识别 Boat/Bridge/Structure → 从植被掩膜中排除
3. WH = GLI_veg - V7_non_WH
4. V7 tree 与 WH 重叠 > 50% → 吸收为 WH
5. 输出 YOLO 格式标签到 datasets/hyacinth8/

用法: python scripts/gli_v8_pipeline.py
"""
import cv2, os, shutil, warnings
from pathlib import Path
from collections import Counter

import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────
V7_PATH = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_DIR = Path("data")
OUT_IMAGES = Path("datasets/hyacinth8/images/train")
OUT_LABELS = Path("datasets/hyacinth8/labels/train")
CONFIG_PATH = Path("configs/dataset_hyacinth8_seg.yaml")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_EXT = {".mrk", ".nav", ".obs", ".rtk", ".zip"}
NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]

CONF = 0.10          # V7 置信度
GLI_THRESHOLD = 0.12  # 固定 GLI 阈值
MIN_AREA = 50         # 最小轮廓面积 (px)
TREE_OVERLAP = 0.5    # tree 与 WH 重叠超过此比例则吸收
# ───────────────────────────────────────────────────


def compute_gli_mask(img: np.ndarray) -> np.ndarray:
    """GLI 固定阈值植被掩膜"""
    rgb = img.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    _, veg = cv2.threshold(gli, GLI_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    veg = (veg * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
    return veg


def poly_to_yolo(pts, w, h) -> str:
    flat = [f"{float(p[0]) / w:.6f} {float(p[1]) / h:.6f}" for p in pts]
    return " ".join(flat)


def rect_to_yolo(x1, y1, x2, y2, w, h) -> str:
    return f"{x1 / w:.6f} {y1 / h:.6f} {x2 / w:.6f} {y1 / h:.6f} {x2 / w:.6f} {y2 / h:.6f} {x1 / w:.6f} {y2 / h:.6f}"


# ── 主流程 ────────────────────────────────────────
def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    model = YOLO(V7_PATH)

    # 收集图片
    images = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SKIP_EXT:
                continue
            if p.suffix.lower() in IMG_EXT:
                images.append(p)
    print(f"Images: {len(images)}")

    cnt = Counter()
    tagged, total_wh, total_tree = 0, 0, 0

    for idx, img_path in enumerate(sorted(images)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. GLI 固定阈值
        veg = compute_gli_mask(img)

        # 2. V7 推理
        results = model(str(img_path), conf=CONF, iou=0.5, imgsz=640,
                        verbose=False)
        r0 = results[0]
        v7_cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else np.array([])
        v7_xyxy = r0.boxes.xyxy.cpu().numpy() if len(v7_cls) > 0 else np.zeros((0, 4))
        v7_xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

        # 3. V7 非 WH 掩膜 (Boat=0, Bridge=1, Structure=2)
        exclude = np.zeros((h, w), np.uint8)
        tree_pts = []
        for j in range(len(v7_cls)):
            c = int(v7_cls[j])
            if c in (0, 1, 2):
                if j < len(v7_xy) and len(v7_xy[j]) >= 3:
                    cv2.fillPoly(exclude, [v7_xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = v7_xyxy[j].astype(int)
                    cv2.rectangle(exclude, (max(0, x1), max(0, y1)),
                                  (min(w, x2), min(h, y2)), 255, -1)
            elif c == 4:
                if j < len(v7_xy) and len(v7_xy[j]) >= 3:
                    tree_pts.append(v7_xy[j])

        # 4. WH = GLI - exclude
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        wh_mask = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh_mask = cv2.morphologyEx(wh_mask, cv2.MORPH_OPEN, kernel)
        wh_cs, _ = cv2.findContours(wh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_lines = []

        # 写入 WH
        wh_union = np.zeros((h, w), np.uint8)
        for ct in wh_cs:
            if cv2.contourArea(ct) > MIN_AREA:
                cv2.fillPoly(wh_union, [ct], 255)
                yolo_lines.append(f"3 {poly_to_yolo(ct[:, 0, :], w, h)}")
                cnt["Water Hyacinth"] += 1
                total_wh += 1

        # 5. tree 与 WH 重叠判断
        for pts in tree_pts:
            tm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
            overlap = cv2.countNonZero(cv2.bitwise_and(tm, wh_union))
            area = cv2.countNonZero(tm)
            if area > 0 and overlap / area < TREE_OVERLAP:
                yolo_lines.append(f"4 {poly_to_yolo(pts, w, h)}")
                cnt["tree"] += 1
                total_tree += 1

        # 6. Boat/Bridge/Structure 直接保留
        for j in range(len(v7_cls)):
            c = int(v7_cls[j])
            if c in (0, 1, 2):
                x1, y1, x2, y2 = [float(v) for v in v7_xyxy[j]]
                yolo_lines.append(f"{c} {rect_to_yolo(x1, y1, x2, y2, w, h)}")
                cnt[NAMES[c]] += 1

        # 写入标签 + 复制图片
        if yolo_lines:
            stem = img_path.stem
            (OUT_LABELS / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            dst = OUT_IMAGES / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            tagged += 1
        else:
            dst = OUT_IMAGES / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(images)}] tagged {tagged}, WH={total_wh}, tree={total_tree}")

    # ── 统计 + 配置 ────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Images: {len(images)}, Tagged: {tagged}")
    for name in NAMES:
        print(f"  {name}: {cnt[name]}")
    print(f"  (GLI threshold: {GLI_THRESHOLD})")

    config = f"""# 水葫芦 GLI(>{GLI_THRESHOLD}) + V7 融合数据集
path: datasets/hyacinth8
train: images/train
val: images/train

names:
  0: Boat
  1: Bridge
  2: Structure
  3: Water Hyacinth
  4: tree
"""
    CONFIG_PATH.write_text(config, encoding="utf-8")
    print(f"\nConfig: {CONFIG_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
