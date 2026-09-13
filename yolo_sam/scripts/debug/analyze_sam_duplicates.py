"""
统计 V8+SAM 软约束结果的 mask 重复情况
"""
import random
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # yolo_sam/
sys.path.insert(0, str(ROOT / "src"))

from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

DATA_DIR = ROOT / "data"
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SAM_PATH = "weights/sam_vit_h.pth"
SEED = 42
SAMPLE_SIZE = 50
DILATE_PX = 30
IOU_THRESHOLDS = [0.3, 0.5, 0.7]


def mask_from_poly(xy, h, w):
    m = np.zeros((h, w), np.uint8)
    if len(xy) >= 3:
        cv2.fillPoly(m, [xy.astype(np.int32)], 255)
    return m


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)


def main():
    random.seed(SEED)
    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))

    yolo = YOLO(YOLO_PATH)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1))

    total_instances = 0
    duplicate_pairs = {t: 0 for t in IOU_THRESHOLDS}
    duplicate_instances = {t: 0 for t in IOU_THRESHOLDS}
    image_dup_counts = []

    for img_path in sorted(sample):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        res = yolo(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
        r0 = res[0]
        cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
        xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else np.zeros((0, 4))
        xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []
        target_idx = [j for j in range(len(cls)) if int(cls[j]) == 3]

        if len(target_idx) == 0:
            continue

        predictor.set_image(img_rgb)
        masks = []

        for j in target_idx:
            if j < len(xy) and len(xy[j]) >= 3:
                yolo_mask_full = mask_from_poly(xy[j], h, w).astype(np.float32) / 255.0
                yolo_mask_input = cv2.resize(yolo_mask_full, (256, 256), interpolation=cv2.INTER_NEAREST)
                M = cv2.moments((yolo_mask_full * 255).astype(np.uint8))
                pts = None
                lbls = None
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pts = np.array([[cx, cy]])
                    lbls = np.array([1])
                m, _, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    mask_input=yolo_mask_input[None, ...],
                    multimask_output=False,
                )
            else:
                m, _, _ = predictor.predict(box=xyxy[j], multimask_output=False)

            sam_mask = m[0].astype(np.uint8) * 255
            if j < len(xy) and len(xy[j]) >= 3:
                yolo_mask = mask_from_poly(xy[j], h, w)
                yolo_dilated = cv2.dilate(yolo_mask, kernel, iterations=1)
                constrained = cv2.bitwise_and(sam_mask, yolo_dilated)
                constrained = cv2.morphologyEx(constrained, cv2.MORPH_OPEN,
                                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                masks.append(constrained > 0)
            else:
                masks.append(sam_mask > 0)

        n = len(masks)
        total_instances += n

        dup_per_img = {t: [] for t in IOU_THRESHOLDS}
        for i in range(n):
            for j in range(i + 1, n):
                iou = mask_iou(masks[i], masks[j])
                for t in IOU_THRESHOLDS:
                    if iou > t:
                        duplicate_pairs[t] += 1
                        dup_per_img[t].append((i, j, iou))

        for t in IOU_THRESHOLDS:
            # 统计涉及重复的 instance 数
            involved = set()
            for i, j, _ in dup_per_img[t]:
                involved.add(i)
                involved.add(j)
            duplicate_instances[t] += len(involved)
            if involved:
                image_dup_counts.append((img_path.name, t, len(involved), len(dup_per_img[t])))

    print(f"Total images: {len(sample)}")
    print(f"Total WH instances: {total_instances}")
    print(f"\nDuplicate pairs by IoU threshold:")
    for t in IOU_THRESHOLDS:
        print(f"  IoU > {t}: {duplicate_pairs[t]} pairs, {duplicate_instances[t]} instances involved")

    print(f"\nImages with duplicates (IoU > 0.5):")
    for name, t, n_inv, n_pair in sorted([x for x in image_dup_counts if x[1] == 0.5], key=lambda x: x[3], reverse=True)[:15]:
        print(f"  {name}: {n_inv} instances, {n_pair} pairs")


if __name__ == "__main__":
    main()
