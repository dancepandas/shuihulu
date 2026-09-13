"""
V8+SAM 软约束 + mask NMS 去重
输出: temp_v8_sam_masksoft_nms_50/
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
OUT_DIR = ROOT / "temp_v8_sam_masksoft_nms_50"
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SAM_PATH = "weights/sam_vit_h.pth"
SEED = 42
SAMPLE_SIZE = 50
DILATE_PX = 30
NMS_IOU = 0.5


def mask_from_poly(xy, h, w):
    m = np.zeros((h, w), np.uint8)
    if len(xy) >= 3:
        cv2.fillPoly(m, [xy.astype(np.int32)], 255)
    return m


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)


def nms_masks(masks, iou_thresh=0.5):
    """按面积从大到小，去掉与已保留 mask IoU 过高的"""
    if not masks:
        return []
    areas = [m.sum() for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    keep = []
    while order:
        cur = order.pop(0)
        keep.append(cur)
        order = [i for i in order if mask_iou(masks[cur], masks[i]) < iou_thresh]
    return [masks[i] for i in keep]


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))
    print(f"Sample: {len(sample)}")

    yolo = YOLO(YOLO_PATH)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1))

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
            cv2.imwrite(str(OUT_DIR / img_path.name), img_bgr)
            print(f"  {img_path.name}: before=0 after=0, area=0")
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

        before_n = len(masks)
        masks = nms_masks(masks, NMS_IOU)
        after_n = len(masks)

        # draw
        canvas = img_bgr.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        for i, m in enumerate(masks):
            color = colors[i % len(colors)]
            m = m.astype(bool)
            overlay = canvas.copy()
            overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            canvas = overlay
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cs, -1, color, 2)
        cv2.putText(canvas, f"mask+soft+nms n={after_n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), canvas)

        total = sum(int(m.sum()) for m in masks)
        print(f"  {img_path.name}: before={before_n} after={after_n}, area={total}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
