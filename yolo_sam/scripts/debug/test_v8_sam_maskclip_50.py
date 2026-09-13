"""
用 V8 YOLO + SAM(mask prompt + clip) 跑 50 张图
输出: temp_v8_sam_maskclip_50/
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
OUT_DIR = ROOT / "temp_v8_sam_maskclip_50"
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SAM_PATH = "weights/sam_vit_h.pth"
SEED = 42
SAMPLE_SIZE = 50


def mask_from_poly(xy, h, w):
    m = np.zeros((h, w), np.uint8)
    if len(xy) >= 3:
        cv2.fillPoly(m, [xy.astype(np.int32)], 255)
    return m


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
            print(f"  {img_path.name}: n=0, area=0")
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

            m = m[0].copy()
            # clip to yolo box
            x1, y1, x2, y2 = xyxy[j].astype(int)
            m[:max(0, y1), :] = False
            m[min(h, y2):, :] = False
            m[:, :max(0, x1)] = False
            m[:, min(w, x2):] = False
            masks.append(m)

        # draw
        canvas = img_bgr.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        for i, m in enumerate(masks):
            color = colors[i % len(colors)]
            overlay = canvas.copy()
            overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            canvas = overlay
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cs, -1, color, 2)
        cv2.putText(canvas, f"mask+clip n={len(masks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), canvas)

        total = sum(int(m.sum()) for m in masks)
        print(f"  {img_path.name}: n={len(masks)}, area={total}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
