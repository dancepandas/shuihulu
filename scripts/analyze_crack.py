from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import YoloSamPipeline
import cv2
import numpy as np

pipeline = YoloSamPipeline(
    yolo_model_path=ROOT / "runs/segment/runs/segment/crack_yolo_sam/weights/best.pt",
    sam_checkpoint=ROOT / "weights/sam_vit_h.pth",
    sam_model_type="vit_h",
    device="cpu",
)

val_img_dir = ROOT / "datasets/crack_seg/images/val"
val_lbl_dir = ROOT / "datasets/crack_seg/labels/val"

# Analyze one example in detail
img_path = sorted(val_img_dir.glob("*.jpg"))[1]  # crack_0001
image_rgb = pipeline._to_rgb_numpy(str(img_path))
h, w = image_rgb.shape[:2]

# Load GT
lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
combined_gt = np.zeros((h, w), dtype=bool)
with open(lbl_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        polygon = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        polygon[:, 0] *= w
        polygon[:, 1] *= h
        polygon = polygon.astype(np.int32)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [polygon], 1)
        combined_gt |= gt_mask.astype(bool)

# YOLO detection
results, boxes_xyxy = pipeline.detect(image_rgb, conf=0.25, imgsz=640)
print(f"YOLO detected {len(boxes_xyxy)} boxes")
for i, box in enumerate(boxes_xyxy):
    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
    print(f"  Box {i}: [{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1}")

# YOLO masks
if results[0].masks is not None:
    print(f"YOLO masks: {len(results[0].masks.data)} masks")
    for i, mask_data in enumerate(results[0].masks.data):
        m = mask_data.cpu().numpy()
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m_binary = m_resized > 0.5
        print(f"  YOLO mask {i}: {m_binary.sum()} pixels")

# SAM masks
masks, scores = pipeline.segment(image_rgb, boxes_xyxy)
print(f"SAM masks: {len(masks)} masks")
for i, m in enumerate(masks):
    print(f"  SAM mask {i}: {m.sum()} pixels, score={scores[i]:.3f}")

# Compare areas
gt_pixels = combined_gt.sum()
yolo_pixels = sum(cv2.resize(m.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR).sum() for m in results[0].masks.data) if results[0].masks is not None else 0
sam_pixels = sum(m.sum() for m in masks)

print(f"\nGT area:     {gt_pixels} pixels")
print(f"YOLO area:   {int(yolo_pixels)} pixels")
print(f"SAM area:    {int(sam_pixels)} pixels")

# Save comparison visualization
save_dir = ROOT / "runs/segment/crack_analysis"
save_dir.mkdir(parents=True, exist_ok=True)

# GT visualization
gt_vis = image_rgb.copy()
gt_vis[combined_gt] = (gt_vis[combined_gt].astype(float) * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
cv2.imwrite(str(save_dir / f"{img_path.stem}_gt.jpg"), cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))

# YOLO-only visualization
if results[0].masks is not None:
    yolo_vis = image_rgb.copy()
    for mask_data in results[0].masks.data:
        m = cv2.resize(mask_data.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        yolo_mask = m > 0.5
        yolo_vis[yolo_mask] = (yolo_vis[yolo_mask].astype(float) * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(save_dir / f"{img_path.stem}_yolo.jpg"), cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))

# SAM visualization
if masks:
    sam_vis = image_rgb.copy()
    for m in masks:
        sam_vis[m] = (sam_vis[m].astype(float) * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(save_dir / f"{img_path.stem}_sam.jpg"), cv2.cvtColor(sam_vis, cv2.COLOR_RGB2BGR))

print(f"\nAnalysis saved to: {save_dir}")