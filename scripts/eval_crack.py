from pathlib import Path
import sys
import time

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

exts = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in exts)
# Only evaluate 10 images for speed
images = images[:10]
print(f"[INFO] Evaluating {len(images)} validation images")

def compute_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

ious_yolo_sam = []
ious_yolo_only = []
inference_times = []

for idx, img_path in enumerate(images):
    image_rgb = pipeline._to_rgb_numpy(str(img_path))
    h, w = image_rgb.shape[:2]

    lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
    combined_gt = np.zeros((h, w), dtype=bool)
    if lbl_path.exists():
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

    t0 = time.time()
    results, masks, scores = pipeline.predict(image_rgb, conf=0.25, iou=0.5, imgsz=640)
    t1 = time.time()
    inference_times.append(t1 - t0)

    # YOLO+SAM combined mask
    if masks:
        combined_pred_sam = np.zeros((h, w), dtype=bool)
        for m in masks:
            combined_pred_sam |= m
        ious_yolo_sam.append(compute_mask_iou(combined_pred_sam, combined_gt))
    else:
        ious_yolo_sam.append(0.0)

    # YOLO-only mask
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        combined_pred_yolo = np.zeros((h, w), dtype=bool)
        for mask_data in results[0].masks.data:
            m = mask_data.cpu().numpy()
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            combined_pred_yolo |= (m_resized > 0.5)
        ious_yolo_only.append(compute_mask_iou(combined_pred_yolo, combined_gt))
    else:
        ious_yolo_only.append(0.0)

    print(f"  [{idx+1}/{len(images)}] {img_path.name}: YOLO IoU={ious_yolo_only[-1]:.4f}, YOLO+SAM IoU={ious_yolo_sam[-1]:.4f}, time={t1-t0:.1f}s")

n = len(images)
mean_iou_sam = sum(ious_yolo_sam) / n
mean_iou_yolo = sum(ious_yolo_only) / n

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Dataset: {n} validation images (synthetic crack)")
print(f"Model: YOLOv8m-seg (6 epochs) + SAM ViT-H on CPU")
print()
print(f"YOLOv8-seg Only:  Mean IoU = {mean_iou_yolo:.4f}")
print(f"YOLO+SAM:         Mean IoU = {mean_iou_sam:.4f}")
print(f"IoU Change:       {mean_iou_sam - mean_iou_yolo:+.4f}")
print(f"Avg time:         {sum(inference_times)/n:.2f}s/image")
print(f"SAM better:       {sum(1 for s,y in zip(ious_yolo_sam, ious_yolo_only) if s>y)}/{n}")
print(f"YOLO better:      {sum(1 for s,y in zip(ious_yolo_sam, ious_yolo_only) if y>s)}/{n}")

# Save visualizations
save_dir = ROOT / "runs/segment/crack_eval_vis"
save_dir.mkdir(parents=True, exist_ok=True)
for img_path in images[:5]:
    image_rgb = pipeline._to_rgb_numpy(str(img_path))
    results, masks, scores = pipeline.predict(image_rgb, conf=0.25, imgsz=640)
    if masks:
        boxes_xyxy = results[0].boxes.xyxy
        vis = pipeline.visualize(image_rgb, masks, boxes_xyxy, color=(0, 255, 0), alpha=0.4)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_dir / f"{img_path.stem}_yolosam.jpg"), vis_bgr)
print(f"\nVisualizations saved to: {save_dir}")