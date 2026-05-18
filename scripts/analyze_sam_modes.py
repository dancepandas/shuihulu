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

def compute_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

# Test with multimask_output=True
images = sorted(val_img_dir.glob("*.jpg"))[:5]

print("=== Testing SAM with multimask_output=True ===")
for img_path in images:
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

    results, boxes_xyxy = pipeline.detect(image_rgb, conf=0.25, imgsz=640)
    if boxes_xyxy.numel() == 0:
        continue

    # Test multimask_output=True (3 masks per box, pick best by IoU)
    from segment_anything import SamPredictor
    pipeline.sam_predictor.set_image(image_rgb)

    boxes_np = boxes_xyxy.cpu().numpy()
    combined_pred_multi = np.zeros((h, w), dtype=bool)
    combined_pred_single = np.zeros((h, w), dtype=bool)

    for box in boxes_np:
        # multimask=True: get 3 masks, pick best IoU with GT
        masks_multi, scores_multi, _ = pipeline.sam_predictor.predict(
            box=box, multimask_output=True
        )
        best_idx = np.argmax(scores_multi)
        combined_pred_multi |= masks_multi[best_idx]

        # multimask=False (original)
        mask_single, score_single, _ = pipeline.sam_predictor.predict(
            box=box, multimask_output=False
        )
        combined_pred_single |= mask_single[0]

    iou_multi = compute_mask_iou(combined_pred_multi, combined_gt)
    iou_single = compute_mask_iou(combined_pred_single, combined_gt)

    # YOLO-only
    iou_yolo = 0.0
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        combined_pred_yolo = np.zeros((h, w), dtype=bool)
        for mask_data in results[0].masks.data:
            m = mask_data.cpu().numpy()
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            combined_pred_yolo |= (m_resized > 0.5)
        iou_yolo = compute_mask_iou(combined_pred_yolo, combined_gt)

    print(f"{img_path.name}: YOLO={iou_yolo:.4f}, SAM(single)={iou_single:.4f}, SAM(multi-best)={iou_multi:.4f}")

# Also test: expand bbox by margin before feeding to SAM
print("\n=== Testing SAM with expanded bbox (1.2x margin) ===")
for img_path in images:
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

    results, boxes_xyxy = pipeline.detect(image_rgb, conf=0.25, imgsz=640)
    if boxes_xyxy.numel() == 0:
        continue

    pipeline.sam_predictor.set_image(image_rgb)
    boxes_np = boxes_xyxy.cpu().numpy()

    # Original bbox
    combined_orig = np.zeros((h, w), dtype=bool)
    for box in boxes_np:
        mask, score, _ = pipeline.sam_predictor.predict(box=box, multimask_output=False)
        combined_orig |= mask[0]

    # Expanded bbox (1.2x margin)
    combined_exp = np.zeros((h, w), dtype=bool)
    for box in boxes_np:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        margin = 1.2
        new_bw, new_bh = bw * margin, bh * margin
        exp_box = np.array([
            max(0, cx - new_bw/2),
            max(0, cy - new_bh/2),
            min(w, cx + new_bw/2),
            min(h, cy + new_bh/2)
        ])
        mask, score, _ = pipeline.sam_predictor.predict(box=exp_box, multimask_output=False)
        combined_exp |= mask[0]

    iou_orig = compute_mask_iou(combined_orig, combined_gt)
    iou_exp = compute_mask_iou(combined_exp, combined_gt)

    print(f"{img_path.name}: SAM(orig bbox)={iou_orig:.4f}, SAM(1.2x bbox)={iou_exp:.4f}")