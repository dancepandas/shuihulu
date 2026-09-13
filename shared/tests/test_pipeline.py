from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # 项目根
SRC = ROOT / "yolo_sam" / "src"
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

val_dir = ROOT / "datasets/crack_seg/images/val"
imgs = sorted(val_dir.glob("*.jpg"))[:3]

for img_path in imgs:
    results, masks, scores = pipeline.predict(str(img_path), conf=0.25, imgsz=640)
    print(f"{img_path.name}: {len(masks)} masks detected, scores={[round(s,3) for s in scores]}")
    if masks:
        area = pipeline.estimate_area(masks)
        print(f"  Area: {area:.0f} pixels")