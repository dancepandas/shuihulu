"""
5张图 × 3种方法对比：pure_yolo / yolo+sam / yolo+gli
输出: temp_compare_5x3/
"""
import random
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from scripts.pipelines.inference import DualPipeline

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "temp_compare_5x3"
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SAM_PATH = "weights/sam_vit_h.pth"
SEED = 456
SAMPLE_SIZE = 5


def pure_yolo_result(img_bgr, img_rgb, img_path, model):
    h, w = img_bgr.shape[:2]
    res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
    xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else []
    xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    overlay = img_bgr.copy()
    wh_cnt = 0
    for j in range(len(cls)):
        c = int(cls[j])
        if c != 3:
            continue
        wh_cnt += 1
        x1, y1, x2, y2 = xyxy[j].astype(int)
        if j < len(xy) and len(xy[j]) >= 3:
            cv2.drawContours(overlay, [xy[j].astype(int)], -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(overlay, f"pure YOLO n={wh_cnt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay, wh_cnt


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
    pipe = DualPipeline(yolo_model_path=YOLO_PATH, device="cuda:0", conf=0.10, imgsz=640)

    for img_path in sorted(sample):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # pure yolo
        py_img, py_n = pure_yolo_result(img_bgr, img_rgb, img_path, yolo)
        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_pure_yolo.jpg"), py_img)

        # yolo + sam
        sam_result = pipe.segment(img_path, mode="sam")
        sam_result.save_vis(OUT_DIR / f"{img_path.stem}_yolo_sam.jpg")

        # yolo + gli
        gli_result = pipe.segment(img_path, mode="gli")
        gli_result.save_vis(OUT_DIR / f"{img_path.stem}_yolo_gli.jpg")

        print(f"  {img_path.name}: pure={py_n}, sam={sam_result.summary()['n_objects']}, gli={gli_result.summary()['n_objects']}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
