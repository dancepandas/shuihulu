"""
对比 YOLO box 和 SAM mask：看 SAM 是否超出 YOLO 框定范围
"""
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # yolo_sam/
sys.path.insert(0, str(ROOT / "src"))

from shuihulu_yolo_sam import Segmenter
from ultralytics import YOLO

IMG_PATH = "data/20260618190000428800/DJI_202606181900_002_20260618190000428800/DJI_20260618190525_0063_V.jpeg"
OUT_DIR = Path("temp_debug_sam_vs_yolo")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    img = cv2.imread(IMG_PATH)
    h, w = img.shape[:2]

    # YOLO boxes
    yolo = YOLO("runs/hyacinth8_yolo_sam/weights/best.pt")
    res = yolo(IMG_PATH, conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int)
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()

    # SAM masks
    seg = Segmenter(
        yolo_model_path="runs/hyacinth8_yolo_sam/weights/best.pt",
        sam_checkpoint="weights/sam_vit_h.pth",
        sam_model_type="vit_h",
        device=0,
        conf=0.10,
        iou=0.5,
        imgsz=640,
        alpha=0.45,
        target_class="Water Hyacinth",
    )
    result = seg.segment(IMG_PATH)

    # 画1: YOLO boxes
    canvas1 = img.copy()
    for j in range(len(cls)):
        if int(cls[j]) != 3:
            continue
        x1, y1, x2, y2 = xyxy[j].astype(int)
        cv2.rectangle(canvas1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas1, f"WH {float(confs[j]):.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(str(OUT_DIR / "01_yolo_boxes.jpg"), canvas1)

    # 画2: SAM masks
    canvas2 = img.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, m in enumerate(result.masks):
        color = colors[i % len(colors)]
        overlay = canvas2.copy()
        overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        canvas2 = overlay
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas2, cs, -1, color, 2)
    cv2.imwrite(str(OUT_DIR / "02_sam_masks.jpg"), canvas2)

    # 画3: 叠加：YOLO box + SAM mask
    canvas3 = img.copy()
    for i, m in enumerate(result.masks):
        color = colors[i % len(colors)]
        overlay = canvas3.copy()
        overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        canvas3 = overlay
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas3, cs, -1, color, 2)
    for j in range(len(cls)):
        if int(cls[j]) != 3:
            continue
        x1, y1, x2, y2 = xyxy[j].astype(int)
        cv2.rectangle(canvas3, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imwrite(str(OUT_DIR / "03_boxes_and_masks.jpg"), canvas3)

    print(f"Saved to {OUT_DIR}/")
    print(f"YOLO WH boxes: {sum(1 for c in cls if int(c) == 3)}")
    print(f"SAM masks: {len(result.masks)}")


if __name__ == "__main__":
    main()
