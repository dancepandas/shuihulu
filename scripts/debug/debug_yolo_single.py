"""
对单张图输出 YOLO 原始检测框 + mask
用于排查 SAM 前 YOLO 是否已误检
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
IMG_PATH = "data/20260618190000428800/DJI_202606181900_002_20260618190000428800/DJI_20260618190525_0063_V.jpeg"
OUT_DIR = Path("temp_debug_yolo")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    model = YOLO(MODEL_PATH)
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Cannot read {IMG_PATH}")
        return
    h, w = img.shape[:2]

    res = model(IMG_PATH, conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
    xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else []
    xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    overlay = img.copy()
    wh_cnt = 0
    for j in range(len(cls)):
        c = int(cls[j])
        x1, y1, x2, y2 = xyxy[j].astype(int)
        color = {0: (255, 128, 0), 1: (128, 128, 128), 2: (128, 0, 128),
                 3: (0, 255, 0), 4: (0, 0, 255)}.get(c, (255, 255, 255))
        name = {0: "Boat", 1: "Bridge", 2: "Structure", 3: "WH", 4: "tree"}.get(c, str(c))
        if c == 3:
            wh_cnt += 1
        if j < len(xy) and len(xy[j]) >= 3:
            cv2.drawContours(overlay, [xy[j].astype(int)], -1, color, 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{name} {float(r0.boxes.conf[j]):.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(overlay, f"YOLO WH={wh_cnt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_path = OUT_DIR / "DJI_20260618190525_0063_V_yolo.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved: {out_path}")
    print(f"Detections: {len(cls)}, WH={wh_cnt}")
    for j in range(len(cls)):
        c = int(cls[j])
        x1, y1, x2, y2 = xyxy[j].astype(int)
        print(f"  {j}: cls={c} conf={float(r0.boxes.conf[j]):.3f} box=({x1},{y1},{x2},{y2})")


if __name__ == "__main__":
    main()
