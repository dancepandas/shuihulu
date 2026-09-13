"""
用训练好的 V8 模型直接推理 20 张图
输出: temp_v8_infer_20/
"""
import cv2, random
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/segment/runs/segment/hyacinth8_yolo_sam/weights/best.pt"
DATA_DIR = Path("data")
OUT_DIR = Path("temp_v8_infer_20")
SEED = 42


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    model = YOLO(MODEL_PATH)

    imgs = []
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            imgs.append(p)
    sample = random.sample(imgs, min(20, len(imgs)))
    print(f"Sample: {len(sample)}")

    for img_path in sorted(sample):
        img = cv2.imread(str(img_path))
        if img is None or img_path.stat().st_size == 0:
            continue

        res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
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
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(overlay, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(overlay, f"V8 | WH={wh_cnt}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), overlay)

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
