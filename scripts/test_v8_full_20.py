"""
用完整pipeline规则跑20张图看效果
输出: temp_v8_full_20/
"""
import cv2, os, random, shutil
from pathlib import Path
import numpy as np
from ultralytics import YOLO

V7_PATH = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_DIR = Path("data")
OUT_DIR = Path("temp_v8_full_20")
IMG_EXT = {".jpg", ".jpeg", ".png"}
GLI_THRESHOLD = 0.12
MIN_AREA = 50
TREE_OVERLAP = 0.5
SEED = 42


def compute_gli_mask(img):
    rgb = img.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    _, veg = cv2.threshold(gli, GLI_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    veg = (veg * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    return veg


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    images = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMG_EXT:
                images.append(p)
    sample = random.sample(images, min(20, len(images)))
    print(f"Sample: {len(sample)}")

    model = YOLO(V7_PATH)
    merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    for img_path in sorted(sample):
        img = cv2.imread(str(img_path))
        if img is None or img_path.stat().st_size == 0:
            continue
        h, w = img.shape[:2]

        veg = compute_gli_mask(img)

        res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
        r0 = res[0]
        cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
        xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else np.zeros((0, 4))
        xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

        exclude = np.zeros((h, w), np.uint8)
        tree_pts = []
        for j in range(len(cls)):
            c = int(cls[j])
            if c in (0, 1, 2):
                if j < len(xy) and len(xy[j]) >= 3:
                    cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = xyxy[j].astype(int)
                    cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
            elif c == 4:
                if j < len(xy) and len(xy[j]) >= 3:
                    tree_pts.append(xy[j])

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN, k)
        wh = cv2.morphologyEx(wh, cv2.MORPH_CLOSE, merge_k, iterations=2)
        wh_cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wh_cs = [c for c in wh_cs if cv2.contourArea(c) > MIN_AREA]

        # 统计
        wh_union = np.zeros((h, w), np.uint8)
        for c in wh_cs:
            cv2.fillPoly(wh_union, [c], 255)

        overlay = img.copy()
        cv2.drawContours(overlay, wh_cs, -1, (0, 255, 0), 2)
        for pts in tree_pts:
            tm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
            tarea = cv2.countNonZero(tm)
            inter = cv2.countNonZero(cv2.bitwise_and(tm, wh_union))
            if tarea > 0 and inter / tarea < TREE_OVERLAP:
                cv2.drawContours(overlay, [pts.astype(np.int32)], -1, (0, 0, 255), 2)
        for j in range(len(cls)):
            c = int(cls[j])
            if c in (0, 1, 2):
                x1, y1, x2, y2 = xyxy[j].astype(int)
                color = {0: (255, 128, 0), 1: (128, 128, 128), 2: (128, 0, 128)}.get(c)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        cv2.putText(overlay, f"WH={len(wh_cs)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), overlay)

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
