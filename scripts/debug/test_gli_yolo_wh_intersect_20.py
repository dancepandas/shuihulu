"""
GLI + YOLO-WH 交集约束：GLI 候选区域必须和 YOLO-WH 重叠 > 50% 才保留
输出: temp_gli_yolo_wh_intersect_20/
"""
import random, os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA_DIR = Path("data")
OUT_DIR = Path("temp_gli_yolo_wh_intersect_20")
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
GLI_THRESHOLD = 0.12
MIN_AREA = 50
SEED = 789
SAMPLE_SIZE = 20


def compute_gli_mask(img):
    rgb = img.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    # Otsu 自适应阈值
    gli_u8 = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
    _, veg = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    return veg


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                imgs.append(p)
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))
    print(f"Sample: {len(sample)}")

    model = YOLO(YOLO_PATH)
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

        # 1. YOLO 排除所有非 WH 类
        exclude = np.zeros((h, w), np.uint8)
        for j in range(len(cls)):
            c = int(cls[j])
            if c in (0, 1, 2, 4):
                if j < len(xy) and len(xy[j]) >= 3:
                    cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = xyxy[j].astype(int)
                    cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        # 2. GLI 候选 WH
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN, k)
        wh = cv2.morphologyEx(wh, cv2.MORPH_CLOSE, merge_k, iterations=2)

        # 3. YOLO-WH mask
        yolo_wh = np.zeros((h, w), np.uint8)
        for j in range(len(cls)):
            if int(cls[j]) == 3:
                if j < len(xy) and len(xy[j]) >= 3:
                    cv2.fillPoly(yolo_wh, [xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = xyxy[j].astype(int)
                    cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        # 4. GLI 候选 ∩ YOLO-WH 约束：每个 GLI 连通域和 YOLO-WH 重叠 > 25% 才保留
        #    最终保留 GLI 本身的范围
        wh_cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept_cs = []
        for ct in wh_cs:
            if cv2.contourArea(ct) < MIN_AREA:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [ct], -1, 255, -1)
            inter = cv2.countNonZero(cv2.bitwise_and(m, yolo_wh))
            area = cv2.countNonZero(m)
            if area > 0 and inter / area > 0.25:
                kept_cs.append(ct)

        # draw
        overlay = img.copy()
        cv2.drawContours(overlay, kept_cs, -1, (0, 255, 0), 2)
        cv2.putText(overlay, f"GLI+YOLO-WH n={len(kept_cs)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), overlay)

        total = sum(int(cv2.contourArea(c)) for c in kept_cs)
        print(f"  {img_path.name}: n={len(kept_cs)}, area={total}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
