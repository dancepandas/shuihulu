"""
纯 YOLO 结果按面积分4级着色
输出: temp_wh_area_levels_yolo_20/
"""
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA_DIR = Path("data")
OUT_DIR = Path("temp_wh_area_levels_yolo_20")
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SEED = 123
SAMPLE_SIZE = 20

LEVEL_COLORS = [
    (0, 255, 0),    # 绿 - 小
    (0, 255, 255),  # 黄
    (0, 128, 255),  # 橙
    (0, 0, 255),    # 红 - 大
]


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))
    print(f"Sample: {len(sample)}")

    model = YOLO(YOLO_PATH)

    for img_path in sorted(sample):
        img = cv2.imread(str(img_path))
        if img is None or img_path.stat().st_size == 0:
            continue
        h, w = img.shape[:2]

        res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
        r0 = res[0]
        cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
        xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else []
        xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

        # 只保留 WH
        wh_masks = []
        for j in range(len(cls)):
            if int(cls[j]) == 3:
                if j < len(xy) and len(xy[j]) >= 3:
                    m = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(m, [xy[j].astype(np.int32)], 255)
                    wh_masks.append(m > 0)

        if not wh_masks:
            cv2.putText(img, "no WH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(OUT_DIR / img_path.name), img)
            print(f"  {img_path.name}: n=0")
            continue

        # 按面积分级
        areas = [int(m.sum()) for m in wh_masks]
        sorted_idx = sorted(range(len(wh_masks)), key=lambda i: areas[i])
        n = len(wh_masks)
        level_size = max(1, n // 4)

        mask_to_color = {}
        for rank, idx in enumerate(sorted_idx):
            if rank < level_size:
                level = 0
            elif rank < 2 * level_size:
                level = 1
            elif rank < 3 * level_size:
                level = 2
            else:
                level = 3
            mask_to_color[idx] = LEVEL_COLORS[level]

        # draw
        canvas = img.copy()
        for i, m in enumerate(wh_masks):
            color = mask_to_color[i]
            overlay = canvas.copy()
            overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            canvas = overlay
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cs, -1, color, 2)

        legend_y = 25
        for text, color in [("green: small", LEVEL_COLORS[0]),
                            ("yellow: medium-small", LEVEL_COLORS[1]),
                            ("orange: medium-large", LEVEL_COLORS[2]),
                            ("red: large", LEVEL_COLORS[3])]:
            cv2.putText(canvas, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            legend_y += 22

        cv2.imwrite(str(OUT_DIR / img_path.name), canvas)
        print(f"  {img_path.name}: n={n}, area={sum(areas)}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
