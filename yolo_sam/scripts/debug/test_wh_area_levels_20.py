"""
水葫芦按面积分4级着色：红>橙>黄>绿
只显示水葫芦，其他类别不画
输出: temp_wh_area_levels_20/
"""
import random
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pipelines.inference import DualPipeline

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "temp_wh_area_levels_20"
SEED = 123
SAMPLE_SIZE = 20

# 面积分级颜色：小->大 绿->黄->橙->红
LEVEL_COLORS = [
    (0, 255, 0),    # 绿 - 小面积
    (0, 255, 255),  # 黄
    (0, 128, 255),  # 橙
    (0, 0, 255),    # 红 - 大面积
]


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))
    print(f"Sample: {len(sample)}")

    pipe = DualPipeline(
        yolo_model_path="runs/hyacinth8_yolo_sam/weights/best.pt",
        device="cuda:0",
        conf=0.10,
        imgsz=640,
    )

    for img_path in sorted(sample):
        result = pipe.segment(str(img_path), mode="gli")

        img_bgr = result.image_bgr.copy()
        masks = result.masks

        if not masks:
            cv2.putText(img_bgr, "no WH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(OUT_DIR / img_path.name), img_bgr)
            print(f"  {img_path.name}: n=0")
            continue

        # 按面积排序并分4级
        areas = [int(m.sum()) for m in masks]
        sorted_idx = sorted(range(len(masks)), key=lambda i: areas[i])
        n = len(masks)
        level_size = max(1, n // 4)

        # 小->大分配颜色
        mask_to_color = {}
        for rank, idx in enumerate(sorted_idx):
            if rank < level_size:
                level = 0  # 绿
            elif rank < 2 * level_size:
                level = 1  # 黄
            elif rank < 3 * level_size:
                level = 2  # 橙
            else:
                level = 3  # 红
            mask_to_color[idx] = LEVEL_COLORS[level]

        # 绘制
        for i, m in enumerate(masks):
            color = mask_to_color[i]
            overlay = img_bgr.copy()
            overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            img_bgr = overlay
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, cs, -1, color, 2)

        # 图例
        legend_y = 25
        legend_items = [
            ("green: small", LEVEL_COLORS[0]),
            ("yellow: medium-small", LEVEL_COLORS[1]),
            ("orange: medium-large", LEVEL_COLORS[2]),
            ("red: large", LEVEL_COLORS[3]),
        ]
        for text, color in legend_items:
            cv2.putText(img_bgr, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            legend_y += 22

        cv2.imwrite(str(OUT_DIR / img_path.name), img_bgr)
        print(f"  {img_path.name}: n={n}, area={sum(areas)}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
