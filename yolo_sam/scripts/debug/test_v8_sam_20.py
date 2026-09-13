"""
用 V8 YOLO + SAM 完整 pipeline 跑 20 张图
输出: temp_v8_sam_20/
"""
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # yolo_sam/
sys.path.insert(0, str(ROOT / "src"))

from shuihulu_yolo_sam import Segmenter

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "temp_v8_sam_20"
SEED = 123


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(20, len(imgs)))
    print(f"Sample: {len(sample)}")

    seg = Segmenter(
        yolo_model_path="runs/segment/runs/segment/hyacinth8_yolo_sam/weights/best.pt",
        sam_checkpoint="weights/sam_vit_h.pth",
        sam_model_type="vit_h",
        device=0,
        conf=0.10,
        iou=0.5,
        imgsz=640,
        alpha=0.45,
        target_class="Water Hyacinth",
    )

    for img_path in sorted(sample):
        try:
            result = seg.segment(str(img_path))
            result.save_vis(OUT_DIR / img_path.name)
            print(f"  {img_path.name}: n={result.summary()['n_objects']}, area={result.summary()['total_area_px']}")
        except Exception as e:
            print(f"  {img_path.name}: ERROR {e}")

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
