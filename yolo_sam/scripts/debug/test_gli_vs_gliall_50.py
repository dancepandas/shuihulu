"""
对比 GLI 模式: gli vs gli_all，50 张图，生产级填充可视化
输出: temp_cmp_gli_vs_gliall/
"""
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pipelines.inference import DualPipeline

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "temp_cmp_gli_vs_gliall"
SEED = 42
SAMPLE_SIZE = 50


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(SAMPLE_SIZE, len(imgs)))
    print(f"Sample: {len(sample)}")

    pipe = DualPipeline(
        yolo_model_path="runs/hyacinth7_yolo_sam/weights/best.pt",
        device="cuda:0",
        conf=0.10,
        imgsz=640,
    )

    tree_detected = 0
    diff_count = 0

    for img_path in sorted(sample):
        for mode in ["gli", "gli_all"]:
            result = pipe.segment(str(img_path), mode=mode)
            result.save_vis(OUT_DIR / f"{img_path.stem}_{mode}.jpg")

        s_gli = pipe.segment(str(img_path), mode="gli").summary()
        s_all = pipe.segment(str(img_path), mode="gli_all").summary()

        if s_gli["total_area_px"] != s_all["total_area_px"]:
            diff_count += 1
            print(f"  DIFF {img_path.name}: gli={s_gli['total_area_px']} gli_all={s_all['total_area_px']}")
        else:
            print(f"  SAME {img_path.name}: n={s_gli['n_objects']}, area={s_gli['total_area_px']}")

    print(f"\nTotal: {len(sample)}, different: {diff_count}")
    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
