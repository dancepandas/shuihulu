"""
水葫芦 V8 一站式训练脚本
用法: python scripts/run_v8.py
"""
import subprocess, sys, zipfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

# ── 配置 ──────────────────────────────────────
MODEL = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_CONFIG = "configs/dataset_hyacinth8_seg.yaml"
EPOCHS = 100
BATCH = 8
IMGSZ = 640
DEVICE = "0"
WORKERS = 4
PROJECT = "runs/segment"
NAME = "hyacinth8_yolo_sam"
# ───────────────────────────────────────────────


def step1_unzip():
    """解压 data/ 下所有 zip"""
    print("[1/3] Unzipping data...")
    data_dir = ROOT / "data"
    zips = sorted(data_dir.glob("*.zip"))
    if not zips:
        print("  no zip files, skip")
        return
    for z in zips:
        print(f"  extracting: {z.name}")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(data_dir)
    imgs = len(list(data_dir.rglob("*.jpg"))) + len(list(data_dir.rglob("*.jpeg"))) + len(list(data_dir.rglob("*.png")))
    print(f"  done. {imgs} images\n")


def step2_pipeline():
    """GLI+V7 伪标签流水线"""
    print("[2/3] Running GLI+V7 pseudo-label pipeline...")
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/gli_v8_pipeline.py")],
        cwd=str(ROOT), check=False
    )
    if result.returncode != 0:
        print("Pipeline failed!")
        sys.exit(1)
    print()


def step3_train():
    """训练 V8"""
    print("[3/3] Training YOLOv8m-seg V8...")
    cmd = [
        sys.executable, str(ROOT / "scripts/train_yolo_sam.py"),
        "--model", MODEL,
        "--data", DATA_CONFIG,
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--batch", str(BATCH),
        "--device", DEVICE,
        "--workers", str(WORKERS),
        "--project", PROJECT,
        "--name", NAME,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print()


def main():
    print("=" * 44)
    print("  Water Hyacinth V8 Training Pipeline")
    print("=" * 44)
    print()
    step1_unzip()
    step2_pipeline()
    step3_train()
    print("=" * 44)
    print(f"  V8 Training Complete!")
    print(f"  Model: {PROJECT}/{NAME}/weights/best.pt")
    print("=" * 44)


if __name__ == "__main__":
    main()
