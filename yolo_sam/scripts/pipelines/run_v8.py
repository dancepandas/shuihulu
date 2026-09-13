"""
水葫芦 V8 一站式训练脚本
用法: python scripts/run_v8.py

前提: 图片已放在 datasets/hyacinth8/images/train/

步骤:
  1. GLI(>0.12) + V7 伪标签流水线 → labels/train/
  2. YOLOv8m-seg 训练
"""
import subprocess, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]  # yolo_sam/
PROJECT_ROOT = ROOT.parent  # runs/ datasets/ 所在的项目根

MODEL = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_CONFIG = "yolo_sam/configs/dataset_hyacinth8_seg.yaml"
EPOCHS = 100
BATCH = 8
IMGSZ = 640
DEVICE = "0"
WORKERS = 4
PROJECT = "runs/segment"
NAME = "hyacinth8_yolo_sam"


def step1_pipeline():
    print("[1/2] Running GLI+V7 pseudo-label pipeline...")
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/gli_v8_pipeline.py")],
        cwd=str(PROJECT_ROOT), check=False,
    )
    if result.returncode != 0:
        print("Pipeline failed!")
        sys.exit(1)
    print()


def step2_train():
    print("[2/2] Training YOLOv8m-seg V8...")
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
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print()


def main():
    print("=" * 44)
    print("  Water Hyacinth V8 Training Pipeline")
    print("=" * 44)
    print()
    step1_pipeline()
    step2_train()
    print("=" * 44)
    print(f"  V8 Training Complete!")
    print(f"  Model: {PROJECT}/{NAME}/weights/best.pt")
    print("=" * 44)


if __name__ == "__main__":
    main()
