from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import build_cli_defaults
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="训练 YOLOv8 检测模型（YOLO+SAM 方案第一阶段：检测水葫芦位置，输出边界框）。"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="COCO 预训练 YOLOv8 检测权重路径，例如 weights/yolov8m.pt。",
    )
    parser.add_argument(
        "--data",
        default="configs/dataset_hyacinth_det.yaml",
        help="数据集配置文件路径。",
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument("--project", default=defaults.get("project", "runs/detect"))
    parser.add_argument("--name", default="hyacinth_detect")
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        print("请先下载 YOLOv8 检测权重：")
        print("  python scripts/download_weights.py --model yolov8m.pt")
        return

    model = YOLO(str(model_path))

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer="SGD",
        task="detect",
    )


if __name__ == "__main__":
    main()