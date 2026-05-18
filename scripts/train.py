from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolov10_seg import build_cli_defaults, load_model


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(description="训练 YOLOv8-seg 水葫芦实例分割模型。")
    parser.add_argument("--model", required=True, help="模型权重名或模型配置路径，例如 yolov8n-seg.pt。")
    parser.add_argument("--data", default="configs/dataset_hyacinth_seg.yaml", help="Dataset config file.")
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--imgsz", type=int, default=defaults.get("imgsz", 960))
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument("--project", default=defaults.get("project", "runs/segment"))
    parser.add_argument("--name", default=defaults.get("name", "hyacinth_yolov8_seg"))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 30))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
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
        task="segment",
    )


if __name__ == "__main__":
    main()
