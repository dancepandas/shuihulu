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

    parser = argparse.ArgumentParser(description="验证 YOLOv8-seg 水葫芦实例分割模型。")
    parser.add_argument("--model", required=True, help="训练后的模型权重路径。")
    parser.add_argument("--data", default="configs/dataset_hyacinth_seg.yaml", help="Dataset config file.")
    parser.add_argument("--imgsz", type=int, default=defaults.get("imgsz", 960))
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    parser.add_argument("--device", default=defaults.get("device", 0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        task="segment",
    )


if __name__ == "__main__":
    main()
