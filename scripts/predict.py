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

    parser = argparse.ArgumentParser(description="运行 YOLOv8-seg 水葫芦实例分割推理。")
    parser.add_argument("--model", required=True, help="训练后的模型权重路径。")
    parser.add_argument("--source", required=True, help="图片、视频、目录或摄像头输入源。")
    parser.add_argument("--imgsz", type=int, default=defaults.get("imgsz", 960))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument("--conf", type=float, default=defaults.get("conf", 0.25))
    parser.add_argument("--iou", type=float, default=defaults.get("iou", 0.5))
    parser.add_argument("--project", default=defaults.get("project", "runs/segment"))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save=args.save,
        task="segment",
    )


if __name__ == "__main__":
    main()
