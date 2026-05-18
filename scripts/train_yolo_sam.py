from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import build_cli_defaults, load_yolo_for_train


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="训练 YOLOv8-seg 水葫芦检测模型（为 YOLO+SAM 两阶段流水线第一阶段）。"
        "训练策略：COCO 预训练迁移学习，冻结 FPN P3 层，SGD 优化器。"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="COCO 预训练 YOLOv8-seg 权重路径，例如 weights/yolov8n-seg.pt。",
    )
    parser.add_argument(
        "--data",
        default="configs/dataset_hyacinth_seg.yaml",
        help="数据集配置文件路径。",
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument("--project", default=defaults.get("project", "runs/segment"))
    parser.add_argument("--name", default="hyacinth_yolo_sam")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--no-freeze-p3",
        action="store_true",
        help="不冻结 FPN P3 层（默认冻结）。",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="训练结束后使用遗传算法进行超参数搜索优化。",
    )
    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=300,
        help="遗传算法迭代次数（仅 --tune 时生效）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_p3 = not args.no_freeze_p3
    model = load_yolo_for_train(args.model, freeze_p3=freeze_p3)

    if freeze_p3:
        print("[INFO] 已冻结 FPN P3 层权重（迁移学习策略）")

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
        task="segment",
    )

    if args.tune:
        print("[INFO] 启动遗传算法超参数搜索...")
        model.tune(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            iterations=args.tune_iterations,
            optimizer="SGD",
            task="segment",
            device=args.device,
        )


if __name__ == "__main__":
    main()
