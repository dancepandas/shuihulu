from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import YoloSamPipeline, build_cli_defaults


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="验证 YOLOv8 + SAM 两阶段水葫芦分割模型的精度。"
        "使用 YOLOv8 检测 + SAM 分割，将融合掩膜与标注掩膜计算 IoU。"
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="训练后的 YOLOv8-seg 权重路径。",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="weights/sam_vit_h.pth",
        help="SAM 模型权重路径。",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 模型类型。",
    )
    parser.add_argument(
        "--data",
        default="configs/dataset_hyacinth_seg.yaml",
        help="数据集配置文件路径。",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=defaults.get("batch", 8))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 4))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument("--conf", type=float, default=defaults.get("conf", 0.25))
    parser.add_argument("--iou", type=float, default=defaults.get("iou", 0.5))
    parser.add_argument(
        "--target-class",
        default="hyacinth",
        help="只评估该类（默认 hyacinth）：仅该类的 GT 多边形与 SAM 掩膜参与 IoU。",
    )
    return parser.parse_args()


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def main() -> None:
    args = parse_args()

    from ultralytics.data.utils import check_det_dataset
    import yaml
    import cv2
    import json

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = Path(data_cfg["path"])
    val_img_dir = dataset_path / data_cfg.get("val", "images/val")
    val_lbl_dir = dataset_path / "labels" / "val"

    if not val_img_dir.exists():
        print(f"[ERROR] 验证集图像目录不存在：{val_img_dir}")
        print("请先准备好数据集并检查 configs/dataset_hyacinth_seg.yaml 中的路径。")
        return

    pipeline = YoloSamPipeline(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        target_class=args.target_class,
    )
    target_idx = pipeline.target_index
    print(f"[INFO] 评估目标类索引：{target_idx}（{args.target_class}）")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in exts)
    print(f"[INFO] 找到 {len(images)} 张验证图片")

    ious: list[float] = []
    for img_path in images:
        image_rgb = pipeline._to_rgb_numpy(str(img_path))
        results, masks, scores = pipeline.predict(
            image=image_rgb,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
        )

        if not masks:
            continue

        combined_pred = np.zeros(image_rgb.shape[:2], dtype=bool)
        for m in masks:
            combined_pred |= m

        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        h, w = image_rgb.shape[:2]
        combined_gt = np.zeros((h, w), dtype=bool)
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                # 只取目标类（默认水葫芦）的 GT 多边形
                if target_idx is not None and int(parts[0]) != target_idx:
                    continue
                polygon = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                polygon[:, 0] *= w
                polygon[:, 1] *= h
                polygon = polygon.astype(np.int32)
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(gt_mask, [polygon], 1)
                combined_gt |= gt_mask.astype(bool)

        iou = compute_mask_iou(combined_pred, combined_gt)
        ious.append(iou)

    if ious:
        mean_iou = sum(ious) / len(ious)
        print(f"\n[RESULT] 验证图片数: {len(images)}, 有效评估: {len(ious)}")
        print(f"[RESULT] Mean IoU: {mean_iou:.4f}")
        print(f"[RESULT] Min IoU:  {min(ious):.4f}")
        print(f"[RESULT] Max IoU:  {max(ious):.4f}")
    else:
        print("[WARN] 没有可评估的结果，请检查数据和模型。")


if __name__ == "__main__":
    main()
