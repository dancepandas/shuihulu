from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TINYSAM_DIR = ROOT / "TinySAM"
if str(TINYSAM_DIR) not in sys.path:
    sys.path.insert(0, str(TINYSAM_DIR))

from tinysam import sam_model_registry, SamHierarchicalMaskGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 TinySAM 自动分割对图片进行预标注，输出 Label Studio 可导入的 JSON 格式。"
    )
    parser.add_argument("--source", required=True, help="图片目录路径。")
    parser.add_argument(
        "--sam-checkpoint",
        default=str(TINYSAM_DIR / "weights" / "tinysam_42.3.pth"),
        help="TinySAM 权重路径。",
    )
    parser.add_argument("--device", default="auto", help="运行设备（auto/cuda/cpu/0）。")
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="每边采样点数，越大越精细但越慢。",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="最小掩膜面积（像素），过滤小碎片。",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=50,
        help="每张图片最多保留的掩膜数量（按面积排序取最大的）。",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs" / "prelabel_tinysam"),
        help="输出目录。",
    )
    parser.add_argument(
        "--class-name",
        default="water_hyacinth",
        help="标注类别名称。",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.lstrip("-").isdigit():
        n = int(device)
        return f"cuda:{n}" if n >= 0 else "cpu"
    return device


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return []
    polygon = largest.squeeze(1).tolist()
    return polygon


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)

    print(f"使用设备: {device_str}")
    print("正在加载 TinySAM 模型...")
    sam = sam_model_registry["vit_t"](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device_str)
    sam.eval()

    mask_generator = SamHierarchicalMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=args.min_area,
    )
    print("模型加载完成。")

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    mask_dir = output_dir / "masks"
    vis_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"找到 {len(images)} 张图片")

    ls_items: list[dict] = []
    total_masks = 0

    for img_idx, img_path in enumerate(images):
        print(f"[{img_idx + 1}/{len(images)}] 处理: {img_path.name} ...")

        image_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("  无法读取，跳过")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        masks_data = mask_generator.hierarchical_generate(image_rgb)
        print(f"  TinySAM 生成 {len(masks_data)} 个掩膜")

        masks_data = sorted(masks_data, key=lambda m: m["area"], reverse=True)
        masks_data = masks_data[:args.max_masks]
        print(f"  保留 {len(masks_data)} 个掩膜（按面积取前 {args.max_masks}）")

        overlay = image_rgb.copy()
        np.random.seed(42)
        annotations = []

        for i, m in enumerate(masks_data):
            seg = m["segmentation"]
            color = np.random.randint(0, 255, 3).tolist()
            overlay[seg] = (
                overlay[seg].astype(np.float32) * 0.5
                + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

            polygon = mask_to_polygon(seg)
            if not polygon:
                continue

            points: list[dict] = []
            for x, y in polygon:
                points.append({"x": x / w * 100, "y": y / h * 100})

            annotations.append(
                {
                    "type": "polygon",
                    "value": {
                        "points": points,
                        "polygonlabels": [args.class_name],
                    },
                    "to_name": "image",
                    "from_name": "label",
                    "image_rotation": 0,
                    "original_width": w,
                    "original_height": h,
                }
            )

            cv2.imwrite(
                str(mask_dir / f"{img_path.stem}_mask{i}.png"),
                seg.astype(np.uint8) * 255,
            )

        vis_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_dir / img_path.name), vis_bgr)

        rel_path = str(img_path.relative_to(source_dir)).replace("\\", "/")
        ls_items.append(
            {
                "data": {"image": f"/data/local-files/?d={rel_path}"},
                "predictions": [{"result": annotations}],
            }
        )
        total_masks += len(annotations)
        print(f"  输出 {len(annotations)} 个多边形标注")

    json_path = output_dir / "prelabel_tinysam.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ls_items, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共 {len(images)} 张图片，{total_masks} 个预标注多边形")
    print(f"Label Studio 导入文件: {json_path}")
    print(f"可视化结果: {vis_dir}")
    print(f"掩码文件: {mask_dir}")


if __name__ == "__main__":
    main()
