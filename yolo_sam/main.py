from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent  # runs/ weights/ datasets/ 仍在项目根
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import Segmenter, SegResult, build_cli_defaults


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="水葫芦图像切割与像素掩码生成（YOLOv8 + SAM 两阶段流水线）",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="图片路径、图片目录或视频路径。",
    )
    parser.add_argument(
        "--yolo-model",
        default=str(PROJECT_ROOT / "runs" / "segment" / "runs" / "segment" / "hyacinth5_yolo_sam" / "weights" / "best.pt"),
        help="YOLOv8-seg 权重路径。默认指向 5 类训练产物"
        "（runs/segment/runs/segment/hyacinth5_yolo_sam/weights/best.pt）；"
        "训练前可用 weights/yolov8m-seg.pt（COCO 预训练）测试流程。",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=defaults.get("sam_checkpoint", str(PROJECT_ROOT / "weights" / "sam_vit_h.pth")),
        help="SAM 权重路径。",
    )
    parser.add_argument(
        "--sam-model-type",
        default=defaults.get("sam_model_type", "vit_h"),
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 模型类型。",
    )
    parser.add_argument("--conf", type=float, default=defaults.get("conf", 0.25), help="检测置信度阈值。")
    parser.add_argument("--iou", type=float, default=defaults.get("iou", 0.5), help="NMS IoU 阈值。")
    parser.add_argument("--imgsz", type=int, default=defaults.get("imgsz", 640), help="YOLO 推理输入尺寸。")
    parser.add_argument("--device", default=defaults.get("device", 0), help="运行设备（0=cuda:0, cpu）。")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="类别名称列表，如 water_hyacinth。默认从 YOLO 模型自动读取。",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="每米对应像素数。提供后面积单位为 m²，否则为 px。",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "runs" / "segment" / "output"),
        help="输出目录。",
    )
    parser.add_argument("--show", action="store_true", help="显示可视化结果。")
    parser.add_argument(
        "--target-class",
        default="hyacinth",
        help="只把该类（默认 hyacinth）的检测框传给 SAM 做像素掩码；"
        "其余类只画框不分割。可用整数索引或名称。",
    )
    return parser.parse_args()


def process_single(segmenter: Segmenter, image_path: Path, output_dir: Path, args: argparse.Namespace) -> SegResult:
    result = segmenter.segment(str(image_path), pixels_per_meter=args.pixels_per_meter)

    unit = "m²" if args.pixels_per_meter else "px"
    area = result.total_area_px
    if args.pixels_per_meter and args.pixels_per_meter > 0:
        area = result.total_area_px / (args.pixels_per_meter ** 2)

    if result.masks:
        print(f"[{image_path.name}] 检测到 {len(result.masks)} 个目标，覆盖面积 {area:.2f} {unit}")
        for i, (s, a, cn) in enumerate(zip(result.scores, result.areas_px, result.class_names)):
            print(f"  目标 {i}: class={cn}, score={s:.3f}, area={a} px")
    else:
        print(f"[{image_path.name}] 未检测到目标")

    vis_path = output_dir / image_path.name
    result.save_vis(vis_path)
    print(f"  可视化已保存: {vis_path}")

    if result.masks:
        mask_dir = output_dir / "masks"
        result.save_masks(mask_dir, stem=image_path.stem)
        print(f"  掩码已保存: {mask_dir / image_path.stem}_*.png")

    if args.show:
        import cv2
        cv2.imshow("YOLO+SAM", result.vis_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def main() -> None:
    args = parse_args()

    print("正在加载模型...")
    segmenter = Segmenter(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        class_names=args.class_names,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        target_class=args.target_class,
    )
    print("模型加载完成。")

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
        print(f"找到 {len(images)} 张图片")
        for img_path in images:
            process_single(segmenter, img_path, output_dir, args)
    else:
        process_single(segmenter, source, output_dir, args)

    print("完成。")


if __name__ == "__main__":
    main()
