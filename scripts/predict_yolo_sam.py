from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import YoloSamPipeline, build_cli_defaults


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="YOLOv8 + SAM 两阶段推理：检测水葫芦并精细分割，输出掩膜与面积。"
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="训练后的 YOLOv8-seg 权重路径。",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="weights/sam_vit_h.pth",
        help="SAM 模型权重路径（默认 ViT-H）。",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 模型类型。",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="图片、视频或目录路径。",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=defaults.get("conf", 0.25))
    parser.add_argument("--iou", type=float, default=defaults.get("iou", 0.5))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument(
        "--project",
        default=defaults.get("project", "runs/segment"),
    )
    parser.add_argument("--name", default="predict_yolo_sam")
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="每米对应像素数。提供后面积单位为平方米，否则为像素数。",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存可视化结果到 project/name 目录。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示可视化结果。",
    )
    return parser.parse_args()


def process_image(
    pipeline: YoloSamPipeline,
    source: str | Path,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[float]]:
    """处理单张图片：检测 → 分割 → 面积估算。"""
    source_path = Path(source)

    results, masks, scores = pipeline.predict(
        image=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    if not masks:
        print(f"[INFO] {source_path.name}: 未检测到水葫芦")
        return [], []

    area = pipeline.estimate_area(masks, pixels_per_meter=args.pixels_per_meter)
    unit = "m²" if args.pixels_per_meter else "px"
    print(
        f"[INFO] {source_path.name}: 检测到 {len(masks)} 个水葫芦目标，"
        f"覆盖面积 {area:.2f} {unit}"
    )

    if args.save or args.show:
        image_rgb = pipeline._to_rgb_numpy(str(source_path))
        boxes_xyxy = results[0].boxes.xyxy
        vis = pipeline.visualize(image_rgb, masks, boxes_xyxy)

        if args.save:
            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / source_path.name), vis_bgr)

            mask_dir = save_dir / "masks"
            mask_dir.mkdir(exist_ok=True)
            for i, m in enumerate(masks):
                cv2.imwrite(
                    str(mask_dir / f"{source_path.stem}_mask{i}.png"),
                    (m.astype(np.uint8) * 255),
                )

        if args.show:
            cv2.imshow("YOLO+SAM", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return masks, scores


def process_video(
    pipeline: YoloSamPipeline,
    source: str | Path,
    args: argparse.Namespace,
) -> None:
    """处理视频：逐帧检测 → 分割 → 面积估算 → 写出。"""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}_result.mp4"),
            fourcc,
            fps,
            (w, h),
        )

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results, masks, scores = pipeline.predict(
            image=frame_rgb,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
        )

        if masks:
            area = pipeline.estimate_area(masks, pixels_per_meter=args.pixels_per_meter)
            unit = "m²" if args.pixels_per_meter else "px"
            boxes_xyxy = results[0].boxes.xyxy
            vis = pipeline.visualize(frame_rgb, masks, boxes_xyxy)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            area_text = f"Area: {area:.2f} {unit} | Objects: {len(masks)}"
            cv2.putText(vis_bgr, area_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            vis_bgr = frame_bgr

        if writer:
            writer.write(vis_bgr)
        if args.show:
            cv2.imshow("YOLO+SAM", vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] 已处理 {frame_idx} 帧")

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[INFO] 视频处理完成，共 {frame_idx} 帧")


def main() -> None:
    args = parse_args()

    pipeline = YoloSamPipeline(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
    )

    source = Path(args.source)
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
        print(f"[INFO] 找到 {len(images)} 张图片")
        for img_path in images:
            process_image(pipeline, img_path, args)
    elif source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        process_video(pipeline, source, args)
    else:
        process_image(pipeline, source, args)


if __name__ == "__main__":
    main()
