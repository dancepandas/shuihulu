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

PALETTE = [
    (0, 200, 0),
    (0, 100, 255),
    (255, 50, 50),
    (255, 180, 0),
    (180, 0, 255),
    (0, 220, 220),
    (220, 0, 140),
    (60, 60, 220),
    (220, 180, 60),
    (60, 220, 100),
]


def draw(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    boxes_xyxy: np.ndarray | None = None,
    scores: list[float] | None = None,
    class_names: list[str] | None = None,
    alpha: float = 0.45,
    box_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    contour_thickness: int = 2,
) -> np.ndarray:
    canvas = image_rgb.copy()

    for i, m in enumerate(masks):
        color = PALETTE[i % len(PALETTE)]

        overlay = canvas.copy()
        overlay[m] = (
            overlay[m].astype(np.float32) * (1 - alpha)
            + np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
        canvas = overlay

        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, contour_thickness)

    if boxes_xyxy is not None:
        if hasattr(boxes_xyxy, "cpu"):
            boxes_np = boxes_xyxy.cpu().numpy()
        else:
            boxes_np = np.asarray(boxes_xyxy)

        for i, box in enumerate(boxes_np):
            color = PALETTE[i % len(PALETTE)]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

            label_parts: list[str] = []
            if class_names and i < len(class_names):
                label_parts.append(class_names[i])
            if scores and i < len(scores):
                label_parts.append(f"{scores[i]:.2f}")
            label = " ".join(label_parts)

            if label:
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                cv2.rectangle(
                    canvas,
                    (x1, y1 - th - baseline - 6),
                    (x1 + tw, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (x1, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )

    return canvas


def parse_args() -> argparse.Namespace:
    defaults = build_cli_defaults()

    parser = argparse.ArgumentParser(
        description="绘制图像分割可视化：底图为原图，叠加检测框 + 分割掩膜 + 轮廓 + 标签。"
    )
    parser.add_argument(
        "--yolo-model", required=True, help="训练后的 YOLOv8-seg 权重路径。"
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
    )
    parser.add_argument("--source", required=True, help="图片路径或目录。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=defaults.get("conf", 0.25))
    parser.add_argument("--iou", type=float, default=defaults.get("iou", 0.5))
    parser.add_argument("--device", default=defaults.get("device", 0))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="掩膜透明度 (0=全透明, 1=全覆盖)。",
    )
    parser.add_argument(
        "--class-name",
        default="",
        help="类别名称，留空则不显示类别标签。",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="不绘制掩膜填充，仅画框和轮廓。",
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="不绘制轮廓线。",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出保存路径（单张图片）或目录（多张图片）。默认保存到 runs/vis 目录。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示结果。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = YoloSamPipeline(
        yolo_model_path=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
    )

    source = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    else:
        images = [source]

    print(f"[INFO] 共 {len(images)} 张图片待处理")

    out_dir = Path(args.output) if args.output else Path("runs/vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image_rgb = pipeline._to_rgb_numpy(str(img_path))
        results, masks, scores = pipeline.predict(
            image=image_rgb, conf=args.conf, iou=args.iou, imgsz=args.imgsz
        )

        if not masks:
            print(f"[INFO] {img_path.name}: 未检测到目标")
            continue

        boxes_xyxy = results[0].boxes.xyxy
        class_names = [args.class_name] * len(masks) if args.class_name else None

        if args.no_mask:
            vis = image_rgb.copy()
        else:
            vis = image_rgb.copy()
            for i, m in enumerate(masks):
                color = PALETTE[i % len(PALETTE)]
                overlay = vis.copy()
                overlay[m] = (
                    overlay[m].astype(np.float32) * (1 - args.alpha)
                    + np.array(color, dtype=np.float32) * args.alpha
                ).astype(np.uint8)
                vis = overlay

        if not args.no_contour:
            for i, m in enumerate(masks):
                color = PALETTE[i % len(PALETTE)]
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, color, 2)

        if boxes_xyxy is not None:
            if hasattr(boxes_xyxy, "cpu"):
                boxes_np = boxes_xyxy.cpu().numpy()
            else:
                boxes_np = np.asarray(boxes_xyxy)
            for i, box in enumerate(boxes_np):
                color = PALETTE[i % len(PALETTE)]
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                label_parts: list[str] = []
                if class_names and i < len(class_names):
                    label_parts.append(class_names[i])
                if scores and i < len(scores):
                    label_parts.append(f"{scores[i]:.2f}")
                label = " ".join(label_parts)

                if label:
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                    )
                    cv2.rectangle(
                        vis, (x1, y1 - th - baseline - 6), (x1 + tw, y1), color, -1
                    )
                    cv2.putText(
                        vis,
                        label,
                        (x1, y1 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis_bgr)
        print(
            f"[INFO] {img_path.name}: {len(masks)} 个目标 -> {save_path}"
        )

        if args.show:
            cv2.imshow("Segmentation Vis", vis_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"[INFO] 结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
