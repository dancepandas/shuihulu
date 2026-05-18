from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Label Studio 导出的 JSON 标注转换为 YOLO 分割格式（txt）。"
    )
    parser.add_argument("--input", required=True, help="Label Studio 导出的 JSON 文件路径。")
    parser.add_argument("--images", required=True, help="原始图片目录路径。")
    parser.add_argument("--output", required=True, help="YOLO 标签输出目录。")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO 类别 ID。")
    parser.add_argument("--class-name", default="water_hyacinth", help="类别名称，用于验证。")
    return parser.parse_args()


def ls_points_to_yolo_polygon(
    points: list[list[float]],
    img_w: int,
    img_h: int,
) -> list[float]:
    coords: list[float] = []
    for pt in points:
        x_pct, y_pct = pt[0], pt[1]
        x_norm = x_pct / 100.0
        y_norm = y_pct / 100.0
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        coords.extend([x_norm, y_norm])
    return coords


def main() -> None:
    args = parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        ls_data = json.load(f)

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for item in ls_data:
        anns = item.get("annotations", [])
        if not anns:
            skipped += 1
            continue

        results = anns[0].get("result", [])
        if not results:
            skipped += 1
            continue

        file_upload = item.get("file_upload", "")
        original_name = file_upload.split("-", 1)[-1] if "-" in file_upload else file_upload

        img_file = None
        for candidate in images_dir.rglob("*"):
            if candidate.name == original_name:
                img_file = candidate
                break

        if img_file is None:
            print(f"  无法找到图片: {original_name}, 跳过")
            skipped += 1
            continue

        img_bgr = cv2.imdecode(
            np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img_bgr is None:
            print(f"  无法读取图片: {original_name}, 跳过")
            skipped += 1
            continue

        img_h, img_w = img_bgr.shape[:2]

        yolo_lines: list[str] = []

        for r in results:
            rtype = r.get("type", "")
            value = r.get("value", {})
            orig_w = r.get("original_width", img_w)
            orig_h = r.get("original_height", img_h)

            if rtype == "polygon":
                points = value.get("points", [])
                if not points:
                    continue
                coords = ls_points_to_yolo_polygon(points, orig_w, orig_h)
                if len(coords) < 6:
                    continue
                line = f"{args.class_id} " + " ".join(f"{c:.6f}" for c in coords)
                yolo_lines.append(line)

            elif rtype == "rectangle":
                x_pct = value.get("x", 0)
                y_pct = value.get("y", 0)
                w_pct = value.get("width", 0)
                h_pct = value.get("height", 0)
                x_center = (x_pct + w_pct / 2) / 100.0
                y_center = (y_pct + h_pct / 2) / 100.0
                w_norm = w_pct / 100.0
                h_norm = h_pct / 100.0
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                line = f"{args.class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines.append(line)

        if yolo_lines:
            stem = original_name.rsplit(".", 1)[0]
            txt_path = output_dir / f"{stem}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            converted += 1
            print(f"  {original_name}: {len(yolo_lines)} 个标注 -> {txt_path.name}")
        else:
            skipped += 1

    print(f"\n完成！转换 {converted} 张，跳过 {skipped} 张")
    print(f"YOLO 标签目录: {output_dir}")


if __name__ == "__main__":
    main()