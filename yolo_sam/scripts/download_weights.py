from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlretrieve


YOLO_GH_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
SAM_WEIGHTS_BASE = "https://dl.fbaipublicfiles.com/segment_anything"

PRESET_MODELS = {
    "yolov8n-seg.pt": f"{YOLO_GH_BASE}/yolov8n-seg.pt",
    "yolov8s-seg.pt": f"{YOLO_GH_BASE}/yolov8s-seg.pt",
    "yolov8m-seg.pt": f"{YOLO_GH_BASE}/yolov8m-seg.pt",
    "yolov8l-seg.pt": f"{YOLO_GH_BASE}/yolov8l-seg.pt",
    "yolov8x-seg.pt": f"{YOLO_GH_BASE}/yolov8x-seg.pt",
    "sam_vit_h.pth": f"{SAM_WEIGHTS_BASE}/sam_vit_h_4b8939.pth",
    "sam_vit_l.pth": f"{SAM_WEIGHTS_BASE}/sam_vit_l_0b3195.pth",
    "sam_vit_b.pth": f"{SAM_WEIGHTS_BASE}/sam_vit_b_01ec64.pth",
}

YOLO_MODELS = {k for k in PRESET_MODELS if k.endswith(".pt") and k.startswith("yolo")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载模型权重。")
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="内置权重名，例如 yolov8n-seg.pt。",
    )
    parser.add_argument(
        "--url",
        default="",
        help="自定义完整下载地址。传入后会优先使用该地址。",
    )
    parser.add_argument(
        "--output-dir",
        default="weights",
        help="权重保存目录。",
    )
    return parser.parse_args()


def resolve_download_url(model_name: str, custom_url: str) -> str:
    if custom_url:
        return custom_url
    if model_name in PRESET_MODELS:
        return PRESET_MODELS[model_name]
    raise ValueError(
        f"未内置权重地址：{model_name}。请改用 --url 传入完整下载地址。"
    )


def download_with_fallback(url: str, output_path: Path) -> None:
    try:
        print(f"正在下载：{url}")
        urlretrieve(url, output_path)
        return
    except Exception as e:
        print(f"下载失败：{e}")

    if output_path.exists():
        output_path.unlink()

    fallback = url.replace("https://github.com", "https://ghgo.xyz")
    if fallback != url:
        print(f"尝试镜像：{fallback}")
        try:
            urlretrieve(fallback, output_path)
            return
        except Exception as e2:
            print(f"镜像下载也失败：{e2}")
            if output_path.exists():
                output_path.unlink()

    raise RuntimeError(
        f"无法下载 {url}，请手动下载后放到 {output_path}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_url = resolve_download_url(args.model, args.url)
    output_path = output_dir / args.model

    if output_path.exists():
        print(f"文件已存在，跳过下载：{output_path}")
        return

    download_with_fallback(download_url, output_path)
    print(f"下载完成：{output_path}")


if __name__ == "__main__":
    main()
