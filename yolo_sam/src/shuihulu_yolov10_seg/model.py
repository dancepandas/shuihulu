from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_cli_defaults(runtime_path: str | Path = "yolo_sam/configs/runtime.yaml") -> dict[str, Any]:
    runtime_file = Path(runtime_path)
    if not runtime_file.exists():
        return {}
    return load_yaml(runtime_file)


def load_model(model_path: str | Path) -> YOLO:
    model_file = Path(model_path)
    if model_file.suffix in {".pt", ".yaml", ".yml"} and not model_file.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{model_file}。请先放入你的 YOLOv8-seg 权重或模型配置。"
        )
    return YOLO(str(model_path))
