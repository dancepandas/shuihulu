from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_cli_defaults(runtime_path: str | Path = "yolo_sam/configs/runtime.yaml") -> dict[str, Any]:
    runtime_file = Path(runtime_path)
    if not runtime_file.exists():
        return {}
    return load_yaml(runtime_file)


def resolve_class_index(
    names: dict | list,
    target: int | str | None = "hyacinth",
) -> int | None:
    """从 YOLO 类别名称表解析「只把该类框传给 SAM」的目标类索引。

    target 为 int 时直接返回；为 str 时按子串（大小写不敏感）匹配，
    兼容 'Water Hyacinth' / 'water_hyacinth' 等写法；为 None 时不过滤（传全部框给 SAM）。
    """
    if target is None:
        return None
    if isinstance(target, int):
        return target
    items = names.items() if isinstance(names, dict) else enumerate(names)
    target_low = str(target).lower()
    matches = [idx for idx, name in items if target_low in str(name).lower()]
    available = list(names.values()) if isinstance(names, dict) else list(names)
    if not matches:
        raise ValueError(
            f"未在类别名称中找到匹配 '{target}' 的类别。可用类别：{available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"目标类 '{target}' 匹配到多个类别索引 {matches}，请用整数索引明确指定。"
        )
    return matches[0]


class YoloSamPipeline:
    """两阶段水葫芦画面切割流水线：YOLOv8 检测 + SAM 精细分割。

    Step 1: YOLOv8-seg 对图像推理，识别水葫芦目标，输出边界框坐标；
    Step 2: 将边界框作为提示输入 SAM，生成水葫芦像素级掩膜；
    Output: 水葫芦精准分割轮廓、覆盖区域掩膜，为面积估算提供基础。
    """

    def __init__(
        self,
        yolo_model_path: str | Path,
        sam_checkpoint: str | Path,
        sam_model_type: str = "vit_h",
        device: str | int = 0,
        target_class: int | str | None = "hyacinth",
    ):
        self.device = self._resolve_device(device)

        if not Path(yolo_model_path).exists():
            raise FileNotFoundError(f"未找到 YOLOv8 模型文件：{yolo_model_path}")
        self.yolo = YOLO(str(yolo_model_path))

        # 解析「只把该类（默认水葫芦）的框传给 SAM」的目标类索引
        try:
            names = self.yolo.model.names
        except Exception:
            names = {}
        try:
            self.target_index = resolve_class_index(names, target_class)
        except ValueError as e:
            print(f"[WARN] 目标类解析失败，将不做类别过滤（全部框送 SAM）：{e}")
            self.target_index = None

        if not Path(sam_checkpoint).exists():
            raise FileNotFoundError(
                f"未找到 SAM 权重文件：{sam_checkpoint}。"
                "请先下载：python scripts/download_weights.py --model sam_vit_h.pth"
            )
        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    @staticmethod
    def _resolve_device(device: str | int) -> str:
        if isinstance(device, int):
            return f"cuda:{device}" if device >= 0 else "cpu"
        return str(device)

    # ------------------------------------------------------------------
    # Step 1: YOLOv8 检测
    # ------------------------------------------------------------------
    def detect(
        self,
        image: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ) -> tuple[Any, torch.Tensor]:
        """YOLOv8 目标检测，返回 (ultralytics Results, 边界框 xyxy tensor)。"""
        results = self.yolo(
            source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )
        boxes_xyxy = results[0].boxes.xyxy
        return results, boxes_xyxy

    # ------------------------------------------------------------------
    # Step 2: SAM 精细分割
    # ------------------------------------------------------------------
    def segment(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: torch.Tensor | np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        """SAM 精细分割，将边界框作为提示输入 SAM，生成像素级掩膜。

        Parameters
        ----------
        image_rgb : np.ndarray
            HWC、RGB 格式的 uint8 图像。
        boxes_xyxy : tensor or ndarray
            (N, 4) 边界框 [x1, y1, x2, y2]。

        Returns
        -------
        masks : list[np.ndarray]
            每个目标对应的 bool 掩膜 (H, W)。
        scores : list[float]
            每个掩膜的置信度。
        """
        self.sam_predictor.set_image(image_rgb)

        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_np = boxes_xyxy.cpu().numpy()
        else:
            boxes_np = np.asarray(boxes_xyxy)

        masks: list[np.ndarray] = []
        scores: list[float] = []
        for box in boxes_np:
            mask, score, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            masks.append(mask[0])
            scores.append(float(score[0]))
        return masks, scores

    # ------------------------------------------------------------------
    # 完整两阶段推理
    # ------------------------------------------------------------------
    def predict(
        self,
        image: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ) -> tuple[Any, list[np.ndarray], list[float]]:
        """完整两阶段推理：YOLOv8 检测 → 仅目标类（默认水葫芦）框送 SAM 分割。"""
        results, boxes_xyxy = self.detect(image, conf=conf, iou=iou, imgsz=imgsz)

        if boxes_xyxy.numel() == 0:
            return results, [], []

        # 只把目标类的框传给 SAM（其余类别仅检测、不分割）
        if self.target_index is not None:
            cls = results[0].boxes.cls.long()
            keep = (cls == self.target_index).nonzero(as_tuple=False).squeeze(1)
            boxes_xyxy = boxes_xyxy[keep]
            if boxes_xyxy.numel() == 0:
                return results, [], []

        image_rgb = self._to_rgb_numpy(image)
        masks, scores = self.segment(image_rgb, boxes_xyxy)
        return results, masks, scores

    # ------------------------------------------------------------------
    # 面积估算
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_area(
        masks: list[np.ndarray],
        pixels_per_meter: float | None = None,
    ) -> float:
        """从掩膜估算水葫芦覆盖面积。

        Parameters
        ----------
        masks : list[np.ndarray]
            bool 掩膜列表。
        pixels_per_meter : float, optional
            每米对应的像素数。若提供，返回平方米；否则返回像素数。

        Returns
        -------
        float
            覆盖面积（像素数 或 平方米）。
        """
        total_pixels = sum(int(m.sum()) for m in masks)
        if pixels_per_meter is not None and pixels_per_meter > 0:
            return total_pixels / (pixels_per_meter ** 2)
        return float(total_pixels)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    @staticmethod
    def visualize(
        image_rgb: np.ndarray,
        masks: list[np.ndarray],
        boxes_xyxy: np.ndarray | torch.Tensor | None = None,
        alpha: float = 0.5,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """在原图上叠加分割掩膜与边界框。"""
        overlay = image_rgb.copy()
        for m in masks:
            overlay[m] = (
                overlay[m].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

        if boxes_xyxy is not None:
            if isinstance(boxes_xyxy, torch.Tensor):
                boxes_np = boxes_xyxy.cpu().numpy()
            else:
                boxes_np = np.asarray(boxes_xyxy)
            for box in boxes_np:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        return overlay

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _to_rgb_numpy(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"无法读取图像：{image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise ValueError("不支持的图像格式，期望 HWC RGB uint8 数组。")


# ------------------------------------------------------------------
# YOLOv8 训练辅助：冻结 FPN P3 层
# ------------------------------------------------------------------
def freeze_fpn_p3(model: YOLO) -> None:
    """冻结 YOLOv8-seg 模型 FPN 中 P3 层权重。

    在 YOLOv8-seg 的 model.model 中，Neck 部分包含对 P3 特征的处理层。
    通过匹配层名称中包含 '10' 或 '13' 的 Neck 层来冻结 P3 对应参数。
    """
    for name, param in model.model.named_parameters():
        layer_idx = name.split(".")[1] if name.startswith("model.") else ""
        if layer_idx in {"10", "13"}:
            param.requires_grad = False


def load_yolo_for_train(
    model_path: str | Path,
    freeze_p3: bool = True,
) -> YOLO:
    """加载 YOLOv8-seg 用于训练（迁移学习 + 可选冻结 P3）。"""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{model_file}。请先下载 COCO 预训练权重。"
        )
    model = YOLO(str(model_file))
    if freeze_p3:
        freeze_fpn_p3(model)
    return model
