from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from .model import resolve_class_index

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


@dataclass
class SegResult:
    image_bgr: np.ndarray
    vis_bgr: np.ndarray
    masks: list[np.ndarray] = field(default_factory=list)
    boxes_xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))
    scores: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    areas_px: list[int] = field(default_factory=list)
    total_area_px: int = 0

    def save_vis(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.vis_bgr)

    def save_masks(self, dir_path: str | Path, stem: str = "mask") -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.masks):
            cv2.imwrite(str(d / f"{stem}_{i}.png"), m.astype(np.uint8) * 255)

    def summary(self) -> dict[str, Any]:
        return {
            "n_objects": len(self.masks),
            "class_names": self.class_names,
            "scores": self.scores,
            "areas_px": self.areas_px,
            "total_area_px": self.total_area_px,
        }


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


def _to_bgr_numpy(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"无法读取图像：{image}")
        return bgr
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise ValueError("不支持的图像格式，期望 HWC RGB uint8 数组。")


def _resolve_device(device: str | int) -> str:
    if isinstance(device, int):
        return f"cuda:{device}" if device >= 0 else "cpu"
    if device.lstrip("-").isdigit():
        n = int(device)
        return f"cuda:{n}" if n >= 0 else "cpu"
    return str(device)


def _draw_vis(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    mask_boxes: np.ndarray | None = None,
    mask_names: list[str] | None = None,
    mask_scores: list[float] | None = None,
    other_boxes: np.ndarray | None = None,
    other_names: list[str] | None = None,
    other_scores: list[float] | None = None,
    alpha: float = 0.45,
    box_thickness: int = 2,
    contour_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 1,
) -> np.ndarray:
    """绘制可视化：目标类（水葫芦）叠加掩膜+轮廓+框；其余类仅画框+标签。"""
    canvas = image_rgb.copy()

    def _label_box(x1, y1, x2, y2, color, parts):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)
        label = " ".join(str(p) for p in parts if p is not None)
        if not label:
            return
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(canvas, (x1, y1 - th - baseline - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            canvas, label, (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
            font_thickness, cv2.LINE_AA,
        )

    # 目标类：掩膜 + 轮廓 + 框
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

        if mask_boxes is not None and i < len(mask_boxes):
            x1, y1, x2, y2 = mask_boxes[i].astype(int)
            parts = []
            if mask_names and i < len(mask_names):
                parts.append(mask_names[i])
            if mask_scores and i < len(mask_scores):
                parts.append(f"{mask_scores[i]:.2f}")
            _label_box(x1, y1, x2, y2, color, parts)

    # 非目标类：仅画框 + 标签（体现 YOLO 识别多类，但不送 SAM）
    neutral = (180, 180, 180)
    if other_boxes is not None and len(other_boxes) > 0:
        for i, box in enumerate(other_boxes):
            x1, y1, x2, y2 = box.astype(int)
            parts = []
            if other_names and i < len(other_names):
                parts.append(other_names[i])
            if other_scores and i < len(other_scores):
                parts.append(f"{other_scores[i]:.2f}")
            _label_box(x1, y1, x2, y2, neutral, parts)

    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


class Segmenter:
    """一键图像分割器：传入图片，自动完成 YOLO 检测 + SAM 分割 + 可视化绘制。

    Parameters
    ----------
    yolo_model_path : str or Path
        训练后的 YOLOv8-seg 权重路径。
    sam_checkpoint : str or Path
        SAM 模型权重路径。
    sam_model_type : str
        SAM 模型类型，可选 ``"vit_h"`` / ``"vit_l"`` / ``"vit_b"``。
    device : str or int
        运行设备，如 ``0`` (cuda:0)、``"cpu"``。
    class_names : list[str] or None
        类别名称列表，用于可视化标签。若为 None 则从 YOLO 模型自动读取。
    conf : float
        YOLO 检测置信度阈值。
    iou : float
        YOLO NMS IoU 阈值。
    imgsz : int
        YOLO 推理输入尺寸。
    alpha : float
        可视化掩膜透明度 (0=全透明, 1=全覆盖)。
    """

    def __init__(
        self,
        yolo_model_path: str | Path,
        sam_checkpoint: str | Path,
        sam_model_type: str = "vit_h",
        device: str | int = 0,
        class_names: list[str] | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        alpha: float = 0.45,
        target_class: int | str | None = "hyacinth",
    ):
        self.device = _resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.alpha = alpha

        yolo_path = Path(yolo_model_path)
        if not yolo_path.exists():
            raise FileNotFoundError(f"未找到 YOLOv8 模型文件：{yolo_model_path}")
        self.yolo = YOLO(str(yolo_path))

        sam_path = Path(sam_checkpoint)
        if not sam_path.exists():
            raise FileNotFoundError(
                f"未找到 SAM 权重文件：{sam_checkpoint}。"
                "请先下载：python scripts/download_weights.py --model sam_vit_h.pth"
            )
        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_path))
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        if class_names is not None:
            self.class_names = class_names
        else:
            try:
                self.class_names = list(self.yolo.model.names.values())
            except Exception:
                self.class_names = []

        # 解析「只把该类（默认水葫芦）的框传给 SAM」的目标类索引
        try:
            self.target_index = resolve_class_index(self.class_names, target_class)
        except ValueError as e:
            print(f"[WARN] 目标类解析失败，将不做类别过滤（全部框送 SAM）：{e}")
            self.target_index = None

    def segment(
        self,
        image: str | Path | np.ndarray,
        pixels_per_meter: float | None = None,
    ) -> SegResult:
        """对单张图片执行完整的 YOLO+SAM 分割流程，返回结构化结果。

        Parameters
        ----------
        image : str, Path, or np.ndarray
            图片路径或 HWC RGB uint8 数组。
        pixels_per_meter : float or None
            每米对应像素数。若提供，面积单位为平方米；否则为像素数。

        Returns
        -------
        SegResult
            包含可视化图、掩膜、边界框、置信度、面积等全部信息。
        """
        image_rgb = _to_rgb_numpy(image)
        image_bgr = _to_bgr_numpy(image)

        results = self.yolo(
            source=image_rgb, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, verbose=False,
        )
        det = results[0].boxes
        boxes_xyxy = det.xyxy

        if boxes_xyxy.numel() == 0:
            vis_bgr = _draw_vis(image_rgb, [], alpha=self.alpha)
            return SegResult(
                image_bgr=image_bgr, vis_bgr=vis_bgr,
                masks=[], boxes_xyxy=np.empty((0, 4), dtype=int),
                scores=[], class_ids=[], class_names=[], areas_px=[], total_area_px=0,
            )

        boxes_np = (
            boxes_xyxy.cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor)
            else np.asarray(boxes_xyxy)
        )
        cls_ids = det.cls.cpu().numpy().astype(int)
        det_scores = (
            det.conf.cpu().numpy() if det.conf is not None
            else np.zeros(len(cls_ids))
        )

        def _name(cid: int) -> str:
            return self.class_names[cid] if cid < len(self.class_names) else str(cid)

        # 目标类（默认水葫芦）框 → SAM；其余类仅检测、不分割
        if self.target_index is not None:
            is_target = (cls_ids == self.target_index)
        else:
            is_target = np.ones(len(cls_ids), dtype=bool)
        target_idx = np.where(is_target)[0]
        other_idx = np.where(~is_target)[0]

        masks: list[np.ndarray] = []
        sam_scores: list[float] = []
        if len(target_idx) > 0:
            self.sam_predictor.set_image(image_rgb)
            for i in target_idx:
                m, s, _ = self.sam_predictor.predict(
                    box=boxes_np[i], multimask_output=False,
                )
                masks.append(m[0])
                sam_scores.append(float(s[0]))

        mask_boxes = (
            boxes_np[target_idx] if len(target_idx)
            else np.empty((0, 4), dtype=float)
        )
        mask_names = [_name(int(cls_ids[i])) for i in target_idx]
        other_boxes = (
            boxes_np[other_idx] if len(other_idx)
            else np.empty((0, 4), dtype=float)
        )
        other_names = [_name(int(cls_ids[i])) for i in other_idx]
        other_scores = [float(det_scores[i]) for i in other_idx]

        areas_px = [int(m.sum()) for m in masks]
        total_area_px = sum(areas_px)

        vis_bgr = _draw_vis(
            image_rgb, masks,
            mask_boxes=mask_boxes, mask_names=mask_names, mask_scores=sam_scores,
            other_boxes=other_boxes, other_names=other_names, other_scores=other_scores,
            alpha=self.alpha,
        )

        # SegResult 字段均与「目标类掩膜」对齐；非目标类仅在可视化中体现
        return SegResult(
            image_bgr=image_bgr,
            vis_bgr=vis_bgr,
            masks=masks,
            boxes_xyxy=mask_boxes.astype(int),
            scores=sam_scores,
            class_ids=[int(cls_ids[i]) for i in target_idx],
            class_names=mask_names,
            areas_px=areas_px,
            total_area_px=total_area_px,
        )

    def segment_batch(
        self,
        images: list[str | Path | np.ndarray],
        pixels_per_meter: float | None = None,
    ) -> list[SegResult]:
        """批量分割多张图片。"""
        return [self.segment(img, pixels_per_meter) for img in images]


def segment_image(
    image: str | Path | np.ndarray,
    yolo_model_path: str | Path = "runs/segment/runs/segment/crack_yolo_sam/weights/best.pt",
    sam_checkpoint: str | Path = "weights/sam_vit_h.pth",
    sam_model_type: str = "vit_h",
    device: str | int = 0,
    class_names: list[str] | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    alpha: float = 0.45,
    pixels_per_meter: float | None = None,
    target_class: int | str | None = "hyacinth",
) -> SegResult:
    """一键函数：传入图片路径，返回完整的分割结果。

    示例::

        from shuihulu_yolo_sam import segment_image

        result = segment_image("photo.jpg")
        result.save_vis("output.jpg")
        print(result.summary())

    Parameters
    ----------
    image : str, Path, or np.ndarray
        图片路径或 HWC RGB uint8 数组。
    yolo_model_path : str or Path
        YOLOv8-seg 权重路径。
    sam_checkpoint : str or Path
        SAM 权重路径。
    sam_model_type : str
        SAM 模型类型。
    device : str or int
        运行设备。
    class_names : list[str] or None
        类别名称列表。
    conf : float
        检测置信度阈值。
    iou : float
        NMS IoU 阈值。
    imgsz : int
        推理输入尺寸。
    alpha : float
        掩膜透明度。
    pixels_per_meter : float or None
        每米对应像素数。

    Returns
    -------
    SegResult
    """
    seg = Segmenter(
        yolo_model_path=yolo_model_path,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
        class_names=class_names,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        alpha=alpha,
        target_class=target_class,
    )
    return seg.segment(image, pixels_per_meter=pixels_per_meter)