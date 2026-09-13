"""
生产级双 pipeline 推理脚本

两套模式:
1. yolo+sam : YOLO 检测 + SAM 精修分割 (推荐，边界精细)
2. yolo+gli : YOLO 检测非 WH 类 + GLI 植被指数生成 WH mask (轻量，无 SAM 依赖)

用法:
    python scripts/pipelines/inference.py --mode sam --input data/xxx.jpg --output out.jpg
    python scripts/pipelines/inference.py --mode gli --input data/xxx.jpg --output out.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segment_anything import SamPredictor, sam_model_registry
from shuihulu_yolo_sam.api import SegResult, _draw_vis, _to_bgr_numpy, _to_rgb_numpy


class DualPipeline:
    def __init__(
        self,
        yolo_model_path: str | Path = "runs/hyacinth8_yolo_sam/weights/best.pt",
        sam_checkpoint: str | Path = "weights/sam_vit_h.pth",
        sam_model_type: str = "vit_h",
        device: str | int = 0,
        conf: float = 0.10,
        iou: float = 0.5,
        imgsz: int = 640,
        alpha: float = 0.45,
        target_class: int | str = "Water Hyacinth",
        sam_dilate_px: int = 30,
        sam_nms_iou: float = 0.5,
        gli_wh_overlap: float = 0.25,
    ):
        self.device = f"cuda:{device}" if isinstance(device, int) else str(device)
        if self.device == "cuda:-1":
            self.device = "cpu"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.alpha = alpha
        self.target_class = target_class
        self.sam_dilate_px = sam_dilate_px
        self.sam_nms_iou = sam_nms_iou
        self.gli_wh_overlap = gli_wh_overlap

        self.yolo = YOLO(str(yolo_model_path))
        self.class_names = list(self.yolo.model.names.values())
        if isinstance(target_class, str):
            self.target_index = self.class_names.index(target_class)
        else:
            self.target_index = int(target_class)

        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        self.sam_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sam_dilate_px * 2 + 1, sam_dilate_px * 2 + 1)
        )
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    def segment(self, image, mode: str = "sam") -> SegResult:
        if mode == "sam":
            return self._segment_sam(image)
        elif mode in ("gli", "gli_v2"):
            return self._segment_gli(image, exclude_tree=False, wh_overlap=self.gli_wh_overlap)
        elif mode == "gli_all":
            return self._segment_gli(image, exclude_tree=True, wh_overlap=self.gli_wh_overlap)
        else:
            raise ValueError(f"Unknown mode: {mode}, choose 'sam', 'gli' or 'gli_all'")

    def _yolo_detect(self, image_rgb):
        results = self.yolo(
            source=image_rgb, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, verbose=False,
        )
        det = results[0].boxes
        cls_ids = det.cls.cpu().numpy().astype(int)
        boxes = det.xyxy.cpu().numpy()
        scores = det.conf.cpu().numpy()
        masks_xy = results[0].masks.xy if results[0].masks and hasattr(results[0].masks, "xy") else []
        target_idx = np.where(cls_ids == self.target_index)[0]
        other_idx = np.where(cls_ids != self.target_index)[0]
        return boxes, cls_ids, scores, masks_xy, target_idx, other_idx

    def _segment_sam(self, image):
        image_rgb = _to_rgb_numpy(image)
        image_bgr = _to_bgr_numpy(image)
        h, w = image_rgb.shape[:2]

        boxes, cls_ids, scores, masks_xy, target_idx, other_idx = self._yolo_detect(image_rgb)

        if len(target_idx) == 0:
            vis = _draw_vis(image_rgb, [], alpha=self.alpha)
            return SegResult(
                image_bgr=image_bgr, vis_bgr=vis,
                masks=[], boxes_xyxy=np.empty((0, 4), dtype=int),
                scores=[], class_ids=[], class_names=[], areas_px=[], total_area_px=0,
            )

        self.sam_predictor.set_image(image_rgb)
        raw_masks = []

        for j in target_idx:
            # mask prompt from YOLO mask
            if j < len(masks_xy) and len(masks_xy[j]) >= 3:
                yolo_mask_full = self._poly_to_mask(masks_xy[j], h, w).astype(np.float32) / 255.0
                yolo_mask_input = cv2.resize(yolo_mask_full, (256, 256), interpolation=cv2.INTER_NEAREST)
                M = cv2.moments((yolo_mask_full * 255).astype(np.uint8))
                pts = None
                lbls = None
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pts = np.array([[cx, cy]])
                    lbls = np.array([1])
                m, _, _ = self.sam_predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    mask_input=yolo_mask_input[None, ...],
                    multimask_output=False,
                )
            else:
                m, _, _ = self.sam_predictor.predict(box=boxes[j], multimask_output=False)

            sam_mask = m[0].astype(np.uint8) * 255

            # soft constraint: intersect with dilated YOLO mask
            if j < len(masks_xy) and len(masks_xy[j]) >= 3:
                yolo_mask = self._poly_to_mask(masks_xy[j], h, w)
                yolo_dilated = cv2.dilate(yolo_mask, self.sam_kernel, iterations=1)
                constrained = cv2.bitwise_and(sam_mask, yolo_dilated)
                constrained = cv2.morphologyEx(constrained, cv2.MORPH_OPEN, self.open_kernel)
                raw_masks.append(constrained > 0)
            else:
                raw_masks.append(sam_mask > 0)

        # mask NMS dedup
        masks = self._mask_nms(raw_masks, self.sam_nms_iou)

        mask_boxes = boxes[target_idx].astype(int)
        mask_names = [self.class_names[cls_ids[i]] for i in target_idx]
        sam_scores = [float(scores[i]) for i in target_idx]
        other_boxes = boxes[other_idx] if len(other_idx) else np.empty((0, 4))
        other_names = [self.class_names[cls_ids[i]] for i in other_idx]
        other_scores = [float(scores[i]) for i in other_idx]

        areas_px = [int(m.sum()) for m in masks]
        total_area_px = sum(areas_px)

        vis_bgr = _draw_vis(
            image_rgb, masks,
            mask_boxes=mask_boxes, mask_names=mask_names, mask_scores=sam_scores,
            other_boxes=other_boxes, other_names=other_names, other_scores=other_scores,
            alpha=self.alpha,
        )

        return SegResult(
            image_bgr=image_bgr,
            vis_bgr=vis_bgr,
            masks=masks,
            boxes_xyxy=mask_boxes,
            scores=sam_scores,
            class_ids=[int(cls_ids[i]) for i in target_idx],
            class_names=mask_names,
            areas_px=areas_px,
            total_area_px=total_area_px,
        )

    def _segment_gli(self, image, exclude_tree: bool = True, wh_overlap: float = 0.25):
        """GLI Otsu + YOLO-WH 交集约束 pipeline

        1. YOLO 检测所有类
        2. GLI Otsu 自适应阈值 → 植被候选区域
        3. 排除 YOLO 非 WH 类 (Boat/Bridge/Structure/tree)
        4. 形态学闭运算合并碎片
        5. YOLO-WH 交集约束: GLI 连通域与 YOLO-WH 重叠 > wh_overlap 才保留
        6. 保留 GLI 自身的轮廓范围
        """
        image_bgr = _to_bgr_numpy(image)
        image_rgb = _to_rgb_numpy(image)
        h, w = image_bgr.shape[:2]

        boxes, cls_ids, scores, masks_xy, target_idx, other_idx = self._yolo_detect(image_rgb)

        # 1. GLI Otsu 自适应植被掩膜
        rgb = image_bgr.astype(np.float32)
        R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
        gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
        gli_u8 = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
        _, veg = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, self.open_kernel)

        # 2. 排除所有非 WH 类
        exclude_classes = (0, 1, 2, 4) if exclude_tree else (0, 1, 2)
        exclude = np.zeros((h, w), np.uint8)
        for j in range(len(cls_ids)):
            c = int(cls_ids[j])
            if c in exclude_classes:
                if j < len(masks_xy) and len(masks_xy[j]) >= 3:
                    cv2.fillPoly(exclude, [masks_xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = boxes[j].astype(int)
                    cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        # 3. YOLO-WH mask (用于约束)
        yolo_wh = np.zeros((h, w), np.uint8)
        for j in range(len(cls_ids)):
            if int(cls_ids[j]) == 3:
                if j < len(masks_xy) and len(masks_xy[j]) >= 3:
                    cv2.fillPoly(yolo_wh, [masks_xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = boxes[j].astype(int)
                    cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        # 4. GLI 候选 = veg - exclude, 闭运算合并
        wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN, self.open_kernel)
        wh = cv2.morphologyEx(wh, cv2.MORPH_CLOSE, self.merge_kernel, iterations=2)

        # 5. GLI 连通域与 YOLO-WH 交集约束
        cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        mask_boxes = []
        for ct in cs:
            if cv2.contourArea(ct) < 50:
                continue
            ct_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(ct_mask, [ct], -1, 255, -1)
            inter = cv2.countNonZero(cv2.bitwise_and(ct_mask, yolo_wh))
            area = cv2.countNonZero(ct_mask)
            if area > 0 and inter / area > wh_overlap:
                m = ct_mask > 0
                masks.append(m)
                x, y, bw, bh = cv2.boundingRect(ct)
                mask_boxes.append([x, y, x + bw, y + bh])

        mask_boxes = np.array(mask_boxes, dtype=int) if mask_boxes else np.empty((0, 4), dtype=int)
        mask_names = ["Water Hyacinth"] * len(masks)
        other_boxes = boxes[other_idx] if len(other_idx) else np.empty((0, 4))
        other_names = [self.class_names[cls_ids[i]] for i in other_idx]
        other_scores = [float(scores[i]) for i in other_idx]

        areas_px = [int(m.sum()) for m in masks]
        total_area_px = sum(areas_px)

        vis_bgr = _draw_vis(
            image_rgb, masks,
            mask_boxes=mask_boxes, mask_names=mask_names,
            other_boxes=other_boxes, other_names=other_names, other_scores=other_scores,
            alpha=self.alpha,
        )

        return SegResult(
            image_bgr=image_bgr,
            vis_bgr=vis_bgr,
            masks=masks,
            boxes_xyxy=mask_boxes,
            scores=[],
            class_ids=[3] * len(masks),
            class_names=mask_names,
            areas_px=areas_px,
            total_area_px=total_area_px,
        )

    @staticmethod
    def _poly_to_mask(pts, h, w):
        m = np.zeros((h, w), np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        return m

    @staticmethod
    def _mask_iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / (union + 1e-8)

    def _mask_nms(self, masks, iou_thresh):
        if not masks:
            return []
        areas = [m.sum() for m in masks]
        order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
        keep = []
        while order:
            cur = order.pop(0)
            keep.append(cur)
            order = [i for i in order if self._mask_iou(masks[cur], masks[i]) < iou_thresh]
        return [masks[i] for i in keep]


def parse_args():
    parser = argparse.ArgumentParser(description="水葫芦双 pipeline 推理")
    parser.add_argument("--mode", choices=["sam", "gli", "gli_all"], required=True, help="推理模式")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="输出可视化图片路径")
    parser.add_argument("--yolo", default="runs/hyacinth8_yolo_sam/weights/best.pt")
    parser.add_argument("--sam", default="weights/sam_vit_h.pth")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = DualPipeline(
        yolo_model_path=args.yolo,
        sam_checkpoint=args.sam,
        device=args.device,
        conf=args.conf,
    )
    result = pipe.segment(args.input, mode=args.mode)
    result.save_vis(args.output)
    print(result.summary())


if __name__ == "__main__":
    main()
