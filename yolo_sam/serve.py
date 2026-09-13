#!/usr/bin/env python3
"""水葫芦 YOLOv8 + SAM 两阶段推理 HTTP 服务。

YOLOv8 检测 5 类（Boat/Bridge/Structure/Water Hyacinth/tree），
仅 Water Hyacinth 框送 SAM 生成像素掩码，其余类仅返回检测结果。

启动：
    python serve.py --yolo /models/yolov8m-seg.pt --sam /models/sam_vit_h.pth --port 13000
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

# ── add src to path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from shuihulu_yolo_sam import resolve_class_index  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────
def mask_to_base64(mask: np.ndarray) -> str:
    """bool 掩膜 → PNG → base64 字符串."""
    img = (mask.astype(np.uint8) * 255)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def mask_to_contours(mask: np.ndarray) -> list[list[list[int]]]:
    """bool 掩膜 → 轮廓多边形列表."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    return [c.reshape(-1, 2).astype(int).tolist() for c in contours if len(c) >= 3]


# ── predictor ───────────────────────────────────────────────────
class Predictor:
    """加载 YOLOv8 + SAM，提供单图推理."""

    def __init__(
        self,
        yolo_path: str,
        sam_ckpt: str,
        sam_type: str = "vit_h",
        device: str = "cuda",
        target_class: str | int | None = "hyacinth",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ):
        if isinstance(device, str) and device.isdigit():
            device = f"cuda:{device}"
        self.device = device if torch.cuda.is_available() else "cpu"

        self.yolo = YOLO(str(yolo_path))
        self.class_names: list[str] = list(self.yolo.model.names.values())
        self.target_index = resolve_class_index(self.class_names, target_class)

        sam = sam_model_registry[sam_type](checkpoint=str(sam_ckpt))
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("无法解码图片，请确认上传了 JPG/PNG/BMP 格式的文件。")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def predict(self, image_rgb: np.ndarray) -> dict:
        """返回全量检测结果 + 水葫芦 SAM 掩膜."""
        results = self.yolo(
            source=image_rgb, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, verbose=False,
        )
        det = results[0].boxes
        boxes_xyxy = det.xyxy

        if boxes_xyxy.numel() == 0:
            return {"objects": [], "total_area_px": 0}

        boxes_np = boxes_xyxy.cpu().numpy()
        cls_ids = det.cls.cpu().numpy().astype(int)
        confs = det.conf.cpu().numpy()

        # SAM 仅对水葫芦框
        hy_idx = (
            np.where(cls_ids == self.target_index)[0]
            if self.target_index is not None
            else np.arange(len(cls_ids))
        )
        masks: dict[int, tuple[np.ndarray, float]] = {}
        if len(hy_idx) > 0:
            self.sam.set_image(image_rgb)
            for i in hy_idx:
                m, s, _ = self.sam.predict(box=boxes_np[i], multimask_output=False)
                masks[int(i)] = (m[0], float(s[0]))

        objects: list[dict] = []
        total_area = 0
        for i in range(len(boxes_np)):
            cid = int(cls_ids[i])
            cname = (
                self.class_names[cid]
                if cid < len(self.class_names)
                else str(cid)
            )
            x1, y1, x2, y2 = boxes_np[i].astype(int)
            is_hy = (cid == self.target_index) if self.target_index is not None else False

            obj: dict = {
                "class": cname,
                "class_id": cid,
                "confidence": round(float(confs[i]), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "is_water_hyacinth": bool(is_hy),
            }

            if is_hy and i in masks:
                mask_arr, sam_score = masks[i]
                obj["sam_score"] = round(sam_score, 4)
                obj["area_px"] = int(mask_arr.sum())
                obj["mask_base64"] = mask_to_base64(mask_arr)
                obj["mask_contour"] = mask_to_contours(mask_arr)
                total_area += obj["area_px"]
            else:
                obj["sam_score"] = None
                obj["area_px"] = None
                obj["mask_base64"] = None
                obj["mask_contour"] = None

            objects.append(obj)

        return {"objects": objects, "total_area_px": total_area}


# ── Flask app ───────────────────────────────────────────────────
app = Flask(__name__)
predictor: Predictor | None = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """上传图片文件，返回检测 + SAM 掩膜。

    curl -X POST -F image=@photo.jpg http://localhost:13000/api/predict
    """
    if predictor is None:
        return jsonify({"error": "服务未初始化"}), 500

    if "image" not in request.files:
        return jsonify({"error": "请通过 image 字段上传图片文件"}), 400

    file = request.files["image"]
    data = file.read()
    if not data:
        return jsonify({"error": "上传文件为空"}), 400

    try:
        img_rgb = Predictor._decode_image(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = predictor.predict(img_rgb)
    result["filename"] = file.filename
    result["image_size"] = {
        "width": int(img_rgb.shape[1]),
        "height": int(img_rgb.shape[0]),
    }
    return jsonify(result)


@app.route("/api/predict_file", methods=["POST"])
def api_predict_file():
    """指定服务器上的图片路径（/data 挂载），完成推理。

    curl -X POST -H "Content-Type: application/json" \
         -d '{"path":"/data/photo.jpg"}' http://localhost:13000/api/predict_file
    """
    if predictor is None:
        return jsonify({"error": "服务未初始化"}), 500

    body = request.get_json(silent=True) or {}
    image_path = body.get("path", "")
    if not image_path or not os.path.isfile(image_path):
        return jsonify({"error": f"图片路径不存在：{image_path}"}), 400

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": f"无法读取图片：{image_path}"}), 400
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = predictor.predict(img_rgb)
    result["filename"] = os.path.basename(image_path)
    result["image_size"] = {
        "width": int(img_rgb.shape[1]),
        "height": int(img_rgb.shape[0]),
    }
    return jsonify(result)


# ── main ────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="水葫芦 YOLO+SAM 推理服务")
    p.add_argument("--yolo", default="/models/yolov8m-seg.pt", help="YOLO 权重路径")
    p.add_argument("--sam", default="/models/sam_vit_h.pth", help="SAM 权重路径")
    p.add_argument("--sam-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    p.add_argument("--target-class", default="hyacinth", help="目标类（水葫芦）")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--port", type=int, default=13000, help="HTTP 监听端口")
    p.add_argument("--host", default="0.0.0.0", help="HTTP 监听地址")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[INFO] 加载 YOLO: {args.yolo}")
    print(f"[INFO] 加载 SAM:  {args.sam}")
    predictor = Predictor(
        yolo_path=args.yolo,
        sam_ckpt=args.sam,
        sam_type=args.sam_type,
        device=args.device,
        target_class=args.target_class,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    print(f"[INFO] 目标类索引: {predictor.target_index}（{args.target_class}）")
    print(f"[INFO] 可用类别: {predictor.class_names}")
    print(f"[INFO] 设备: {predictor.device}")
    print(f"[INFO] 监听 http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
