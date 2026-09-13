"""
独立测试环境 — ONNX CPU + GLI Pipeline 综合评测
依赖: onnxruntime, opencv-python, numpy (无 ultralytics, 无 torch)

用法: python tests_env/benchmark.py
"""
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "best.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png"}

# ── 配置 ──────────────────────────────────────
CONF = 0.10
IOU = 0.5
IMGSZ = 640
GLI_WH_OVERLAP = 0.25
MIN_AREA = 50
MERGE_KSIZE = 25
# ───────────────────────────────────────────────


def preprocess(img_bgr, target_size=640):
    """YOLO 预处理: resize + normalize + BCHW"""
    h0, w0 = img_bgr.shape[:2]
    r = target_size / max(h0, w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    img = cv2.resize(img_bgr, (new_w, new_h))
    # letterbox pad
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dy, dx = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = img
    # RGB, normalize, BCHW
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return tensor, (h0, w0), (dy, dx, new_h, new_w, r)


def postprocess(outputs, orig_shape, pad_info, conf_thresh=0.10, iou_thresh=0.5):
    """YOLO 后处理: 提取 detections 和 masks"""
    det_out = outputs[0]   # (1, 41, 8400)
    mask_out = outputs[1]  # (1, 32, 160, 160)
    h0, w0 = orig_shape
    dy, dx, new_h, new_w, r = pad_info

    det = det_out[0].T  # (8400, 41)
    boxes, scores, class_ids, mask_coeffs = [], [], [], []
    nc = 5  # 5 classes

    for i in range(len(det)):
        d = det[i]
        box = d[:4]
        cls_conf = d[4:4 + nc]
        cls_id = int(np.argmax(cls_conf))
        score = float(cls_conf[cls_id])
        if score < conf_thresh:
            continue

        # box decode (cxcywh → xyxy)
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r
        y2 = (cy + bh / 2 - dy) / r
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(score)
        class_ids.append(cls_id)
        mask_coeffs.append(d[4 + nc:])  # 32 mask coefficients

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 32))

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    mask_coeffs = np.array(mask_coeffs)

    # NMS
    keep = []
    order = scores.argsort()[::-1]
    while len(order) > 0:
        cur = order[0]
        keep.append(cur)
        if len(order) == 1:
            break
        others = order[1:]
        x1 = np.maximum(boxes[cur, 0], boxes[others, 0])
        y1 = np.maximum(boxes[cur, 1], boxes[others, 1])
        x2 = np.minimum(boxes[cur, 2], boxes[others, 2])
        y2 = np.minimum(boxes[cur, 3], boxes[others, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes[cur, 2] - boxes[cur, 0]) * (boxes[cur, 3] - boxes[cur, 1])
        area2 = (boxes[others, 2] - boxes[others, 0]) * (boxes[others, 3] - boxes[others, 1])
        union = area1 + area2 - inter
        iou_vals = inter / (union + 1e-8)
        order = others[iou_vals < iou_thresh]

    keep = np.array(keep)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]
    return boxes, scores, class_ids, mask_coeffs, mask_out[0]


def compute_gli_mask(img_bgr):
    """GLI Otsu 植被掩膜"""
    rgb = img_bgr.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    gli_u8 = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
    _, veg = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    return veg


def get_yolo_wh_mask(boxes, class_ids, h, w):
    """从 YOLO 检测结果提取 WH mask（只用 box，无 mask coeff）"""
    yolo_wh = np.zeros((h, w), np.uint8)
    for j in range(len(class_ids)):
        if class_ids[j] == 3:  # WH
            x1, y1, x2, y2 = boxes[j].astype(int)
            cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)),
                          (min(w, x2), min(h, y2)), 255, -1)
    return yolo_wh


def get_exclude_mask(boxes, class_ids, h, w, exclude_tree=True):
    """排除所有非 WH 类的 mask"""
    exclude = np.zeros((h, w), np.uint8)
    exclude_classes = (0, 1, 2, 4) if exclude_tree else (0, 1, 2)
    for j in range(len(class_ids)):
        if class_ids[j] in exclude_classes:
            x1, y1, x2, y2 = boxes[j].astype(int)
            cv2.rectangle(exclude, (max(0, x1), max(0, y1)),
                          (min(w, x2), min(h, y2)), 255, -1)
    return exclude


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()

    # ── 加载 ONNX ────────────────────────────
    print(f"Loading ONNX model: {MODEL_PATH}")
    t0 = time.time()
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    load_time = time.time() - t0
    print(f"  load: {load_time:.2f}s")
    print(f"  providers: {session.get_providers()}")
    print(f"  inputs: {[i.name for i in session.get_inputs()]}")
    print(f"  outputs: {[o.name for o in session.get_outputs()]}")

    # ── 收集图片 ─────────────────────────────
    images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXT])
    print(f"\nTest images: {len(images)}")

    # ── 统计 ─────────────────────────────────
    stats = {
        "env": {
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "model": str(MODEL_PATH),
        },
        "config": {
            "conf": CONF, "iou": IOU, "imgsz": IMGSZ,
            "gli_wh_overlap": GLI_WH_OVERLAP,
            "min_area": MIN_AREA, "merge_ksize": MERGE_KSIZE,
        },
        "per_image": [],
    }

    total_wh = 0
    total_area = 0
    all_yolo_times = []
    all_gli_times = []
    all_total_times = []
    det_counts = {"Boat": 0, "Bridge": 0, "Structure": 0, "WH": 0, "tree": 0}

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # YOLO 推理
        t1 = time.time()
        tensor, orig_shape, pad_info = preprocess(img_bgr, IMGSZ)
        outputs = session.run(None, {input_name: tensor})
        yolo_time = time.time() - t1

        boxes, scores, class_ids, mask_coeffs, mask_proto = postprocess(
            outputs, orig_shape, pad_info, CONF, IOU)

        # 统计各类别
        for cid in class_ids:
            name = {0: "Boat", 1: "Bridge", 2: "Structure", 3: "WH", 4: "tree"}[int(cid)]
            det_counts[name] += 1

        # GLI pipeline
        t2 = time.time()
        veg = compute_gli_mask(img_bgr)
        exclude = get_exclude_mask(boxes, class_ids, h, w, exclude_tree=False)
        yolo_wh = get_yolo_wh_mask(boxes, class_ids, h, w)

        wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_KSIZE, MERGE_KSIZE))
        wh = cv2.morphologyEx(wh, cv2.MORPH_CLOSE, merge_k, iterations=2)

        cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = 0
        kept_area = 0
        kept_cs = []
        for ct in cs:
            if cv2.contourArea(ct) < MIN_AREA:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [ct], -1, 255, -1)
            inter = cv2.countNonZero(cv2.bitwise_and(m, yolo_wh))
            area = cv2.countNonZero(m)
            if area > 0 and inter / area > GLI_WH_OVERLAP:
                kept += 1
                kept_area += int(area)
                kept_cs.append(ct)
        gli_time = time.time() - t2
        total_time = time.time() - t1

        all_yolo_times.append(yolo_time)
        all_gli_times.append(gli_time)
        all_total_times.append(total_time)
        total_wh += kept
        total_area += kept_area

        stats["per_image"].append({
            "name": img_path.name,
            "size": [h, w],
            "yolo_ms": round(yolo_time * 1000, 1),
            "gli_ms": round(gli_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
            "wh_instances": kept,
            "wh_area_px": kept_area,
            "yolo_detections": {k: sum(1 for c in class_ids if int(c) == i)
                                for i, k in enumerate(["Boat","Bridge","Structure","WH","tree"])},
        })

        # 保存可视化（只画 WH）
        vis = img_bgr.copy()
        cv2.drawContours(vis, kept_cs, -1, (0, 255, 0), 2)
        cv2.putText(vis, f"WH={kept}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), vis)

    # ── 汇总 ─────────────────────────────────
    yt = np.array(all_yolo_times)
    gt = np.array(all_gli_times)
    tt = np.array(all_total_times)

    stats["summary"] = {
        "images": len(images),
        "total_wh_instances": total_wh,
        "total_wh_area_px": total_area,
        "mean_wh_per_image": round(total_wh / len(images), 1),
        "mean_area_per_image": round(total_area / len(images), 1),
        "det_counts": det_counts,
        "yolo_time_ms": {"mean": round(yt.mean() * 1000, 1), "p50": round(np.median(yt) * 1000, 1),
                         "p95": round(np.percentile(yt, 95) * 1000, 1), "min": round(yt.min() * 1000, 1),
                         "max": round(yt.max() * 1000, 1)},
        "gli_time_ms": {"mean": round(gt.mean() * 1000, 1), "p50": round(np.median(gt) * 1000, 1),
                        "p95": round(np.percentile(gt, 95) * 1000, 1), "min": round(gt.min() * 1000, 1),
                        "max": round(gt.max() * 1000, 1)},
        "total_time_ms": {"mean": round(tt.mean() * 1000, 1), "p50": round(np.median(tt) * 1000, 1),
                          "p95": round(np.percentile(tt, 95) * 1000, 1), "min": round(tt.min() * 1000, 1),
                          "max": round(tt.max() * 1000, 1)},
        "throughput_fps": round(1 / tt.mean(), 1),
        "model_load_time_s": round(load_time, 2),
    }

    # 打印
    s = stats["summary"]
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Images: {s['images']}")
    print(f"WH instances: {s['total_wh_instances']} ({s['mean_wh_per_image']}/image)")
    print(f"WH area: {s['total_wh_area_px']} px ({s['mean_area_per_image']}/image)")
    print(f"\nYOLO detections: {s['det_counts']}")
    print(f"\nTiming (per image):")
    print(f"  YOLO inference: {s['yolo_time_ms']['mean']}ms avg (p95={s['yolo_time_ms']['p95']}ms)")
    print(f"  GLI pipeline:   {s['gli_time_ms']['mean']}ms avg (p95={s['gli_time_ms']['p95']}ms)")
    print(f"  Total:          {s['total_time_ms']['mean']}ms avg (p95={s['total_time_ms']['p95']}ms)")
    print(f"  Throughput:     {s['throughput_fps']} fps")
    print(f"  Model load:     {s['model_load_time_s']}s")

    # 保存 JSON
    json_path = OUTPUT_DIR / "benchmark.json"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report: {json_path}")
    print(f"Visualizations: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
