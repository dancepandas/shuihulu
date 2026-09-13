# 水葫芦检测 GLI Pipeline 生产部署文档

## 1. 环境依赖

```bash
pip install onnxruntime opencv-python numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**不需要**: PyTorch, ultralytics, CUDA, SAM

---

## 2. 模型文件

```
runs/hyacinth8_yolo_sam/weights/best.onnx  (104.1 MB)
```

导出参数：`imgsz=640, opset=12`

---

## 3. GLI Pipeline 完整流程

```
输入图片 (H×W×3 BGR)
    │
    ├──[YOLO ONNX 推理]────────────────────────────
    │   输入: 640×640 RGB, BCHW, /255 归一化
    │   输出: boxes(5类), class_ids, scores
    │
    ├── YOLO 输出分三路:
    │   ├── Boat(0) / Bridge(1) / Structure(2) / tree(4) → exclude_mask
    │   ├── Water Hyacinth(3) → yolo_wh_mask
    │   └── 其他类别统计
    │
    ├──[GLI 植被指数]──────────────────────────────
    │   GLI = (2G - R - B) / (2G + R + B)
    │   Otsu 自适应二值化
    │   形态学开运算 5×5
    │   → veg_mask
    │
    ├──[GLI 候选 WH = veg_mask - exclude_mask]─────
    │   形态学闭运算 25×25 ×2 (合并相邻碎片)
    │   findContours → 候选多边形列表
    │
    ├──[YOLO-WH 交集约束]───────────────────────────
    │   对每个候选多边形:
    │     overlap = 与 yolo_wh_mask 的交集面积 / 候选面积
    │     if overlap > 0.25 → 保留 (使用 GLI 轮廓)
    │     else → 丢弃 (绿色水体误检)
    │
    └──[输出]──────────────────────────────────────
        多边形列表, 面积统计, 可视化
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| imgsz | 640 | YOLO 输入分辨率 |
| conf | 0.10 | YOLO 置信度阈值 |
| iou | 0.5 | YOLO NMS IoU |
| gli_otsu | 自适应 | GLI 归一化后 Otsu 自动找阈值 |
| min_area | 50 px | 最小 WH 面积 |
| merge_ksize | 25 | 闭运算核大小 |
| wh_overlap | 0.25 | GLI 候选与 YOLO-WH 的最小重叠比例 |

---

## 4. Python 实现

### 4.1 YOLO ONNX 推理

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

def yolo_detect(img_bgr):
    """YOLO 检测, 返回 boxes[dtype=float32], class_ids, scores"""
    h0, w0 = img_bgr.shape[:2]

    # ---- 预处理 ----
    r = 640 / max(h0, w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    img = cv2.resize(img_bgr, (new_w, new_h))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    dy, dx = (640 - new_h) // 2, (640 - new_w) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = img
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]

    # ---- 推理 ----
    det_out, mask_out = session.run(None, {'images': tensor})
    # det_out: (1, 41, 8400) = 4 box + 5 cls + 32 mask_coeff
    # mask_out: (1, 32, 160, 160)

    # ---- 后处理 ----
    det = det_out[0].T  # (8400, 41)
    boxes, scores, class_ids = [], [], []
    for row in det:
        cls_conf = row[4:9]
        cls_id = int(np.argmax(cls_conf))
        score = float(cls_conf[cls_id])
        if score < 0.10:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = (cx - bw/2 - dx) / r
        y1 = (cy - bh/2 - dy) / r
        x2 = (cx + bw/2 - dx) / r
        y2 = (cy + bh/2 - dy) / r
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(cls_id)

    # ---- NMS ----
    keep = cv2.dnn.NMSBoxes(
        boxes, scores, 0.10, 0.5, top_k=500)
    if isinstance(keep, tuple):
        keep = keep[0][:, 0]
    elif isinstance(keep, list):
        keep = np.array(keep).flatten()
    boxes = np.array(boxes)[keep]
    scores = np.array(scores)[keep]
    class_ids = np.array(class_ids)[keep]
    return boxes.astype(np.float32), class_ids.astype(int), scores.astype(np.float32)
```

### 4.2 GLI 植被掩膜

```python
def compute_gli_mask(img_bgr):
    """Otsu 自适应 GLI 植被掩膜"""
    rgb = img_bgr.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    gli_u8 = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
    _, veg = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)
    return veg
```

### 4.3 WH 提取主函数

```python
def extract_wh(img_bgr, boxes, class_ids, scores):
    """
    完整的 GLI + YOLO-WH 约束 pipeline
    返回: [(mask_bool, area_px), ...] 排序按面积从大到小
    """
    h, w = img_bgr.shape[:2]
    CLASS_NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]

    # 1. GLI 植被掩膜
    veg = compute_gli_mask(img_bgr)

    # 2. 排除非 WH 类 (Boat=0, Bridge=1, Structure=2, tree=4)
    exclude = np.zeros((h, w), np.uint8)
    for j in range(len(class_ids)):
        if class_ids[j] in (0, 1, 2, 4):
            x1, y1, x2, y2 = boxes[j].astype(int)
            cv2.rectangle(exclude, (max(0, x1), max(0, y1)),
                          (min(w, x2), min(h, y2)), 255, -1)

    # 3. YOLO WH mask (用于约束)
    yolo_wh = np.zeros((h, w), np.uint8)
    for j in range(len(class_ids)):
        if class_ids[j] == 3:
            x1, y1, x2, y2 = boxes[j].astype(int)
            cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)),
                          (min(w, x2), min(h, y2)), 255, -1)

    # 4. GLI 候选 WH
    wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
    wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    wh = cv2.morphologyEx(wh, cv2.MORPH_CLOSE, merge_k, iterations=2)

    # 5. findContours + YOLO-WH 交集约束
    cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for ct in cs:
        if cv2.contourArea(ct) < 50:
            continue
        ct_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(ct_mask, [ct], -1, 255, -1)
        inter = cv2.countNonZero(cv2.bitwise_and(ct_mask, yolo_wh))
        area = cv2.countNonZero(ct_mask)
        if area > 0 and inter / area > 0.25:
            results.append({
                "mask": ct_mask > 0,
                "area_px": int(area),
                "contour": ct,
            })

    results.sort(key=lambda x: x["area_px"], reverse=True)
    return results
```

### 4.4 完整推理入口

```python
def process_image(img_path):
    """完整处理单张图片"""
    session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    boxes, class_ids, scores = yolo_detect(session, img_bgr)
    wh_results = extract_wh(img_bgr, boxes, class_ids, scores)

    total_area = sum(r["area_px"] for r in wh_results)
    return {
        "image": img_path,
        "wh_count": len(wh_results),
        "total_area_px": total_area,
        "instances": wh_results,
        "other_detections": {
            "Boat": int((class_ids == 0).sum()),
            "Bridge": int((class_ids == 1).sum()),
            "Structure": int((class_ids == 2).sum()),
            "tree": int((class_ids == 4).sum()),
        },
    }
```

---

## 5. 生产环境性能预估

| 场景 | CPU | 单图耗时 | 吞吐 |
|------|-----|---------|------|
| i7-13700KF (测试) | 16核 | ~670ms | 1.5 fps |
| 服务器 Xeon | 24+核 | ~400-500ms | 2+ fps |
| ARM 边缘设备 | 4-8核 | ~2-3s | 0.3-0.5 fps |

YOLO 推理 ~138ms，GLI 管线 ~511ms（主要是形态学运算和轮廓查找）

---

## 6. 依赖清单

```
# requirements_prod.txt
onnxruntime==1.27.0
opencv-python>=4.10.0
numpy>=1.24.0
```

总计安装体积 ~200MB，无需 GPU 驱动。
