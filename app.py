#!/usr/bin/env python
"""模型识别结果测试 — Flask 单文件 Web 推理服务
三种模式: 纯 YOLO / YOLO+GLI 融合 / YOLO+SAM
imgsz: 640 / 1280
"""

import io, base64, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2, numpy as np, torch
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

# ═══════════ 配置 ═══════════
YOLO_MODEL_PATH = r"D:\chengs\9.project\shuihulu\runs\segment\runs\segment\hyacinth9_yolo_sam3\weights\best.pt"
YOLO_MODEL_FALLBACK = r"D:\chengs\9.project\shuihulu\runs\hyacinth8_yolo_sam\weights\best.pt"
SAM_WEIGHTS = r"D:\chengs\9.project\shuihulu\weights\sam_vit_h.pth"
SEGFORMER_PATH = r"D:\chengs\9.project\shuihulu\runs\segformer\segformer_b2_ls+v9"

# SegFormer 推理预处理 (ImageNet 归一化)
SF_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SF_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEGFORMER_PATH = r"D:\chengs\9.project\shuihulu\runs\segformer\segformer_b2_ls+v9"  # SegFormer B2
NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]
# COLORS 不再使用，每个实例随机颜色
MIN_AREA = 50
GLI_THRESHOLD = 0.12
TREE_OVERLAP = 0.5
SEGFORMER_SIZE = 512   # SegFormer 推理分辨率

# ═══════════ 加载模型 ═══════════
yolo_path = YOLO_MODEL_PATH if Path(YOLO_MODEL_PATH).exists() else YOLO_MODEL_FALLBACK
print(f"[init] loading YOLO: {yolo_path}")
yolo_model = YOLO(yolo_path)

sam_segmenter = None
if Path(SAM_WEIGHTS).exists():
    try:
        from shuihulu_yolo_sam import Segmenter
        sam_segmenter = Segmenter(
            yolo_model_path=str(Path(yolo_path).resolve()),
            sam_checkpoint=str(Path(SAM_WEIGHTS).resolve()),
            target_class=None,
        )
        print("[init] SAM segmenter loaded")
    except Exception as e:
        print(f"[init] SAM load failed: {e}")
else:
    print("[init] SAM weights not found, YOLO+SAM mode disabled")

# SegFormer
segformer_model = None
segformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if Path(SEGFORMER_PATH).joinpath("config.json").exists():
    try:
        from transformers import SegformerForSemanticSegmentation
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_PATH)
        segformer_model.to(segformer_device)
        segformer_model.eval()
        print("[init] SegFormer B2 loaded")
    except Exception as e:
        print(f"[init] SegFormer load failed: {e}")
else:
    print(f"[init] SegFormer not found at {SEGFORMER_PATH}, mode disabled")

# ═══════════ GLI Pipeline ═══════════
def compute_gli_mask(img_bgr):
    """GLI 植被指数 → Otsu 自适应二值化 → 开运算"""
    rgb = img_bgr.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    gli_norm = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
    _, veg = cv2.threshold(gli_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)


def yolo_gli_pipeline(img_bgr, imgsz):
    """YOLO + GLI 融合: WH 走 GLI 多边形, 其他走模型"""
    h, w = img_bgr.shape[:2]

    # YOLO 推理
    results = yolo_model(img_bgr, conf=0.10, iou=0.5, imgsz=imgsz, verbose=False, retina_masks=True)
    r0 = results[0]
    boxes = r0.boxes
    cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) > 0 else np.array([])
    confs  = boxes.conf.cpu().numpy() if len(cls_arr) > 0 else np.array([])
    xyxy   = boxes.xyxy.cpu().numpy() if len(cls_arr) > 0 else np.zeros((0, 4))
    xy     = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    # GLI
    veg = compute_gli_mask(img_bgr)

    # 排除非 WH + YOLO-WH 掩膜
    exclude = np.zeros((h, w), np.uint8)
    yolo_wh = np.zeros((h, w), np.uint8)
    tree_pts_list = []
    for j in range(len(cls_arr)):
        c = int(cls_arr[j])
        if c in (0, 1, 2):   # Boat, Bridge, Structure → exclude
            if j < len(xy) and len(xy[j]) >= 3:
                cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
            else:
                x1, y1, x2, y2 = xyxy[j].astype(int)
                cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
        elif c == 3:          # Water Hyacinth → YOLO mask
            if j < len(xy) and len(xy[j]) >= 3:
                cv2.fillPoly(yolo_wh, [xy[j].astype(np.int32)], 255)
            else:
                x1, y1, x2, y2 = xyxy[j].astype(int)
                cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
        elif c == 4:          # tree
            if j < len(xy) and len(xy[j]) >= 3:
                tree_pts_list.append((xy[j], confs[j]))

    # WH = GLI - exclude, 闭运算合并碎片
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    wh_mask = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
    wh_mask = cv2.morphologyEx(wh_mask, cv2.MORPH_OPEN, kernel)
    wh_mask = cv2.morphologyEx(wh_mask, cv2.MORPH_CLOSE, merge_k, iterations=2)

    # 提取 WH 轮廓 (YOLO-WH 交集约束)
    wh_contours, _ = cv2.findContours(wh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wh_polys = []
    wh_union = np.zeros((h, w), np.uint8)
    for ct in wh_contours:
        if cv2.contourArea(ct) < MIN_AREA:
            continue
        ct_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(ct_mask, [ct], -1, 255, -1)
        inter = cv2.countNonZero(cv2.bitwise_and(ct_mask, yolo_wh))
        area = cv2.countNonZero(ct_mask)
        if area > 0 and inter / area > 0.25:
            cv2.fillPoly(wh_union, [ct], 255)
            wh_polys.append(("Water Hyacinth", ct[:, 0, :], -1.0))

    # Tree 重叠判断
    trees = []
    for pts, conf in tree_pts_list:
        tm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
        overlap = cv2.countNonZero(cv2.bitwise_and(tm, wh_union))
        area = cv2.countNonZero(tm)
        if area > 0 and overlap / area < TREE_OVERLAP:
            trees.append(("tree", pts, conf))

    # Boat/Bridge/Structure
    others = []
    for j in range(len(cls_arr)):
        c = int(cls_arr[j])
        if c in (0, 1, 2):
            geom = xy[j] if j < len(xy) and len(xy[j]) >= 3 else xyxy[j]
            others.append((NAMES[c], geom, confs[j]))

    return wh_polys + trees + others


def pure_yolo(img_bgr, imgsz):
    """纯 YOLO 推理, 返回 [(类别名, 几何, 置信度), ...]"""
    h, w = img_bgr.shape[:2]
    results = yolo_model(img_bgr, conf=0.10, iou=0.5, imgsz=imgsz, verbose=False, retina_masks=True)
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    cls_arr = r0.boxes.cls.cpu().numpy().astype(int)
    confs  = r0.boxes.conf.cpu().numpy()
    xyxy   = r0.boxes.xyxy.cpu().numpy()
    xy     = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    dets = []
    for j in range(len(cls_arr)):
        c = int(cls_arr[j])
        if c >= len(NAMES):
            continue
        geom = xy[j] if j < len(xy) and len(xy[j]) >= 3 else xyxy[j]
        dets.append((NAMES[c], geom, confs[j]))
    return dets


def yolo_sam(img_bgr, imgsz):
    """YOLO+SAM 两阶段"""
    if sam_segmenter is None:
        raise RuntimeError("SAM segmenter not loaded")

    result = sam_segmenter.segment(img_bgr)
    dets = []
    for mask, cls_id, score in zip(result.masks, result.class_ids, result.scores):
        name = result.class_names[int(cls_id)] if int(cls_id) < len(result.class_names) else f"cls_{cls_id}"
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 3:
                dets.append((name, largest[:, 0, :], float(score)))
    return dets


def segformer_predict(img_bgr):
    """SegFormer 语义分割 → 每个连通域作为一个检测"""
    h, w = img_bgr.shape[:2]

    # 预处理: RGB → resize 512 → normalize
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SEGFORMER_SIZE, SEGFORMER_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
    tensor = (tensor - torch.from_numpy(SF_MEAN).view(3, 1, 1)) / torch.from_numpy(SF_STD).view(3, 1, 1)
    tensor = tensor.unsqueeze(0).to(segformer_device)  # (1,3,512,512)

    with torch.no_grad():
        logits = segformer_model(pixel_values=tensor).logits  # (1,6,128,128)
        # 上采样回原始尺寸
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        class_map = logits[0].argmax(dim=0).cpu().numpy()  # (H,W) values 0-5

    # 每个非背景类找连通域
    dets = []
    for cls_id in range(1, 6):  # 1=Boat, ..., 5=tree
        binary = (class_map == cls_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        name = NAMES[cls_id - 1]  # Boat=0, Bridge=1, ..., tree=4
        for cnt in contours:
            if len(cnt) < 3: continue
            area = cv2.contourArea(cnt)
            if area < MIN_AREA: continue
            dets.append((name, cnt[:, 0, :], -1.0))  # -1 = no confidence score

    return dets


# ═══════════ 可视化 ═══════════
def draw_detections(img_bgr, detections):
    """实例分割风格: 每个实例填充随机颜色半透明掩膜 + 轮廓 + 标签"""
    vis = img_bgr.copy()
    overlay = vis.copy()

    import random
    for name, geom, conf in detections:
        # 每个实例随机鲜艳颜色
        hue = random.randint(0, 179)
        color_hsv = np.uint8([[[hue, 220, 220]]])
        color = tuple(int(c) for c in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0])

        if isinstance(geom, np.ndarray) and geom.ndim == 2 and geom.shape[1] >= 2:
            pts = geom.astype(np.int32)
            # 填充半透明掩膜
            cv2.fillPoly(overlay, [pts], color)
            # 轮廓线
            cv2.drawContours(vis, [pts], -1, color, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        else:
            x1, y1, x2, y2 = [int(v) for v in geom[:4]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cx, cy = (x1 + x2) // 2, y1 - 10

        label = f"{name}" if conf < 0 else f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(vis, (cx - tw // 2 - 2, cy - th - 4), (cx + tw // 2 + 2, cy), color, -1)
        cv2.putText(vis, label, (cx - tw // 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # 半透明混合
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis


def img_to_b64(img_bgr):
    """BGR numpy 图片 → base64 JPEG data URI"""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def b64_to_img(data_uri):
    """base64 data URI → BGR numpy 图片"""
    _, encoded = data_uri.split(",", 1)
    buf = base64.b64decode(encoded)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ═══════════ Flask App ═══════════
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型识别结果测试</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  background: #f0f2f5; color: #333; min-height: 100vh;
}
.header {
  background: linear-gradient(135deg, #1a73e8, #0d47a1);
  color: #fff; padding: 20px 0; text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
}
.header h1 { font-size: 22px; font-weight: 600; letter-spacing: 1px; }
.header p { font-size: 13px; opacity: .8; margin-top: 4px; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px 16px; }
.card {
  background: #fff; border-radius: 10px; padding: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px;
}
.upload-zone {
  border: 2px dashed #c0c8d4; border-radius: 8px;
  padding: 40px 20px; text-align: center; cursor: pointer;
  transition: border-color .2s, background .2s;
}
.upload-zone:hover, .upload-zone.drag-over {
  border-color: #1a73e8; background: #f4f8ff;
}
.upload-zone.has-image { padding: 12px; }
.upload-zone img { max-height: 240px; border-radius: 4px; }
.upload-zone .placeholder { color: #8899aa; font-size: 14px; }
.upload-zone .placeholder svg { display: block; margin: 0 auto 10px; opacity: .4; }
.options {
  display: flex; flex-wrap: wrap; gap: 20px; align-items: center; margin-top: 16px;
}
.opt-group { display: flex; align-items: center; gap: 8px; }
.opt-group > label { font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }
.btn-group { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid #d0d5dd; }
.btn-group label {
  padding: 7px 14px; font-size: 13px; cursor: pointer;
  background: #fff; color: #555; border-right: 1px solid #d0d5dd;
  transition: background .15s;
}
.btn-group label:last-child { border-right: none; }
.btn-group label:hover { background: #f4f8ff; }
.btn-group input { display: none; }
.btn-group input:checked + span {
  background: #1a73e8; color: #fff; display: block; padding: 7px 14px; font-size: 13px;
}
.method-checks { display: flex; gap: 8px; }
.method-checks label {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 14px; font-size: 13px; cursor: pointer;
  border: 1px solid #d0d5dd; border-radius: 6px;
  background: #fff; color: #555; transition: all .15s; user-select: none;
}
.method-checks label:hover { background: #f4f8ff; }
.method-checks input { accent-color: #1a73e8; width: 15px; height: 15px; }
.method-checks label:has(input:checked) {
  background: #e8f0fe; border-color: #1a73e8; color: #1a73e8; font-weight: 600;
}
.btn-primary {
  padding: 10px 36px; font-size: 15px; font-weight: 600;
  background: #1a73e8; color: #fff; border: none; border-radius: 6px;
  cursor: pointer; transition: background .2s;
  display: flex; align-items: center; gap: 8px;
}
.btn-primary:hover { background: #1557b0; }
.btn-primary:disabled { background: #a0c4f0; cursor: not-allowed; }
.result-area { display: none; margin-top: 20px; }
.result-area.show { display: block; }
.orig-card {
  background: #fff; border-radius: 10px; padding: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; text-align: center;
}
.orig-card h3 { font-size: 13px; font-weight: 600; color: #666; margin-bottom: 8px; }
.orig-card img {
  max-width: 100%; max-height: 400px; border-radius: 6px;
  border: 1px solid #e8ecf0; object-fit: contain; background: #fafafa;
}
.result-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px;
}
.result-col {
  background: #fff; border-radius: 10px; padding: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center;
}
.col-header {
  display: flex; align-items: center; justify-content: center; gap: 10px;
  margin-bottom: 10px; min-height: 28px;
}
.col-header h3 { font-size: 15px; font-weight: 700; color: #1a73e8; }
.result-col img {
  width: 100%; border-radius: 6px; border: 1px solid #e8ecf0;
  max-height: 420px; object-fit: contain; background: #fafafa;
}
.stats {
  margin-top: 10px; font-size: 12px; color: #555;
  display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
}
.stats span { background: #f0f2f5; padding: 3px 8px; border-radius: 4px; }
.stats .total { background: #1a73e8; color: #fff; font-weight: 600; }
.col-footer { margin-top: 12px; }
.btn-download {
  display: inline-block; padding: 7px 20px; font-size: 13px;
  background: #fff; color: #1a73e8; border: 1px solid #1a73e8;
  border-radius: 6px; text-decoration: none; cursor: pointer; transition: all .2s;
}
.btn-download:hover { background: #1a73e8; color: #fff; }
.btn-download.disabled { opacity: .4; pointer-events: none; }
.error-msg {
  color: #d93025; background: #fef0ef; border-radius: 6px;
  padding: 10px 16px; font-size: 13px; margin-top: 12px; display: none;
}
.col-error { color: #d93025; font-size: 12px; margin-top: 8px; min-height: 16px; }
.spinner {
  display: none; width: 18px; height: 18px; border: 3px solid #e0e0e0;
  border-top-color: #1a73e8; border-radius: 50%;
  animation: spin .6s linear infinite;
}
.spinner.show { display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
#fileInput { display: none; }
</style>
</head>
<body>

<div class="header">
  <h1>模型识别结果测试</h1>
  <p>上传无人机影像，选择模型与分辨率，对比三种方法检测效果</p>
</div>

<div class="container">

  <div class="card">
    <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="image/*">
      <img id="preview" style="display:none" alt="preview">
      <div class="placeholder" id="placeholder">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
        </svg>
        点击或拖拽上传图片
      </div>
    </div>

    <div class="options">
      <div class="opt-group">
        <label>方法:</label>
        <div class="method-checks">
          <label><input type="checkbox" name="method" value="yolo" checked><span>纯 YOLO</span></label>
          <label><input type="checkbox" name="method" value="gli" checked><span>YOLO + GLI</span></label>
          <label><input type="checkbox" name="method" value="sam" checked><span>YOLO + SAM</span></label>
          <label><input type="checkbox" name="method" value="segformer"><span>SegFormer</span></label>
        </div>
      </div>
      <div class="opt-group">
        <label>imgsz:</label>
        <div class="btn-group">
          <label><input type="radio" name="imgsz" value="640" checked><span>640</span></label>
          <label><input type="radio" name="imgsz" value="1280"><span>1280</span></label>
        </div>
      </div>
      <button class="btn-primary" id="detectBtn" onclick="runDetection()" disabled>
        开始检测
        <div class="spinner" id="mainSpinner"></div>
      </button>
    </div>
    <div class="error-msg" id="errorMsg"></div>
  </div>

  <div class="result-area" id="resultArea">
    <div class="orig-card">
      <h3>原图</h3>
      <img id="origImg" alt="original">
    </div>
    <div class="result-grid" id="resultGrid">
      <div class="result-col" id="col-yolo" style="display:none">
        <div class="col-header"><h3>纯 YOLO</h3><div class="spinner" id="spin-yolo"></div></div>
        <img id="img-yolo" class="result-img" alt="yolo">
        <div class="stats" id="stats-yolo"></div>
        <div class="col-error" id="err-yolo"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-yolo" download>⬇ 下载结果</a></div>
      </div>
      <div class="result-col" id="col-gli" style="display:none">
        <div class="col-header"><h3>YOLO + GLI</h3><div class="spinner" id="spin-gli"></div></div>
        <img id="img-gli" class="result-img" alt="gli">
        <div class="stats" id="stats-gli"></div>
        <div class="col-error" id="err-gli"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-gli" download>⬇ 下载结果</a></div>
      </div>
      <div class="result-col" id="col-sam" style="display:none">
        <div class="col-header"><h3>YOLO + SAM</h3><div class="spinner" id="spin-sam"></div></div>
        <img id="img-sam" class="result-img" alt="sam">
        <div class="stats" id="stats-sam"></div>
        <div class="col-error" id="err-sam"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-sam" download>⬇ 下载结果</a></div>
      </div>
      <div class="result-col" id="col-segformer" style="display:none">
        <div class="col-header"><h3>SegFormer</h3><div class="spinner" id="spin-segformer"></div></div>
        <img id="img-segformer" class="result-img" alt="segformer">
        <div class="stats" id="stats-segformer"></div>
        <div class="col-error" id="err-segformer"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-segformer" download>⬇ 下载结果</a></div>
      </div>
    </div>
  </div>

</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const placeholder = document.getElementById('placeholder');
const detectBtn = document.getElementById('detectBtn');
const errorMsg = document.getElementById('errorMsg');
const resultArea = document.getElementById('resultArea');
let uploadedB64 = null;
let origStem = 'image';

const METHODS = {
  yolo: { file: '纯YOLO' },
  gli:  { file: 'YOLO+GLI' },
  sam:  { file: 'YOLO+SAM' },
  segformer: { file: 'SegFormer' },
};

fileInput.addEventListener('change', e => { if (e.target.files.length) loadFile(e.target.files[0]); });
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) { showError('请上传图片文件'); return; }
  origStem = (file.name || 'image').replace(/\.[^.]+$/, '');
  const reader = new FileReader();
  reader.onload = e => {
    uploadedB64 = e.target.result;
    preview.src = uploadedB64;
    preview.style.display = 'block';
    placeholder.style.display = 'none';
    dropZone.classList.add('has-image');
    detectBtn.disabled = false;
    errorMsg.style.display = 'none';
    document.getElementById('origImg').src = uploadedB64;
  };
  reader.readAsDataURL(file);
}

function showError(msg) { errorMsg.textContent = msg; errorMsg.style.display = 'block'; }

function getSelectedMethods() {
  return [...document.querySelectorAll('input[name="method"]:checked')].map(el => el.value);
}

async function runDetection() {
  if (!uploadedB64) return;
  const methods = getSelectedMethods();
  if (methods.length === 0) { showError('请至少选择一种方法'); return; }
  const imgsz = parseInt(document.querySelector('input[name="imgsz"]:checked').value);

  detectBtn.disabled = true;
  errorMsg.style.display = 'none';
  resultArea.classList.add('show');
  document.getElementById('origImg').src = uploadedB64;

  // 重置三列：选中的显示+转圈，未选的隐藏
  ['yolo', 'gli','sam', 'segformer'].forEach(m => {
    const show = methods.includes(m);
    document.getElementById('col-' + m).style.display = show ? 'block' : 'none';
    if (show) {
      document.getElementById('spin-' + m).classList.add('show');
      document.getElementById('img-' + m).style.display = 'none';
      document.getElementById('dl-' + m).classList.add('disabled');
      document.getElementById('dl-' + m).removeAttribute('href');
      document.getElementById('stats-' + m).innerHTML = '';
      document.getElementById('err-' + m).textContent = '';
    }
  });

  // 串行推理（避免 GPU 显存冲突），每个完成立即显示
  for (const m of methods) {
    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image: uploadedB64, model: m, imgsz: imgsz})
      });
      const data = await resp.json();
      document.getElementById('spin-' + m).classList.remove('show');
      if (!resp.ok) {
        document.getElementById('err-' + m).textContent = data.error || '推理失败';
        continue;
      }
      const img = document.getElementById('img-' + m);
      img.src = data.result_b64;
      img.style.display = 'block';
      const dl = document.getElementById('dl-' + m);
      dl.href = data.result_b64;
      dl.download = origStem + '_' + METHODS[m].file + '.jpg';
      dl.classList.remove('disabled');
      let html = '<span class="total">共 ' + data.total + '</span>';
      for (const [name, count] of Object.entries(data.counts)) {
        html += '<span>' + name + ': ' + count + '</span>';
      }
      document.getElementById('stats-' + m).innerHTML = html;
    } catch (err) {
      document.getElementById('spin-' + m).classList.remove('show');
      document.getElementById('err-' + m).textContent = '请求失败: ' + err.message;
    }
  }
  detectBtn.disabled = false;
}
</script>

</body>
</html>"""  # noqa: E501


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_b64 = data["image"]
        mode = data.get("model", "yolo")
        imgsz = int(data.get("imgsz", 640))

        if imgsz not in (640, 1280):
            return jsonify({"error": f"imgsz 仅支持 640 / 1280, 收到 {imgsz}"}), 400

        img = b64_to_img(image_b64)
        if img is None:
            return jsonify({"error": "无法解码图片"}), 400

        # 推理
        if mode == "yolo":
            detections = pure_yolo(img, imgsz)
        elif mode == "gli":
            detections = yolo_gli_pipeline(img, imgsz)
        elif mode == "sam":
            if sam_segmenter is None:
                return jsonify({"error": "YOLO+SAM 未就绪: SAM 权重或 segmenter 加载失败"}), 503
            detections = yolo_sam(img, imgsz)
        elif mode == "segformer":
            if segformer_model is None:
                return jsonify({"error": "SegFormer 未就绪: 模型未训练或未找到"}), 503
            detections = segformer_predict(img)
        else:
            return jsonify({"error": f"未知模型模式: {mode}"}), 400

        # 画结果
        result_img = draw_detections(img, detections)

        # 统计
        from collections import Counter
        cnt = Counter(d[0] for d in detections)

        return jsonify({
            "result_b64": img_to_b64(result_img),
            "total": len(detections),
            "counts": dict(cnt),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print(f"\n  浏览器打开 http://localhost:5000")
    print(f"  内网其他人访问 http://<本机IP>:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
