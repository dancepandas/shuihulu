#!/usr/bin/env python
"""SegFormer 路线 Web 推理服务 — SegFormer (6类) / SegFormer-v10 (5类) 两模式
(自根目录 app.py 拆分; YOLO/GLI/SAM 三模式见 yolo_sam/app.py)

启动: python segformer/app.py  →  http://localhost:5001
"""

import io, base64, traceback
from pathlib import Path

import cv2, numpy as np, torch
from flask import Flask, request, jsonify, render_template_string

# ═══════════ 配置 (runs/ 仍在项目根, 绝对路径不变) ═══════════
SEGFORMER_PATH = r"D:\chengs\9.project\shuihulu\runs\segformer\segformer_b2_ls+v9"
SEGFORMER_V10_PATH = r"D:\chengs\9.project\shuihulu\runs\segformer\segformer_b2_ls_v11b"

# SegFormer 推理预处理 (ImageNet 归一化)
SF_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SF_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]
MIN_AREA = 50
SEGFORMER_SIZE = 512   # SegFormer 推理分辨率 (v9)
SEGFORMER_V10_SIZE = 640  # SegFormer v10 推理分辨率 (训练用 640)
# v10 5类: 0=water(背景) 1=water_hyacinth 2=hard_structure 3=shore_vegetation 4=other_aquatic_vegetation
V10_NAMES = {1: "Water Hyacinth", 2: "Hard Structure", 3: "Shore Vegetation",
             4: "Other Aquatic Veg"}

# ═══════════ 加载模型 ═══════════
segformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segformer_model = None
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

# SegFormer v10 (5类新标签)
segformer_v10_model = None
if Path(SEGFORMER_V10_PATH).joinpath("config.json").exists():
    try:
        from transformers import SegformerForSemanticSegmentation
        segformer_v10_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_V10_PATH)
        segformer_v10_model.to(segformer_device)
        segformer_v10_model.eval()
        print("[init] SegFormer v10 (5类) loaded")
    except Exception as e:
        print(f"[init] SegFormer v10 load failed: {e}")
else:
    print(f"[init] SegFormer v10 not found at {SEGFORMER_V10_PATH}, mode disabled")


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


def segformer_v10_predict(img_bgr):
    """SegFormer v10 (5类) 语义分割 → 每个连通域作为一个检测
    类别: 0=water(水体背景, 不检测) 1=water_hyacinth 2=hard_structure
          3=shore_vegetation 4=other_aquatic_vegetation
    """
    h, w = img_bgr.shape[:2]

    # 预处理: RGB → resize 640 → normalize (v10 训练分辨率)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SEGFORMER_V10_SIZE, SEGFORMER_V10_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - torch.from_numpy(SF_MEAN).view(3, 1, 1)) / torch.from_numpy(SF_STD).view(3, 1, 1)
    tensor = tensor.unsqueeze(0).to(segformer_device)

    with torch.no_grad():
        logits = segformer_v10_model(pixel_values=tensor).logits  # (1,5,160,160)
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        class_map = logits[0].argmax(dim=0).cpu().numpy()  # (H,W) values 0-4

    # 前景类 (跳过 0=water): 每个类找连通域
    dets = []
    for cls_id in range(1, 5):  # 1=water_hyacinth, 2=hard_structure, 3=shore_vegetation, 4=other_aquatic_vegetation
        binary = (class_map == cls_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        name = V10_NAMES[cls_id]
        for cnt in contours:
            if len(cnt) < 3: continue
            area = cv2.contourArea(cnt)
            if area < MIN_AREA: continue
            dets.append((name, cnt[:, 0, :], -1.0))

    return dets


# ═══════════ 可视化 ═══════════
def draw_detections(img_bgr, detections):
    """实例分割风格: 每个实例填充随机颜色半透明掩膜 + 轮廓 + 标签"""
    vis = img_bgr.copy()
    overlay = vis.copy()

    import random
    for name, geom, conf in detections:
        hue = random.randint(0, 179)
        color_hsv = np.uint8([[[hue, 220, 220]]])
        color = tuple(int(c) for c in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0])

        if isinstance(geom, np.ndarray) and geom.ndim == 2 and geom.shape[1] >= 2:
            pts = geom.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
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
<title>模型识别结果测试 — SegFormer 语义分割</title>
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
.container { max-width: 1200px; margin: 0 auto; padding: 24px 16px; }
.card {
  background: #fff; border-radius: 10px; padding: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px;
}
.upload-zone {
  border: 2px dashed #c0c8d4; border-radius: 8px;
  padding: 40px 20px; text-align: center; cursor: pointer;
  transition: border-color .2s, background .2s;
}
.upload-zone:hover, .upload-zone.drag-over { border-color: #1a73e8; background: #f4f8ff; }
.upload-zone.has-image { padding: 12px; }
.upload-zone img { max-height: 240px; border-radius: 4px; }
.upload-zone .placeholder { color: #8899aa; font-size: 14px; }
.upload-zone .placeholder svg { display: block; margin: 0 auto 10px; opacity: .4; }
.options { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; margin-top: 16px; }
.opt-group { display: flex; align-items: center; gap: 8px; }
.opt-group > label { font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }
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
.result-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
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
  <h1>模型识别结果测试 — SegFormer 语义分割</h1>
  <p>上传无人机影像，对比 SegFormer (6类) / SegFormer-v10 (5类新标签) 两种分割效果</p>
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
          <label><input type="checkbox" name="method" value="segformer"><span>SegFormer</span></label>
          <label><input type="checkbox" name="method" value="segformer_v10" checked><span>SegFormer-v10</span></label>
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
      <div class="result-col" id="col-segformer" style="display:none">
        <div class="col-header"><h3>SegFormer</h3><div class="spinner" id="spin-segformer"></div></div>
        <img id="img-segformer" class="result-img" alt="segformer">
        <div class="stats" id="stats-segformer"></div>
        <div class="col-error" id="err-segformer"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-segformer" download>⬇ 下载结果</a></div>
      </div>
      <div class="result-col" id="col-segformer_v10" style="display:none">
        <div class="col-header"><h3>SegFormer-v10</h3><div class="spinner" id="spin-segformer_v10"></div></div>
        <img id="img-segformer_v10" class="result-img" alt="segformer_v10">
        <div class="stats" id="stats-segformer_v10"></div>
        <div class="col-error" id="err-segformer_v10"></div>
        <div class="col-footer"><a class="btn-download disabled" id="dl-segformer_v10" download>⬇ 下载结果</a></div>
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
  segformer: { file: 'SegFormer' },
  segformer_v10: { file: 'SegFormer-v10' },
};
const ALL_METHODS = ['segformer', 'segformer_v10'];

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

  detectBtn.disabled = true;
  errorMsg.style.display = 'none';
  resultArea.classList.add('show');
  document.getElementById('origImg').src = uploadedB64;

  ALL_METHODS.forEach(m => {
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
        body: JSON.stringify({image: uploadedB64, model: m})
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
        mode = data.get("model", "segformer_v10")

        img = b64_to_img(image_b64)
        if img is None:
            return jsonify({"error": "无法解码图片"}), 400

        if mode == "segformer":
            if segformer_model is None:
                return jsonify({"error": "SegFormer 未就绪: 模型未训练或未找到"}), 503
            detections = segformer_predict(img)
        elif mode == "segformer_v10":
            if segformer_v10_model is None:
                return jsonify({"error": "SegFormer v10 未就绪: 模型未训练或未找到"}), 503
            detections = segformer_v10_predict(img)
        else:
            return jsonify({"error": f"未知模型模式: {mode} (本服务仅 segformer/segformer_v10)"}), 400

        result_img = draw_detections(img, detections)

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
    print(f"\n  浏览器打开 http://localhost:5001")
    print(f"  内网其他人访问 http://<本机IP>:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
