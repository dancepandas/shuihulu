"""V9 预标注 → Label Studio project 3
1. 从 data/ 随机抽 500 张
2. V9 + GLI 完整 pipeline 推理
3. 清空 project 3 并导入（base64 内嵌图片 + polygon 预测）
"""
import os, sys, json, base64, time, random, re
from pathlib import Path
from collections import Counter

import cv2, numpy as np
import torch
import requests

ROOT = Path(__file__).resolve().parents[2]  # 项目根 (data/ datasets/ 未动)
sys.path.insert(0, str(ROOT / "yolo_sam" / "src"))
from ultralytics import YOLO

# ═══════ 配置 ═══════
MODEL_PATH = r"D:\chengs\9.project\shuihulu\runs\segment\runs\segment\hyacinth9_yolo_sam3\weights\best.pt"
DATA_DIR = ROOT / "data"
N_SAMPLE = 500
LS_BASE = "http://192.168.30.107:8090"
LS_EMAIL = "chs9710@163.com"
LS_PWD = "cHENGS1997"
LS_SRC_PROJECT = 3      # 目标项目
JSONL_PATH = ROOT / "data/v9_500.jsonl"
NAMES = ["Boat", "Bridge", "Structure", "Water Hyacinth", "tree"]
CONF = 0.10
IMGSZ = 640
IMG_EXT = {".jpg", ".jpeg", ".png"}
MODEL_VERSION = "hyacinth9-v1"

# ═══════ 纯 YOLO 推理 ═══════
def v9_predict(img_bgr, model):
    h, w = img_bgr.shape[:2]
    torch.cuda.empty_cache()
    results = model(img_bgr, conf=CONF, iou=0.5, imgsz=IMGSZ, verbose=False, retina_masks=True)
    r0 = results[0]
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return []

    cls_arr = boxes.cls.cpu().numpy().astype(int)
    confs  = boxes.conf.cpu().numpy()
    xyxy   = boxes.xyxy.cpu().numpy()
    xy     = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    preds = []
    for j in range(len(cls_arr)):
        c = int(cls_arr[j])
        if c >= len(NAMES):
            continue
        if j < len(xy) and len(xy[j]) >= 3:
            pts = [[round(float(p[0]) / w * 100, 4), round(float(p[1]) / h * 100, 4)] for p in xy[j]]
        else:
            x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
            pts = [[round(x1/w*100,4), round(y1/h*100,4)], [round(x2/w*100,4), round(y1/h*100,4)],
                   [round(x2/w*100,4), round(y2/h*100,4)], [round(x1/w*100,4), round(y2/h*100,4)]]
        preds.append({"type": "polygonlabels",
                       "value": {"points": pts, "polygonlabels": [NAMES[c]]},
                       "origin": "prediction", "to_name": "image", "from_name": "label"})
    return preds


# ═══════ LS 登录 ═══════
def ls_login():
    s = requests.Session()
    s.trust_env = False
    for attempt in range(10):
        try:
            g = s.get(f"{LS_BASE}/user/login/", timeout=120)
            m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
            csrf = m.group(1) if m else None
            s.post(f"{LS_BASE}/user/login/",
                   data={"email": LS_EMAIL, "password": LS_PWD, "csrfmiddlewaretoken": csrf},
                   headers={"Referer": f"{LS_BASE}/user/login/"}, allow_redirects=True, timeout=120)
            h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
            v = s.get(f"{LS_BASE}/api/projects/{LS_SRC_PROJECT}/tasks?page_size=1", headers=h, timeout=120)
            if v.status_code == 200:
                print(f"[login] OK (attempt {attempt+1})")
                return s, h
        except Exception as e:
            pass
        time.sleep(3)
    raise RuntimeError("LS login failed")


# ═══════ 主流程 ═══════
def main():
    # 1. 收集图片：data/ + V9 训练/验证集
    print("[1/5] 收集图片...")
    all_imgs = []
    # data/ 目录
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                all_imgs.append(Path(root) / f)
    # V9 训练+验证集
    v9_dirs = [ROOT / "datasets/hyacinth9/images/train", ROOT / "datasets/hyacinth9/images/val"]
    v9_imgs = []
    for d in v9_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXT:
                    v9_imgs.append(f)
                    if f not in all_imgs:
                        all_imgs.append(f)
    print(f"  data/ 图片: {len(all_imgs) - len(v9_imgs)}, V9 图片: {len(v9_imgs)}")

    # 随机选 N_SAMPLE 张（确保 V9 图全部入选）
    random.seed(42)
    non_v9 = [p for p in all_imgs if p not in set(v9_imgs)]
    n_from_data = max(0, N_SAMPLE - len(v9_imgs))
    selected = v9_imgs + random.sample(non_v9, min(n_from_data, len(non_v9)))
    print(f"  选中 {len(selected)} 张 (V9={len(v9_imgs)} + data随机={len(selected)-len(v9_imgs)})")

    # 2. 加载模型
    print(f"\n[2/5] 加载模型 {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("  模型就绪")

    # 3. 推理
    print(f"\n[3/5] 纯 YOLO V9 推理 (imgsz={IMGSZ}, {len(selected)} 张)...")
    records = []
    cnt = Counter()
    total_boxes = 0

    for i, img_path in enumerate(selected):
        if img_path.stat().st_size == 0:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = base64.b64encode(buf).decode()
        rec = {"data": {"image": f"data:image/jpeg;base64,{b64}"}}

        preds = v9_predict(img, model)
        if preds:
            rec["predictions"] = [{"model_version": MODEL_VERSION, "result": preds}]
            for p in preds:
                for lbl in p["value"].get("polygonlabels", []):
                    cnt[lbl] += 1
                    total_boxes += 1

        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(selected)}...")

    with_pred = sum(1 for r in records if "predictions" in r)
    print(f"\n  完成: {len(records)} total, {with_pred} with predictions")
    for name, n in cnt.most_common():
        print(f"    {name}: {n}")

    # 保存 JSONL
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  已保存: {JSONL_PATH}")

    # 4. 登录 LS
    print(f"\n[4/5] 登录 Label Studio...")
    s, h = ls_login()

    # 5. 清空 project 3 并导入
    print(f"\n[5/5] 清空 project {LS_SRC_PROJECT} 并导入 {len(records)} 条...")
    print("  清空中...")
    s.delete(f"{LS_BASE}/api/projects/{LS_SRC_PROJECT}/tasks/", headers=h)
    time.sleep(2)

    imported = 0
    for i in range(0, len(records), 20):
        batch = records[i:i + 20]
        for retry in range(5):
            try:
                r = s.post(f"{LS_BASE}/api/projects/{LS_SRC_PROJECT}/import",
                           json=batch, headers={**h, "Content-Type": "application/json"}, timeout=180)
                if r.status_code in (200, 201):
                    imported += len(batch)
                    break
                else:
                    print(f"    batch {i//20+1}: HTTP {r.status_code}, retry {retry+1}")
                    time.sleep(5)
            except Exception as e:
                print(f"    batch {i//20+1}: exc {e}, retry {retry+1}")
                time.sleep(5)
        if (i // 20 + 1) % 5 == 0:
            print(f"  {imported}/{len(records)}...")
        time.sleep(0.3)

    print(f"\n  完成! {imported}/{len(records)} 已导入 project {LS_SRC_PROJECT}")


if __name__ == "__main__":
    main()
