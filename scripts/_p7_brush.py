"""Polygon → BrushLabels RLE 转换 + 导入 project 7

关键: 用 LS SDK 的 mask2rle (bit-packed bitstream → bytes 0-255),
不能用普通 run-length 计数 — 那种 LS 前端 decode_rle 解不出来。
"""
import json, time, base64, re, uuid
import numpy as np
import cv2, requests
from label_studio_sdk.converter.brush import mask2rle

BASE = "http://localhost:8080"
EMAIL, PWD = "chs9710@163.com", "Chengs1997"
JSONL = "data/v9_500.jsonl"
PROJECT = 7


def login():
    s = requests.Session()
    s.trust_env = False
    for _ in range(10):
        try:
            g = s.get(f"{BASE}/user/login/", timeout=30)
            m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
            s.post(f"{BASE}/user/login/",
                   data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": m.group(1)},
                   headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=30)
            h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
            # 验证用 projects 列表 (不依赖某 project 是否为空)
            if s.get(f"{BASE}/api/projects?page_size=1", headers=h, timeout=30).status_code == 200:
                return s, h
        except Exception as e:
            print(f"  login try: {e}")
        time.sleep(3)
    raise RuntimeError("login failed")


s, h = login()
print("login OK")

# 1. 清空
print(f"clearing project {PROJECT}...")
for _ in range(3):
    try:
        s.delete(f"{BASE}/api/projects/{PROJECT}/tasks/", headers=h, timeout=60); break
    except Exception:
        time.sleep(2)
time.sleep(1)

# 2. BrushLabels config
config = ('<View><Header value="Brush tool"/>'
          '<Image name="image" value="$image" zoom="true"/>'
          '<BrushLabels name="label" toName="image">'
          '<Label value="Water Hyacinth" background="#9ef4ff"/>'
          '<Label value="Structure" background="#D4380D"/>'
          '<Label value="Boat" background="#FFC069"/>'
          '<Label value="tree" background="#00ad62"/>'
          '<Label value="Bridge" background="#D3F261"/>'
          '</BrushLabels></View>')
r = s.patch(f"{BASE}/api/projects/{PROJECT}/", json={"label_config": config},
            headers={**h, "Content-Type": "application/json"}, timeout=30)
print(f"config: {r.status_code}")

# 3. 读 JSONL
with open(JSONL, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]
print(f"loaded {len(records)} records")

# 4. 转换
new_records = []
for i, rec in enumerate(records):
    b64_str = rec["data"]["image"]
    _, encoded = b64_str.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        new_records.append(rec)
        continue
    H, W = img.shape[:2]

    # ---- 同标签合成为一个 mask → 减少笔刷层数(前端每层都要解码 4032x3024x4 RGBA) ----
    label_masks = {}  # label_name → binary mask (H,W)
    for pred in rec.get("predictions", []):
        for item in pred["result"]:
            pts_pct = item["value"]["points"]
            label = item["value"]["polygonlabels"][0]  # 每个 item 一个 label
            pts = np.array([[p[0] / 100 * W, p[1] / 100 * H] for p in pts_pct], dtype=np.int32)
            if label not in label_masks:
                label_masks[label] = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(label_masks[label], [pts], 255)

    new_ann_items = []
    for label, mask in label_masks.items():
        rle = mask2rle(mask)
        new_ann_items.append({
            "original_width": W,
            "original_height": H,
            "image_rotation": 0,
            "value": {"format": "rle", "rle": rle, "brushlabels": [label]},
            "id": uuid.uuid4().hex[:8],
            "from_name": "label",
            "to_name": "image",
            "type": "brushlabels",
            "origin": "manual",
        })

    nr = {"data": rec["data"]}
    if new_ann_items:
        nr["annotations"] = [{"result": new_ann_items, "was_cancelled": False}]
    new_records.append(nr)

    if (i + 1) % 50 == 0:
        print(f"  convert {i+1}/{len(records)}")

print(f"converted, importing...")

# 5. 导入
imported = 0
for i in range(0, len(new_records), 5):  # brush RLE 大, 小批量
    batch = new_records[i:i + 5]
    for retry in range(5):
        try:
            r = s.post(f"{BASE}/api/projects/{PROJECT}/import", json=batch,
                       headers={**h, "Content-Type": "application/json"}, timeout=120)
            if r.status_code in (200, 201):
                imported += len(batch); break
            print(f"  batch HTTP {r.status_code}: {r.text[:120]}")
            time.sleep(min(2 ** retry, 10))
        except Exception as e:
            time.sleep(min(2 ** retry, 10))
    if (i // 5 + 1) % 10 == 0:
        print(f"  {imported}/{len(new_records)}")

print(f"DONE: {imported}/{len(new_records)}")
