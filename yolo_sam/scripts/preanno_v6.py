import json, cv2, base64, sys, time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

model = YOLO("runs/segment/runs/segment/hyacinth6_yolo_sam2/weights/best.pt")
NAMES = ['Boat', 'Bridge', 'Structure', 'Water Hyacinth', 'tree']
IMGS = {'.jpg', '.jpeg', '.png'}

# Get latest 1000 from project 1
import requests, re
base = "http://192.168.30.107:8090"
s = requests.Session()
r = s.get(f"{base}/user/login/")
m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)
csrf = m.group(1) if m else None
s.post(f"{base}/user/login/", data={"email": "chs9710@163.com", "password": "cHENGS1997", "csrfmiddlewaretoken": csrf})
h = {"X-CSRFToken": s.cookies.get('csrftoken', '')}

all_tasks = []
for page in range(1, 10):
    r = s.get(f"{base}/api/projects/1/tasks?page={page}&page_size=500", headers=h)
    if r.status_code != 200: break
    data = r.json()
    tasks = data if isinstance(data, list) else data.get('tasks', [])
    if not tasks: break
    all_tasks.extend(tasks)

unlabeled = [t for t in all_tasks if not t.get('annotations')]

def ts(t):
    p = Path(t['data'].get('image', '')).name.split('_')
    return p[1] if len(p) >= 2 else ''

unlabeled.sort(key=ts, reverse=True)
target = unlabeled[:1000]

img_dir = Path('data/ls_images')
cnt = Counter()
total_boxes = 0

# Step 1: Run V6 and save to JSONL
print(f"Inference on {len(target)} images...")
records = []
for i, t in enumerate(target):
    fn = Path(t['data'].get('image', '')).name
    p = img_dir / fn
    if not p.exists() or p.suffix.lower() not in IMGS:
        continue

    img = cv2.imread(str(p))
    if img is None: continue
    hw, ww = img.shape[:2]
    _, buf = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buf).decode()

    rec = {"data": {"image": f"data:image/jpeg;base64,{b64}"}}

    r2 = model(str(p), conf=0.10, iou=0.5, imgsz=640, verbose=False, retina_masks=True)
    boxes = r2[0].boxes
    if boxes is not None and len(boxes) > 0:
        cls = boxes.cls.cpu().numpy().astype(int)
        xy = r2[0].masks.xy if r2[0].masks and hasattr(r2[0].masks, 'xy') else None
        preds = []
        for j in range(len(cls)):
            c = int(cls[j])
            if c >= 5: continue
            cnt[NAMES[c]] += 1
            total_boxes += 1
            if xy and j < len(xy) and len(xy[j]) >= 3:
                pts = [[round(float(p[0]) / ww * 100, 4), round(float(p[1]) / hw * 100, 4)] for p in xy[j]]
                preds.append({"type": "polygonlabels", "value": {"points": pts, "polygonlabels": [NAMES[c]]}, "origin": "prediction", "to_name": "image", "from_name": "label"})
            else:
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[j].cpu().numpy()]
                preds.append({"type": "rectanglelabels", "value": {"x": round(x1 / ww * 100, 4), "y": round(y1 / hw * 100, 4), "width": round((x2 - x1) / ww * 100, 4), "height": round((y2 - y1) / hw * 100, 4), "rectanglelabels": [NAMES[c]]}, "origin": "prediction", "to_name": "image", "from_name": "label"})
        if preds:
            rec["predictions"] = [{"model_version": "hyacinth6-v1", "result": preds}]

    records.append(rec)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(target)}...")

with_pred = sum(1 for r in records if 'predictions' in r)
print(f"Done: {len(records)} total, {with_pred} with predictions")
for c, n in cnt.most_common():
    print(f"  {c}: {n}")

# Save
with open('data/v6_1000.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Step 2: Clear project 3 and import
print("Clearing project 3...")
s.delete(f"{base}/api/projects/3/tasks/", headers=h)

print(f"Importing {len(records)} to project 3...")
imported = 0
for i in range(0, len(records), 20):
    batch = records[i:i + 20]
    r = s.post(f"{base}/api/projects/3/import", json=batch, headers={**h, "Content-Type": "application/json"})
    if r.status_code in (200, 201):
        imported += len(batch)
    else:
        print(f"  ERROR: {r.status_code}")
    if (i // 20 + 1) % 5 == 0:
        print(f"  {imported}/{len(records)}...")
    time.sleep(0.2)

print(f"Done! {imported} tasks in project 3, {with_pred} with predictions")
