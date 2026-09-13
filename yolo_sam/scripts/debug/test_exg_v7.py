"""ExG 低阈值 + V7 双路融合测试 - 50张"""
import cv2, numpy as np, random, warnings, torch
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
model = YOLO("runs/segment/runs/segment/hyacinth7_yolo_sam2/weights/best.pt")
random.seed(123)

imgs = []
for r, _, fs in Path("data").walk():
    for f in fs:
        p = Path(r) / f
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            imgs.append(p)

test = random.sample(imgs, 50)
out = Path("temp_exg_v7_std")
for f in out.glob("*"): f.unlink()
out.mkdir(exist_ok=True)
total = Counter()

for img_path in test:
    img = cv2.imread(str(img_path))
    if img is None: continue
    h, w = img.shape[:2]

    # ExG low threshold (Otsu * 0.7)
    rgb = img.astype(np.float32) / 255.0
    exg = 2 * rgb[:,:,1] - rgb[:,:,2] - rgb[:,:,0]
    exg_u8 = ((exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255).astype(np.uint8)
    t, _ = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, veg = cv2.threshold(exg_u8, int(t * 1.0), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, k)

    # V7 - no retina_masks to save VRAM
    res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else np.array([])
    xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else np.zeros((0, 4))
    xy = r0.masks.xy if r0.masks and hasattr(r0.masks, 'xy') else []

    exclude = np.zeros((h, w), np.uint8)
    tree_v7 = []
    for j in range(len(cls)):
        c = int(cls[j])
        if c in (0, 1, 2, 4):
            if j < len(xy) and len(xy[j]) >= 3:
                cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
                if c == 4: tree_v7.append(xy[j])
            else:
                x1, y1, x2, y2 = xyxy[j].astype(int)
                cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

    wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
    wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN, k)
    wh_cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wh_cnt = sum(1 for c in wh_cs if cv2.contourArea(c) > 50)
    total['WH(ExG+V7)'] += wh_cnt

    wh_union = np.zeros((h, w), np.uint8)
    for c in wh_cs:
        if cv2.contourArea(c) > 50: cv2.fillPoly(wh_union, [c], 255)

    tree_kept = 0
    for pts in tree_v7:
        tm = np.zeros((h, w), np.uint8); cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
        if cv2.countNonZero(cv2.bitwise_and(tm, wh_union)) / (cv2.countNonZero(tm) + 1) < 0.5:
            tree_kept += 1; total['tree'] += 1

    for j in range(len(cls)):
        c = int(cls[j])
        if c == 0: total['Boat'] += 1
        elif c == 1: total['Bridge'] += 1
        elif c == 2: total['Structure'] += 1

    # Draw
    overlay = img.copy()
    for c in wh_cs:
        if cv2.contourArea(c) > 50: cv2.drawContours(overlay, [c], -1, (0, 255, 0), 1)
    for pts in tree_v7:
        tm = np.zeros((h, w), np.uint8); cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
        if cv2.countNonZero(cv2.bitwise_and(tm, wh_union)) / (cv2.countNonZero(tm) + 1) < 0.5:
            cv2.drawContours(overlay, [pts.astype(np.int32)], -1, (0, 0, 255), 2)
    for j in range(len(cls)):
        if int(cls[j]) in (0, 1, 2):
            x1, y1, x2, y2 = xyxy[j].astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          (255, 128, 0) if cls[j] == 0 else (128, 128, 128) if cls[j] == 1 else (128, 0, 128), 2)

    cv2.putText(overlay, f"ExG*0.7+V7: WH={wh_cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"exgv7_{img_path.name}"), overlay)

print(f"ExG(Otsu*0.7) + V7, 50 images:")
for k, v in total.most_common(): print(f"  {k}: {v}")
print(f"-> {out}/")
