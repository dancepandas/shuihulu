"""完整 V8 pipeline (GLI + V8模型) 单图推理 + 可视化"""
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

IMG = Path('data/2a745ca9c98a410c9fd2e3b30218e4f2.png')
OUT = Path('data/2a745ca9_v8_result.png')
NAMES = ['Boat', 'Bridge', 'Structure', 'Water Hyacinth', 'tree']

# ── 1. V8 模型推理 ──
model = YOLO('runs/hyacinth8_yolo_sam/weights/best.pt')
img = cv2.imread(str(IMG))
h, w = img.shape[:2]
print(f'image: {w}x{h}')

results = model(str(IMG), conf=0.10, iou=0.5, imgsz=640, verbose=False, retina_masks=True)
r0 = results[0]
boxes = r0.boxes
v8_cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) > 0 else np.array([])
v8_conf = boxes.conf.cpu().numpy() if len(v8_cls) > 0 else np.array([])
v8_xyxy = boxes.xyxy.cpu().numpy() if len(v8_cls) > 0 else np.zeros((0, 4))
v8_xy = r0.masks.xy if r0.masks and hasattr(r0.masks, 'xy') else []

# ── 2. GLI 植被掩膜 ──
rgb = img.astype(np.float32)
R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
# 归一化到 0-255 后 Otsu 自适应二值化
gli_norm = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
_, veg = cv2.threshold(gli_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# ── 3. 排除非 WH 区域 + YOLO-WH 掩膜 ──
exclude = np.zeros((h, w), np.uint8)
yolo_wh = np.zeros((h, w), np.uint8)
tree_pts = []
for j in range(len(v8_cls)):
    c = int(v8_cls[j])
    if c in (0, 1, 2):  # Boat, Bridge, Structure → exclude
        if j < len(v8_xy) and len(v8_xy[j]) >= 3:
            cv2.fillPoly(exclude, [v8_xy[j].astype(np.int32)], 255)
        else:
            x1, y1, x2, y2 = v8_xyxy[j].astype(int)
            cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
    elif c == 3:  # Water Hyacinth → YOLO mask for constraint
        if j < len(v8_xy) and len(v8_xy[j]) >= 3:
            cv2.fillPoly(yolo_wh, [v8_xy[j].astype(np.int32)], 255)
        else:
            x1, y1, x2, y2 = v8_xyxy[j].astype(int)
            cv2.rectangle(yolo_wh, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
    elif c == 4:  # tree
        if j < len(v8_xy) and len(v8_xy[j]) >= 3:
            tree_pts.append(v8_xy[j])

# ── 4. WH = GLI - exclude, 闭运算合并碎片 ──
wh_mask = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
wh_mask = cv2.morphologyEx(wh_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
wh_mask = cv2.morphologyEx(wh_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=2)
wh_cs, _ = cv2.findContours(wh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ── 5. 收集 WH 多边形（YOLO-WH 交集约束 >25%） ──
wh_polys = []
wh_union = np.zeros((h, w), np.uint8)
for ct in wh_cs:
    if cv2.contourArea(ct) < 50:  # 最小面积
        continue
    ct_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(ct_mask, [ct], -1, 255, -1)
    inter = cv2.countNonZero(cv2.bitwise_and(ct_mask, yolo_wh))
    area = cv2.countNonZero(ct_mask)
    if area > 0 and inter / area > 0.25:  # 交集约束
        cv2.fillPoly(wh_union, [ct], 255)
        wh_polys.append(ct[:, 0, :])

# ── 6. Tree 重叠判断（与 WH 重叠 >50% 则吸收） ──
keep_trees = []
for pts in tree_pts:
    tm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(tm, [pts.astype(np.int32)], 255)
    overlap = cv2.countNonZero(cv2.bitwise_and(tm, wh_union))
    area = cv2.countNonZero(tm)
    if area > 0 and overlap / area < 0.5:  # 重叠 <50% → 保留为 tree
        keep_trees.append(pts)

# ── 7. 收集 Boat/Bridge/Structure ──
non_wh = []
for j in range(len(v8_cls)):
    c = int(v8_cls[j])
    if c in (0, 1, 2):
        if j < len(v8_xy) and len(v8_xy[j]) >= 3:
            non_wh.append((c, v8_xy[j]))
        else:
            non_wh.append((c, v8_xyxy[j]))

# ── 报表 ──
print(f'\n{"="*50}')
print(f'检测结果:')
print(f'  Water Hyacinth (GLI+模型): {len(wh_polys)} 个')
print(f'  tree (保留):                {len(keep_trees)} 个')
print(f'  Boat/Bridge/Structure:     {len(non_wh)} 个')
print(f'  GLI 总 WH 面积:            {cv2.countNonZero(wh_union)} px ({cv2.countNonZero(wh_union)/w/h*100:.1f}%)')
print(f'{"="*50}')

# ── 可视化 ──
vis = img.copy()
colors = {
    'WH': (0, 255, 0),        # 绿色 — Water Hyacinth
    'Boat': (0, 165, 255),    # 橙色
    'Bridge': (255, 255, 0),  # 青色
    'Structure': (0, 0, 255), # 红色
    'tree': (0, 255, 128),    # 蓝绿
}

# GLI WH 多边形
for poly in wh_polys:
    cv2.drawContours(vis, [poly.astype(np.int32)], -1, colors['WH'], 3)
# Tree
for pts in keep_trees:
    cv2.drawContours(vis, [pts.astype(np.int32)], -1, colors['tree'], 2)
# Boat/Bridge/Structure
for c, geom in non_wh:
    name = NAMES[c]
    if len(geom) >= 3 and geom.ndim == 2:  # polygon
        cv2.drawContours(vis, [geom.astype(np.int32)], -1, colors[name], 2)
        cx, cy = int(geom[:, 0].mean()), int(geom[:, 1].mean())
    else:  # bbox
        x1, y1, x2, y2 = [int(v) for v in geom]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors[name], 2)
        cx, cy = (x1 + x2) // 2, y1 - 10
    cv2.putText(vis, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)

cv2.imwrite(str(OUT), vis)
print(f'\n可视化保存: {OUT}')
