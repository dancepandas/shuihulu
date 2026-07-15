"""
Quick script: GLI + 形态学标注图中绿植范围
输入: doc/vis_boat_tree.jpg 的原图
输出: doc/vis_gli_vegetation.jpg
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("D:/chengs/9.project/shuihulu")
IMG = ROOT / "data/20260617060000459136/DJI_202606170600_002_20260617060000459136/DJI_20260617060256_0025_V.jpeg"
OUT = ROOT / "doc/vis_gli_vegetation.jpg"

img = cv2.imread(str(IMG))
h, w = img.shape[:2]

# GLI
rgb = img.astype(np.float32)
R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)

# Otsu 自适应阈值
gli_u8 = ((gli - gli.min()) / (gli.max() - gli.min() + 1e-8) * 255).astype(np.uint8)
_, veg = cv2.threshold(gli_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 形态学
open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, open_k)
veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, merge_k, iterations=2)

# 找轮廓
cs, _ = cv2.findContours(veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cs = [c for c in cs if cv2.contourArea(c) >= 50]

# 画图：半透明绿色填充 + 轮廓
overlay = img.copy()
for ct in cs:
    cv2.drawContours(overlay, [ct], -1, (0, 255, 0), -1)
blend = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
for ct in cs:
    cv2.drawContours(blend, [ct], -1, (0, 200, 0), 1)

total_area = sum(int(cv2.contourArea(c)) for c in cs)
cv2.putText(blend, f"GLI + morphology  close(25x25,2)  n={len(cs)}  area={total_area}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

cv2.imwrite(str(OUT), blend)
print(f"Saved: {OUT}")
print(f"Contours: {len(cs)},  total area: {total_area} px")
