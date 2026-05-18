import sys
sys.path.insert(0, "D:/chengs/9.project/shuihulu/src")

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry["vit_h"](checkpoint="D:/chengs/9.project/shuihulu/weights/sam_vit_h.pth")

sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.85,
    min_mask_region_area=1000,
)

img_bgr = cv2.imread("D:/chengs/9.project/shuihulu/datasets/DJI_20260115104645_0009_V.JPG")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(img_rgb)
print(f"Total masks: {len(masks)}")

for i, m in enumerate(masks[:5]):
    seg = m["segmentation"]
    area = int(seg.sum())
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  mask {i}: area={area}, contours={len(contours)}, seg_shape={seg.shape}, seg_dtype={seg.dtype}, unique={np.unique(seg)}")
    if contours:
        c = max(contours, key=cv2.contourArea)
        squeezed = c.squeeze(1)
        print(f"    largest contour: points={len(c)}, shape={c.shape}, squeeze_shape={squeezed.shape}")
        polygon = squeezed.tolist()
        print(f"    polygon len={len(polygon)}, first 3={polygon[:3]}")
