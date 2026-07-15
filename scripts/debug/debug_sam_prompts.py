"""
对比 SAM prompt 策略：
- 策略0: box prompt
- 策略1: YOLO mask 作为 SAM mask_input prompt
- 策略1+2: YOLO mask prompt + 结果 clip 到 YOLO box
对多张图批量输出
"""
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

IMG_PATHS = [
    "data/20260618190000428800/DJI_202606181900_002_20260618190000428800/DJI_20260618190525_0063_V.jpeg",
    "data/20260617110000847744/DJI_202606171059_002_20260617110000847744/DJI_20260617110553_0070_V.jpeg",
    "data/20260617090000659136/DJI_202606170900_002_20260617090000659136/DJI_20260617090428_0048_V.jpeg",
]
OUT_DIR = Path("temp_debug_sam_prompts")
YOLO_PATH = "runs/hyacinth8_yolo_sam/weights/best.pt"
SAM_PATH = "weights/sam_vit_h.pth"


def mask_from_poly(xy, h, w):
    m = np.zeros((h, w), np.uint8)
    if len(xy) >= 3:
        cv2.fillPoly(m, [xy.astype(np.int32)], 255)
    return m


def draw_masks(img, masks, boxes=None, title=""):
    canvas = img.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        m = m.astype(bool)
        overlay = canvas.copy()
        overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        canvas = overlay
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cs, -1, color, 2)
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def process_image(img_path, yolo, predictor, out_dir):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Cannot read {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    stem = Path(img_path).stem

    res = yolo(img_path, conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int)
    xyxy = r0.boxes.xyxy.cpu().numpy()
    xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []
    target_idx = [j for j in range(len(cls)) if int(cls[j]) == 3]

    predictor.set_image(img_rgb)

    # 0: box prompt
    masks_box = []
    for j in target_idx:
        m, _, _ = predictor.predict(box=xyxy[j], multimask_output=False)
        masks_box.append(m[0])

    # 1: mask prompt
    masks_mask = []
    for j in target_idx:
        if j < len(xy) and len(xy[j]) >= 3:
            yolo_mask_full = mask_from_poly(xy[j], h, w).astype(np.float32) / 255.0
            yolo_mask_input = cv2.resize(yolo_mask_full, (256, 256), interpolation=cv2.INTER_NEAREST)
            M = cv2.moments((yolo_mask_full * 255).astype(np.uint8))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pts = np.array([[cx, cy]])
                lbls = np.array([1])
            else:
                pts = None
                lbls = None
            m, _, _ = predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                mask_input=yolo_mask_input[None, ...],
                multimask_output=False,
            )
            masks_mask.append(m[0])
        else:
            m, _, _ = predictor.predict(box=xyxy[j], multimask_output=False)
            masks_mask.append(m[0])

    # 1+2: mask prompt + clip
    masks_mask_clip = []
    for k, j in enumerate(target_idx):
        m = masks_mask[k].copy()
        x1, y1, x2, y2 = xyxy[j].astype(int)
        m[:max(0, y1), :] = False
        m[min(h, y2):, :] = False
        m[:, :max(0, x1)] = False
        m[:, min(w, x2):] = False
        masks_mask_clip.append(m)

    cv2.imwrite(str(out_dir / f"{stem}_00_yolo_mask.jpg"),
                draw_masks(img_bgr, [mask_from_poly(xy[j], h, w) for j in target_idx if j < len(xy)],
                           boxes=xyxy[target_idx], title="YOLO mask"))
    cv2.imwrite(str(out_dir / f"{stem}_01_box_prompt.jpg"),
                draw_masks(img_bgr, masks_box, boxes=xyxy[target_idx], title="Box prompt"))
    cv2.imwrite(str(out_dir / f"{stem}_02_mask_prompt.jpg"),
                draw_masks(img_bgr, masks_mask, boxes=xyxy[target_idx], title="Mask prompt"))
    cv2.imwrite(str(out_dir / f"{stem}_03_mask_prompt_clip.jpg"),
                draw_masks(img_bgr, masks_mask_clip, boxes=xyxy[target_idx], title="Mask prompt + clip"))

    print(f"{stem}: WH={len(target_idx)}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    yolo = YOLO(YOLO_PATH)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    for img_path in IMG_PATHS:
        process_image(img_path, yolo, predictor, OUT_DIR)

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
