"""
V8 模型推理 + 后处理：把不相连的水葫芦区域拆成多个实例
输出: temp_v8_infer_split_20/
"""
import cv2, random, numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/segment/runs/segment/hyacinth8_yolo_sam/weights/best.pt"
DATA_DIR = Path("data")
OUT_DIR = Path("temp_v8_infer_split_20")
SEED = 42


def split_wh_mask(mask, min_area=50):
    """先断开细桥，再把 WH mask 中不连通的区域拆开"""
    # 开运算打断 A-B 之间的细桥连接
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cs if cv2.contourArea(c) > min_area]


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    model = YOLO(MODEL_PATH)

    imgs = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(imgs, min(20, len(imgs)))
    print(f"Sample: {len(sample)}")

    for img_path in sorted(sample):
        img = cv2.imread(str(img_path))
        if img is None or img_path.stat().st_size == 0:
            continue
        h, w = img.shape[:2]

        res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
        r0 = res[0]
        cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
        xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else []
        xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

        overlay = img.copy()
        wh_split_total = 0

        for j in range(len(cls)):
            c = int(cls[j])
            x1, y1, x2, y2 = xyxy[j].astype(int)
            color = {0: (255, 128, 0), 1: (128, 128, 128), 2: (128, 0, 128),
                     3: (0, 255, 0), 4: (0, 0, 255)}.get(c, (255, 255, 255))
            name = {0: "Boat", 1: "Bridge", 2: "Structure", 3: "WH", 4: "tree"}.get(c, str(c))

            if c == 3 and j < len(xy) and len(xy[j]) >= 3:
                # 拆分 WH mask
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [xy[j].astype(np.int32)], 255)
                cs = split_wh_mask(mask)
                wh_split_total += len(cs)
                for c2 in cs:
                    cv2.drawContours(overlay, [c2], -1, color, 2)
            elif c == 4 and j < len(xy) and len(xy[j]) >= 3:
                cv2.drawContours(overlay, [xy[j].astype(np.int32)], -1, color, 2)
            elif c in (0, 1, 2):
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(overlay, f"V8-split | WH={wh_split_total}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT_DIR / img_path.name), overlay)

    print(f"Done -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
