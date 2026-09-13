"""
分析 GLI 伪标签中水葫芦块之间的距离分布
用于判断是否需要对过近的水葫芦块进行合并

用法: python scripts/analyze_wh_fragment_distance.py
输出: 最近邻距离直方图统计
"""
import random, os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

V7_PATH = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_DIR = Path("datasets/hyacinth8/images/train")
IMG_EXT = {".jpg", ".jpeg", ".png"}
GLI_THRESHOLD = 0.12
MIN_AREA = 50
RANDOM_SEED = 42
SAMPLE_SIZE = 20


def compute_gli_mask(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    _, veg = cv2.threshold(gli, GLI_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    veg = (veg * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
    return veg


def main():
    random.seed(RANDOM_SEED)

    images = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMG_EXT:
                images.append(p)

    sample = random.sample(images, min(SAMPLE_SIZE, len(images)))
    print(f"Sample: {len(sample)} images")

    model = YOLO(V7_PATH)

    all_min_distances = []
    per_image_stats = []

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        veg = compute_gli_mask(img)

        # V7 non-WH exclude
        res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
        r0 = res[0]
        cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
        xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else np.zeros((0, 4))
        xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

        exclude = np.zeros((h, w), np.uint8)
        for j in range(len(cls)):
            c = int(cls[j])
            if c in (0, 1, 2):
                if j < len(xy) and len(xy[j]) >= 3:
                    cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
                else:
                    x1, y1, x2, y2 = xyxy[j].astype(int)
                    cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
        wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        wh_cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wh_cs = [c for c in wh_cs if cv2.contourArea(c) > MIN_AREA]

        if len(wh_cs) < 2:
            per_image_stats.append((img_path.name, len(wh_cs), 0))
            continue

        # 计算各轮廓质心
        centers = []
        for ct in wh_cs:
            M = cv2.moments(ct)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

        centers = np.array(centers)
        # 每个块的最近邻距离
        min_dists = []
        for i, c1 in enumerate(centers):
            dists = np.linalg.norm(centers - c1, axis=1)
            dists[i] = np.inf
            min_dists.append(dists.min())

        all_min_distances.extend(min_dists)
        per_image_stats.append((img_path.name, len(wh_cs), np.mean(min_dists) if min_dists else 0))

    if not all_min_distances:
        print("No fragments found")
        return

    arr = np.array(all_min_distances)
    print(f"\n{'=' * 60}")
    print(f"Fragments analyzed: {len(arr)}")
    print(f"Min distance: {arr.min():.1f} px")
    print(f"Max distance: {arr.max():.1f} px")
    print(f"Mean: {arr.mean():.1f} px")
    print(f"Median: {np.median(arr):.1f} px")
    print(f"P25: {np.percentile(arr, 25):.1f} px")
    print(f"P75: {np.percentile(arr, 75):.1f} px")
    print(f"P90: {np.percentile(arr, 90):.1f} px")
    print(f"P95: {np.percentile(arr, 95):.1f} px")
    print(f"P99: {np.percentile(arr, 99):.1f} px")

    print(f"\nHistogram (px):")
    hist, bins = np.histogram(arr, bins=20)
    for i in range(len(hist)):
        print(f"  {bins[i]:>7.1f} ~ {bins[i+1]:>7.1f}: {hist[i]:>4} ({hist[i]/len(arr)*100:.1f}%)")

    print(f"\nTop 10 images with most fragments:")
    for name, n, md in sorted(per_image_stats, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:40s} n={n:>4} mean_min_dist={md:.1f}px")


if __name__ == "__main__":
    main()
