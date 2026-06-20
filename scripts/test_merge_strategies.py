"""
测试多种水葫芦块合并策略

输入: datasets/hyacinth8/images/train/ 中的图片
输出: temp_merge_test/ 下各种策略的可视化结果 + 统计
"""
import random, os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

V7_PATH = "runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_DIR = Path("data")
OUT_DIR = Path("temp_merge_test")
IMG_EXT = {".jpg", ".jpeg", ".png"}
GLI_THRESHOLD = 0.12
MIN_AREA = 50
RANDOM_SEED = 123
SAMPLE_SIZE = 20


def compute_gli_mask(img):
    rgb = img.astype(np.float32)
    R, G, B = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]
    gli = (2 * G - R - B) / (2 * G + R + B + 1e-8)
    _, veg = cv2.threshold(gli, GLI_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    veg = (veg * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
    return veg


def get_v7_exclude_mask(img_path, h, w, model):
    exclude = np.zeros((h, w), np.uint8)
    res = model(str(img_path), conf=0.10, iou=0.5, imgsz=640, verbose=False)
    r0 = res[0]
    cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None and len(r0.boxes) > 0 else []
    xyxy = r0.boxes.xyxy.cpu().numpy() if len(cls) > 0 else np.zeros((0, 4))
    xy = r0.masks.xy if r0.masks and hasattr(r0.masks, "xy") else []

    for j in range(len(cls)):
        c = int(cls[j])
        if c in (0, 1, 2):
            if j < len(xy) and len(xy[j]) >= 3:
                cv2.fillPoly(exclude, [xy[j].astype(np.int32)], 255)
            else:
                x1, y1, x2, y2 = xyxy[j].astype(int)
                cv2.rectangle(exclude, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
    return exclude


def get_wh_contours(img_path, model):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []
    h, w = img.shape[:2]
    veg = compute_gli_mask(img)
    exclude = get_v7_exclude_mask(img_path, h, w, model)
    wh = cv2.bitwise_and(veg, cv2.bitwise_not(exclude))
    wh = cv2.morphologyEx(wh, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(wh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > MIN_AREA]
    return img, cs


def strategy_none(cs, h=None, w=None):
    return cs


def strategy_dilate_erode(cs, h, w, ksize=15, iterations=2):
    """形态学闭运算合并"""
    mask = np.zeros((h, w), np.uint8)
    for c in cs:
        cv2.drawContours(mask, [c], -1, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=iterations)
    cs2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cs2 if cv2.contourArea(c) > MIN_AREA]


def strategy_dbscan(cs, eps=40, min_samples=1):
    """基于质心的 DBSCAN 聚类后合并"""
    if len(cs) < 2:
        return cs
    centers = []
    valid_idx = []
    for i, c in enumerate(cs):
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        valid_idx.append(i)
    centers = np.array(centers)

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centers)

    merged = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        idxs = [valid_idx[i] for i in range(len(labels)) if labels[i] == lab]
        pts = np.vstack([cs[i].reshape(-1, 2) for i in idxs])
        hull = cv2.convexHull(pts)
        merged.append(hull)
    return merged


def strategy_distance_merge(cs, threshold=40):
    """迭代合并最近邻对，直到最近邻距离 > threshold"""
    cs = list(cs)
    changed = True
    while changed and len(cs) >= 2:
        changed = False
        centers = []
        valid = []
        for c in cs:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            valid.append(c)
        if len(valid) < 2:
            break
        centers = np.array(centers)

        # 找全局最近的两个
        min_dist = np.inf
        pair = (0, 1)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
                    pair = (i, j)
        if min_dist <= threshold:
            i, j = pair
            pts = np.vstack([valid[i].reshape(-1, 2), valid[j].reshape(-1, 2)])
            merged = cv2.convexHull(pts)
            new_cs = [c for k, c in enumerate(valid) if k not in (i, j)]
            new_cs.append(merged)
            cs = new_cs
            changed = True
    return cs


def draw_contours(img, cs, color=(0, 255, 0), thickness=2):
    overlay = img.copy()
    cv2.drawContours(overlay, cs, -1, color, thickness)
    return overlay


def main():
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.glob("*"):
        f.unlink()

    images = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in IMG_EXT:
                images.append(p)
    sample = random.sample(images, min(SAMPLE_SIZE, len(images)))

    model = YOLO(V7_PATH)

    strategies = {
        "none": strategy_none,
        "dilate15": lambda cs, h, w: strategy_dilate_erode(cs, h, w, 15, 2),
        "dilate25": lambda cs, h, w: strategy_dilate_erode(cs, h, w, 25, 2),
        "dist30": lambda cs, h, w: strategy_distance_merge(cs, 30),
        "dist50": lambda cs, h, w: strategy_distance_merge(cs, 50),
        "dist70": lambda cs, h, w: strategy_distance_merge(cs, 70),
    }

    stats = {k: [] for k in strategies}

    for img_path in sample:
        img, cs = get_wh_contours(img_path, model)
        if img is None:
            continue
        h, w = img.shape[:2]
        base = draw_contours(img, cs, (128, 128, 128), 1)

        for name, fn in strategies.items():
            merged = fn(cs, h, w)
            overlay = draw_contours(base.copy(), merged, (0, 255, 0), 2)
            cv2.putText(overlay, f"{name} n={len(merged)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out = OUT_DIR / f"{img_path.stem}_{name}.jpg"
            cv2.imwrite(str(out), overlay)
            stats[name].append(len(merged))

    print("=== Strategy comparison (20 images) ===")
    print(f"{'Strategy':<12} {'Mean':>8} {'Median':>8} {'Max':>6}")
    print("-" * 40)
    for name, counts in stats.items():
        arr = np.array(counts)
        print(f"{name:<12} {arr.mean():>8.1f} {np.median(arr):>8.1f} {arr.max():>6}")
    print(f"\nOutput: {OUT_DIR}/")


if __name__ == "__main__":
    main()
