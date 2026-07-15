"""
检查训练标签中是否存在远距离粘连的 WH 标签
输出每张图最大 WH 标签内的不连通区域数
"""
import os
from pathlib import Path
import cv2
import numpy as np

LABEL_DIR = Path("datasets/hyacinth8/labels/train")


def main():
    files = sorted(LABEL_DIR.rglob("*.txt"))
    print(f"Labels: {len(files)}")

    bad_cases = []
    total_wh = 0
    total_fragments = 0

    for lf in files:
        lines = lf.read_text().strip().split("\n")
        for line in lines:
            parts = line.strip().split()
            if not parts or int(parts[0]) != 3:
                continue
            total_wh += 1
            pts = np.array([[float(parts[i]), float(parts[i+1])]
                            for i in range(1, len(parts), 2)])
            if len(pts) < 3:
                continue
            # 简单估算坐标范围
            xs, ys = pts[:, 0], pts[:, 1]
            w_range = xs.max() - xs.min()
            h_range = ys.max() - ys.min()

            # 用轮廓面积 vs 凸包面积判断是否存在大空洞/多连通
            pts_px = (pts * 1000).astype(np.int32).reshape(-1, 1, 2)
            area = cv2.contourArea(pts_px)
            hull = cv2.convexHull(pts_px)
            hull_area = cv2.contourArea(hull)
            ratio = area / (hull_area + 1)

            total_fragments += 1
            if ratio < 0.5 or w_range > 0.5 or h_range > 0.5:
                bad_cases.append((lf.name, ratio, w_range, h_range))

    print(f"\nWH labels: {total_wh}")
    print(f"Suspicious large/disconnected WH labels: {len(bad_cases)}")
    for name, ratio, wr, hr in sorted(bad_cases, key=lambda x: x[1])[:10]:
        print(f"  {name}: ratio={ratio:.2f} w_range={wr:.2f} h_range={hr:.2f}")


if __name__ == "__main__":
    main()
