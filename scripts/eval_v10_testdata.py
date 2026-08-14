"""v10 模型在测试数据上的可视化验证：
对 data/测试数据/{222,444}/*.jpeg 全部跑推理，输出:
  1. 每张的对比图 (原图 + 预测掩膜叠加)
  2. 水葫芦像素数/面积统计汇总
注意: 中文路径用 np.fromfile + cv2.imdecode 读图 (cv2.imread 在中文路径会失败)
"""
import os, glob, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import json
from collections import Counter
from transformers import SegformerForSemanticSegmentation

ROOT = r"D:\chengs\9.project\shuihulu"
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v10")
TEST_DIR = os.path.join(ROOT, "data", "测试数据")
OUT_DIR = os.path.join(ROOT, "outputs", "v10_test_vis")
SZ = 640
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
               3: "shore_vegetation", 4: "other_aquatic_vegetation"}
CLASS_COLORS = {0: (66, 165, 245), 1: (255, 152, 0), 2: (158, 158, 158),
                3: (67, 160, 71), 4: (171, 71, 188)}
PIXEL_AREA_M2 = (0.0067 * 0.0067)  # 4032x3024 视场~27x20m, 单像素~6.7mm, 面积~0.000045m2


def load_img(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def predict(model, img, dev):
    H, W = img.shape[:2]
    r = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=t).logits, size=(H, W),
                               mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy()


def overlay(img, pred):
    """原图轻度压暗 + 预测掩膜半透明叠加 + 类别描边"""
    out = (img.astype(np.float32) * 0.55)
    H, W = pred.shape[:2]
    for c, col in CLASS_COLORS.items():
        m = pred == c
        if not m.any(): continue
        # 描边
        mu = m.astype(np.uint8)
        er = cv2.erode(mu, np.ones((3, 3), np.uint8))
        bd = (mu - er) > 0
        out[bd] = np.array(col, np.float32) * 1.0
        # 半透明填充
        inner = m & ~bd
        out[inner] = out[inner] * 0.5 + np.array(col, np.float32) * 0.5
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- 中文标签 + 图例 (matplotlib) ----
LABEL_CN = {0: "水体", 1: "水葫芦", 2: "硬结构", 3: "岸基植被", 4: "其他水生植物"}


def save_vis(fig_path, img, pred, wh_px):
    """原图 | 预测叠加 两联, 图下方类别图例 + 文件名"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cnt = Counter(pred.ravel())
    total = pred.size
    ov = overlay(img, pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.2))
    for ax, im, title in [(axes[0], img, "原图"), (axes[1], ov, "LWH-Seg 预测")]:
        ax.imshow(im)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#888"); sp.set_linewidth(0.8)

    # 图例: 所有类别色块 + 占比
    handles = []
    for c in range(5):
        pct = cnt.get(c, 0) / total * 100
        handles.append(Patch(facecolor=np.array(CLASS_COLORS[c]) / 255.0,
                             label=f"{LABEL_CN[c]} {pct:.1f}%"))
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85,
                   ncol=1, title=f"水葫芦 {wh_px:,} px", title_fontsize=10)
    plt.suptitle(os.path.basename(fig_path).replace("_", "  ")[:-4],
                 fontsize=11, color="#444")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval().to(dev)
    print(f"model loaded, dev={dev}")

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*", "*.jpeg")))
    print(f"test images: {len(files)}")

    stats = []
    for i, f in enumerate(files):
        img = load_img(f)
        pred = predict(model, img, dev)
        cnt = Counter(pred.ravel())
        wh_px = cnt.get(1, 0)
        rel = os.path.relpath(f, TEST_DIR).replace("\\", "/")
        # 保存对比图 (matplotlib 带类别图例)
        out_name = rel.replace("/", "_").replace(".jpeg", ".png")
        save_vis(os.path.join(OUT_DIR, out_name), img, pred, wh_px)
        stats.append({"file": rel, "wh_pixels": int(wh_px),
                      "wh_area_m2": round(wh_px * PIXEL_AREA_M2, 3),
                      "classes": {int(k): int(v) for k, v in cnt.items()}})
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {rel}: WH={wh_px}px")

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"pixel_area_m2": PIXEL_AREA_M2, "images": stats},
                  f, ensure_ascii=False, indent=2)

    # 汇总
    total = len(stats)
    with_wh = sum(1 for s in stats if s["wh_pixels"] > 0)
    print(f"\n=== 汇总 ===")
    print(f"测试图: {total} 张, 检出含水葫芦: {with_wh} 张 ({with_wh/total*100:.0f}%)")
    if with_wh:
        wh_areas = [s["wh_pixels"] for s in stats if s["wh_pixels"] > 0]
        print(f"水葫芦像素: min={min(wh_areas):,}  median={sorted(wh_areas)[len(wh_areas)//2]:,}  max={max(wh_areas):,}")
        print(f"水葫芦面积估算: median={sorted(s['wh_pixels'] for s in stats if s['wh_pixels']>0)[len(wh_areas)//2] * PIXEL_AREA_M2:.3f} m2")
    print(f"\n对比图保存: {OUT_DIR}")


if __name__ == "__main__":
    main()