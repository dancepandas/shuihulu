"""用 v11b 模型跑测试数据几张图, 生成带图例的对比图 (原图 | 预测叠加)。
复用 eval_v10_testdata.py 的推理/可视化函数, 仅换模型路径与输出目录。
"""
import os, sys, glob
import numpy as np, cv2, torch
from transformers import SegformerForSemanticSegmentation

ROOT = r"D:\chengs\9.project\shuihulu"
sys.path.insert(0, os.path.join(ROOT, "segformer", "scripts"))
from eval_v10_testdata import (load_img, predict, save_vis, CLASS_NAMES, CLASS_COLORS)  # noqa

MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11b")
TEST_DIR = os.path.join(ROOT, "data", "测试数据")
OUT_DIR = os.path.join(ROOT, "outputs", "v11b_test_vis")

# 挑几张有水葫芦的代表性图 (222 密集/小目标 + 444 大场景)
PICK = [
    "222/_0047_15V2.jpeg",
    "222/_0077_22V2.jpeg",
    "222/_0162_13V2.jpeg",
    "444/DJI_20260702100342_0010_V.jpeg",
    "444/DJI_20260703070320_0015_V.jpeg",
    "444/DJI_20260730070621_0047_V.jpeg",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval().to(dev)
    print(f"model loaded: {os.path.basename(MODEL_PATH)}, dev={dev}")

    from collections import Counter
    for rel in PICK:
        f = os.path.join(TEST_DIR, rel)
        if not os.path.exists(f):
            print(f"  [skip] 不存在: {rel}"); continue
        img = load_img(f)
        pred = predict(model, img, dev)
        cnt = Counter(pred.ravel())
        wh_px = cnt.get(1, 0)
        out_name = rel.replace("/", "_").replace(".jpeg", ".png")
        save_vis(os.path.join(OUT_DIR, out_name), img, pred, wh_px)
        pct = {c: f"{cnt.get(c,0)/pred.size*100:.1f}%" for c in range(5)}
        print(f"  {rel}: WH={wh_px:,}px  {pct}")

    print(f"\n对比图保存: {OUT_DIR}")


if __name__ == "__main__":
    main()
