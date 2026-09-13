# -*- coding: utf-8 -*-
"""B5/B6 (CBAM×4 / Triplet×4, 与 TWA 同位等量: 4 个跳跃连接各嵌一个) 的
各尺度空间注意力热图. 布局与 build_vis_analysis 的 TWA 热图一致:
原图 + s1~s4 四行 × 5 场景列.

空间响应取法 (与各模块 forward 内部一致):
  CBAM:    ch 门控精炼后 sigmoid(spatial conv(cat[mean,max])) → (H,W)
  Triplet: 行注意力 × 列注意力 (h_conv ⊗ w_conv, sigmoid 后相乘) → (H,W)
归一化与 TWA 热图相同: resize 原图 → 2%~98% 百分位 → 高斯平滑 (sigma 逐尺度 2/3/5/7).

输出 doc/figs/fig_attention_cbam.png/pdf, fig_attention_triplet.png/pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import build_vis_analysis as bva
from baseline_seg_train_v11 import build_model

CKPTS = {
    "cbam": os.path.join(bva.ROOT, "runs", "baselines",
                         "unet_resnet34_cbam_x4_v11", "best.pt"),
    "triplet": os.path.join(bva.ROOT, "runs", "baselines",
                            "unet_resnet34_triplet_x4_v11", "best.pt"),
}
LABEL = {"cbam": "CBAM", "triplet": "Triplet"}
SIGMAS = {"s1": 2, "s2": 3, "s3": 5, "s4": 7}   # 与 TWA 热图一致


@torch.no_grad()
def x4_scale_maps(model, img_bgr, kind):
    """encoder features[1..4] (H/4,H/8,H/16,H/32) 上各注意力模块的空间响应."""
    x = bva.preprocess(img_bgr)
    feats = model.base.encoder(x)
    out = {}
    for k in (1, 2, 3, 4):
        f = feats[k]
        m = model.attns[k - 1]
        if kind == "cbam":
            avg = F.adaptive_avg_pool2d(f, 1)
            mx = F.adaptive_max_pool2d(f, 1)
            ch = torch.sigmoid(m.channel(avg) + m.channel(mx))
            xc = f * ch
            sp = torch.sigmoid(m.spatial(torch.cat(
                [xc.mean(1, keepdim=True), xc.max(1, keepdim=True)[0]], 1)))
            hm = sp[0, 0].cpu().numpy()
        else:
            h = torch.sigmoid(m.h_conv(f.mean(3, keepdim=True)))   # (B,1,H,1)
            w = torch.sigmoid(m.w_conv(f.mean(2, keepdim=True)))   # (B,1,1,W)
            hm = (h * w)[0, 0].cpu().numpy()
        out[f"s{k}"] = hm
    return out


def main():
    for kind in ("cbam", "triplet"):
        model = build_model("unet", "resnet34", bva.NUM_CLASSES, f"{kind}_x4").to(bva.dev)
        model.load_state_dict(torch.load(CKPTS[kind], map_location=bva.dev))
        model.eval()

        data_list = []
        for fname, desc in bva.SCENES:
            img = bva.load_image(fname)
            H, W = img.shape[:2]
            raw = x4_scale_maps(model, img, kind)
            hms = {k: bva.norm_attn_map(v, W, H, sigma=SIGMAS[k])
                   for k, v in raw.items()}
            data_list.append({"desc": desc, "img": img, "hms": hms})

        name = LABEL[kind]
        specs = [(f"{name}×4 s{i} · H/{4 * 2 ** (i - 1)}",
                  lambda d, i=i: d["hms"][f"s{i}"]) for i in (1, 2, 3, 4)]
        fig = bva.build_attention_figure(
            data_list, specs, cbar_label="空间门控响应    低 → 高")
        p = os.path.join(bva.FIGDIR, f"fig_attention_{kind}.png")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(bva.FIGDIR, f"fig_attention_{kind}.pdf"),
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("saved:", p)


if __name__ == "__main__":
    main()
