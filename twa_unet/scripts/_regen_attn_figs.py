# -*- coding: utf-8 -*-
"""只重生成 3 张注意力热图 (TWA / CBAM×4 / Triplet×4), 不跑 fig_compare 全流程.
样式改动 (jet 色带 + 非加粗标注) 在 build_vis_analysis 中, 此脚本只做最小重绘.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_vis_analysis as bva
import gen_attn_x4_figs as gax


def regen_twa():
    model = bva.build_twaunet()
    data_list = []
    for fname, desc in bva.SCENES:
        img = bva.load_image(fname)
        hms = bva.attention_heatmaps(model, img)
        data_list.append({"desc": desc, "img": img, "hms": hms})
    specs = [(f"TWA s{i} · H/{4 * 2 ** (i - 1)}", lambda d, i=i: d["hms"][f"s{i}"])
             for i in (1, 2, 3, 4)]
    fig = bva.build_attention_figure(data_list, specs)
    p = os.path.join(bva.FIGDIR, "fig_attention_twa.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(bva.FIGDIR, "fig_attention_twa.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


if __name__ == "__main__":
    os.makedirs(bva.FIGDIR, exist_ok=True)
    regen_twa()
    gax.main()
