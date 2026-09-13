# -*- coding: utf-8 -*-
"""重建 doc/figs/结构图/ 两张 SVG（draw.io 原稿备份为 *_drawio_backup.svg）。

TWAU-Net.svg 修复（对照 runs/twa_u_net/model/twa_u_net.py）:
  1. 删除不存在的 Bottleneck H/64 C=1024（ResNet34 到 layer4=H/32·512 即瓶颈）
  2. Decoder 五级标注改为与代码一致: H/16·256 → H/8·128 → H/4·64 → H/2·64 → H·64
  3. 上采样补足 5 次 2×（原 4 次到不了 640×640, 自相矛盾）
  4. 最末两级无跳跃拼接; EB4 的 TWA 输出经 up4 汇入 H/16 级的拼接
  5. 图内说明文字中文化（模块名保留英文）; EB4 注"瓶颈层"
TWA.svg 修复:
  1. (b) 删除残留散字, 补规范的行列编号
  2. (b) 图例双括号改写
  3. (c) 掩码矩阵改为精确掩码（对角 4×4 块内画 sub-triangle, 与代码一致）
  4. (d) 补双残差连线（Post-Norm: ⊕→GN→FFN→⊕→GN）, 输入/输出块对应代码
"""
import base64
import shutil
import os

DIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(DIR), "doc", "figs", "结构图")

FONT = "Times New Roman, SimSun, STSong, serif"
INK = "#1a1a1a"

MARKER = (
    '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
    '<marker id="ahw" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ffffff"/></marker></defs>'
)


def text(x, y, s, size=15, w="bold", fill=INK, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{w}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def line_arrow(x1, y1, x2, y2, width=1.8, stroke=INK, marker="ah"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}" marker-end="url(#{marker})"/>')


def poly_arrow(points, width=1.8, stroke=INK):
    d = " ".join(f"{p[0]},{p[1]}" for p in points)
    return (f'<polyline points="{d}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" marker-end="url(#ah)"/>')


def block(x, y, w, h, fill, lines, tfill="#17334d", note=None, rx=14):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{fill}" stroke="{INK}" stroke-width="1.6"/>']
    n = len(lines) + (1 if note else 0)
    cy = y + h / 2
    step = 24
    y0 = cy - step * (n - 1) / 2
    for i, s in enumerate(lines):
        out.append(text(x + w / 2, y0 + i * step + 5, s, 17, "bold", tfill))
    if note:
        out.append(text(x + w / 2, y0 + n * step - 6, note, 12, "normal", tfill, ))
    return "".join(out)


def concat(cx, cy, label="拼接"):
    return (f'<circle cx="{cx}" cy="{cy}" r="20" fill="#ffffff" stroke="{INK}" '
            f'stroke-width="1.8"/>'
            f'<line x1="{cx-10}" y1="{cy}" x2="{cx+10}" y2="{cy}" stroke="{INK}" stroke-width="1.8"/>'
            f'<line x1="{cx}" y1="{cy-10}" x2="{cx}" y2="{cy+10}" stroke="{INK}" stroke-width="1.8"/>'
            + text(cx - 26, cy + 42, label, 12.5, "normal", "#555555", "end"))


# ============================================================ TWAU-Net.svg
def gen_twaunet():
    with open(os.path.join(FIGDIR, "_embed0.png"), "rb") as f:
        img_in = base64.b64encode(f.read()).decode()
    with open(os.path.join(FIGDIR, "_embed1.png"), "rb") as f:
        img_out = base64.b64encode(f.read()).decode()

    W, H = 1400, 1075
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', MARKER,
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    # ---- 输入影像 (复用原稿嵌入图) ----
    p.append(f'<image x="50" y="40" width="240" height="180" '
             f'xlink:href="data:image/png;base64,{img_in}" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="none"/>')
    p.append(text(170, 248, "输入 640×640×3", 15.5))
    p.append(line_arrow(170, 258, 170, 362))
    p.append(text(184, 305, "stem：7×7 conv s2 + maxpool（↓4×）", 12.5, "normal",
                  "#666666", "start"))

    # ---- 编码器 ----
    p.append(f'<rect x="20" y="350" width="310" height="670" rx="14" fill="none" '
             f'stroke="#999999" stroke-width="1.6" stroke-dasharray="7 5"/>')
    # 白色描边光晕遮住穿过的箭头 (paint-order: 描边在填充之下)
    p.append(f'<text x="175" y="342" font-family="{FONT}" font-size="17" '
             f'font-weight="bold" fill="{INK}" text-anchor="middle" '
             f'stroke="#ffffff" stroke-width="9" paint-order="stroke" '
             f'stroke-linejoin="round">ResNet34 编码器</text>')
    rows = [(370, "Encoder Block 1", "H/4 · C=64", "#b9d2e8", "#17334d", None),
            (540, "Encoder Block 2", "H/8 · C=128", "#8caee0", "#17334d", None),
            (710, "Encoder Block 3", "H/16 · C=256", "#5b8db8", "#ffffff", None),
            (880, "Encoder Block 4", "H/32 · C=512", "#35506e", "#ffffff", "（瓶颈层）")]
    centers = []
    for y, l1, l2, fill, tf, note in rows:
        p.append(block(40, y, 280, 130, fill, [l1, l2], tf, note))
        centers.append(y + 65)

    # ---- TWA 块 + 跳连箭头 ----
    for cy in centers:
        p.append(line_arrow(320, cy, 466, cy))
        p.append(block(470, cy - 28, 100, 56, "#e8833a", ["TWA"], "#ffffff", rx=10))

    # ---- 解码器 ----
    p.append(f'<rect x="685" y="25" width="270" height="830" rx="14" fill="none" '
             f'stroke="#999999" stroke-width="1.6" stroke-dasharray="7 5"/>')
    p.append(text(820, 18, "解码器", 17))
    # (y, line1, line2, fill, textcolor, note)
    decs = [(370, "ConvBlock", "H/4 · C=64", "#8abf99", "#14301f", None),      # dec2
            (540, "ConvBlock", "H/8 · C=128", "#6e9e7e", "#ffffff", None),     # dec3
            (710, "ConvBlock", "H/16 · C=256", "#3a734d", "#ffffff", None),    # dec4
            (200, "ConvBlock", "H/2 · C=64", "#a9d1b4", "#14301f", "（无拼接）"),  # dec1
            (40, "特征图", "H · C=64", "#c9e4cf", "#14301f", "（up0 上采样输出，无卷积块）")]  # d0
    for y, l1, l2, fill, tf, note in decs:
        p.append(block(700, y, 240, 130, fill, [l1, l2], tf, note))

    c1, c2, c3, c4 = centers  # 435 / 605 / 775 / 945

    # ---- 拼接与主链箭头 ----
    # 行 1-3: TWA -> 拼接 -> ConvBlock
    for cy in (c1, c2, c3):
        p.append(line_arrow(570, cy, 616, cy))
        p.append(concat(640, cy))
        p.append(line_arrow(660, cy, 696, cy))
    # 行 4: TWA4 输出经 up4 上采样汇入行 3 拼接
    p.append(poly_arrow([(570, c4), (640, c4), (640, c3 + 23)]))
    p.append(text(652, 880, "上采样 2×", 13, "bold", "#333333", "start"))
    # 解码器内部上采样链 (自下而上)
    p.append(poly_arrow([(820, 710), (820, 690), (640, 690), (640, c2 + 23)]))   # dec4→⊕2
    p.append(text(652, 678, "上采样 2×", 13, "bold", "#333333", "start"))
    p.append(poly_arrow([(820, 540), (820, 520), (640, 520), (640, c1 + 23)]))   # dec3→⊕1
    p.append(text(652, 508, "上采样 2×", 13, "bold", "#333333", "start"))
    p.append(line_arrow(820, 370, 820, 334))   # dec2 → dec1
    p.append(text(832, 356, "上采样 2×", 13, "bold", "#333333", "start"))
    p.append(line_arrow(820, 200, 820, 174))   # dec1 → d0
    p.append(text(832, 191, "上采样 2×", 13, "bold", "#333333", "start"))
    # d0 -> 分割头 -> 输出掩膜
    p.append(line_arrow(940, 105, 986, 105))
    p.append(block(990, 73, 150, 64, "#fceeba", ["分割头", "1×1 Conv"], "#5a4a10", rx=10))
    p.append(line_arrow(1140, 105, 1156, 105))
    p.append(f'<image x="1160" y="15" width="240" height="180" '
             f'xlink:href="data:image/png;base64,{img_out}" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="none"/>')
    p.append(text(1280, 222, "输出掩膜 640×640×5", 15.5))

    # ---- 图例 (双栏: 分割类别 | 符号) ----
    p.append(f'<rect x="1000" y="790" width="360" height="252" rx="12" fill="none" '
             f'stroke="#999999" stroke-width="1.4" stroke-dasharray="6 4"/>')
    p.append(text(1180, 818, "图例", 17))
    p.append('<line x1="1184" y1="830" x2="1184" y2="1028" stroke="#cccccc" stroke-width="1"/>')
    # 左栏: 分割类别
    p.append(text(1016, 846, "分割类别", 13.5, "bold", "#777777", "start"))
    legend = [("水面", "#4c90c0"), ("水葫芦", "#d94f3d"), ("硬质结构", "#9a9a9a"),
              ("岸边植被", "#5fa05a"), ("其他水生植被", "#d9c24a")]
    for i, (name, col) in enumerate(legend):
        y = 858 + i * 36
        p.append(f'<rect x="1032" y="{y}" width="30" height="22" fill="{col}" '
                 f'stroke="#333333" stroke-width="0.8"/>')
        p.append(text(1078, y + 17, name, 15, "normal", INK, "start"))
    # 右栏: 符号
    p.append(text(1204, 846, "符号", 13.5, "bold", "#777777", "start"))
    ys = [858, 894, 930, 966]
    p.append(f'<rect x="1204" y="{ys[0]}" width="30" height="22" rx="6" fill="#e8833a" '
             f'stroke="{INK}" stroke-width="1"/>')
    p.append(text(1219, ys[0] + 16, "TWA", 10, "bold", "#ffffff"))
    p.append(text(1250, ys[0] + 17, "三角窗注意力", 14, "normal", INK, "start"))
    cy0 = ys[1] + 11
    p.append(f'<circle cx="1219" cy="{cy0}" r="11" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>')
    p.append(f'<line x1="1212" y1="{cy0}" x2="1226" y2="{cy0}" stroke="{INK}" stroke-width="1.4"/>')
    p.append(f'<line x1="1219" y1="{cy0-6}" x2="1219" y2="{cy0+6}" stroke="{INK}" stroke-width="1.4"/>')
    p.append(text(1250, ys[1] + 17, "跳跃特征拼接", 14, "normal", INK, "start"))
    p.append(line_arrow(1204, ys[2] + 11, 1240, ys[2] + 11, 2))
    p.append(text(1250, ys[2] + 17, "上采样 2×", 14, "normal", INK, "start"))
    p.append(f'<rect x="1204" y="{ys[3]}" width="30" height="22" rx="6" fill="#3a734d" '
             f'stroke="{INK}" stroke-width="1"/>')
    p.append(text(1250, ys[3] + 17, "解码器卷积块", 14, "normal", INK, "start"))

    p.append("</svg>")
    with open(os.path.join(FIGDIR, "TWAU-Net.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(p))


# ============================================================ TWA.svg
def gen_twa():
    W, H = 1200, 1310
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">', MARKER,
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    def band(y0, title):
        p.append(f'<rect x="0" y="{y0}" width="{W}" height="52" fill="#e9e9e9"/>')
        p.append(text(600, y0 + 35, title, 20))

    def swatch(x, y, fill, label, stroke="#1a3a5c", tfill=INK, arrow=False):
        if arrow:
            p.append(f'<line x1="{x}" y1="{y+11}" x2="{x+44}" y2="{y+11}" '
                     f'stroke="{INK}" stroke-width="3" marker-end="url(#ah)"/>')
        else:
            p.append(f'<rect x="{x}" y="{y}" width="30" height="22" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1"/>')
        p.append(text(x + 58, y + 16, label, 15, "bold", tfill, "start"))

    # ---------------- (a) 窗口划分 ----------------
    band(8, "(a) 窗口划分（8×8 特征图 → 4 个 4×4 窗口）")
    gx, gy, cs = 200, 80, 28
    for r in range(8):
        for c in range(8):
            col = "#eaf1fb" if ((r // 4 + c // 4) % 2 == 0) else "#fdf3dc"
            p.append(f'<rect x="{gx+c*cs}" y="{gy+r*cs}" width="{cs}" height="{cs}" '
                     f'fill="{col}" stroke="#b0b8c4" stroke-width="0.8"/>')
    for k in (0, 4, 8):
        w = 2.5 if k in (0, 8) else 2.2
        col = "#1a1a1a" if k in (0, 8) else "#1a3a5c"
        p.append(f'<line x1="{gx+k*cs}" y1="{gy}" x2="{gx+k*cs}" y2="{gy+8*cs}" '
                 f'stroke="{col}" stroke-width="{w}"/>')
        p.append(f'<line x1="{gx}" y1="{gy+k*cs}" x2="{gx+8*cs}" y2="{gy+k*cs}" '
                 f'stroke="{col}" stroke-width="{w}"/>')
    for i in range(8):
        p.append(text(gx + i * cs + cs / 2, gy - 8, str(i + 1), 12.5, "normal", "#555555"))
        p.append(text(gx - 10, gy + i * cs + cs / 2 + 4, str(i + 1), 12.5, "normal",
                      "#555555", "end"))
    lx = 580
    swatch(lx, 96, "#eaf1fb", "窗口 1（行 1–4，列 1–4）")
    swatch(lx, 136, "#fdf3dc", "窗口 2（行 1–4，列 5–8）")
    swatch(lx, 176, "#fdf3dc", "窗口 3（行 5–8，列 1–4）")
    swatch(lx, 216, "#eaf1fb", "窗口 4（行 5–8，列 5–8）")

    # ---------------- (b) 下三角聚合 ----------------
    band(338, "(b) 单个 4×4 窗口内的下三角聚合")
    gx, gy, cs = 200, 410, 56
    for r in range(4):
        for c in range(4):
            fill = "#ffe6cc" if (r <= 2 and c <= 1) else "#ffffff"
            p.append(f'<rect x="{gx+c*cs}" y="{gy+r*cs}" width="{cs}" height="{cs}" '
                     f'fill="{fill}" stroke="#333333" stroke-width="1"/>')
    p.append(f'<rect x="{gx+cs}" y="{gy+2*cs}" width="{cs}" height="{cs}" '
             f'fill="#fa6800" stroke="#333333" stroke-width="1"/>')  # Query(3,2)
    for i in range(4):
        p.append(text(gx + i * cs + cs / 2, gy - 10, str(i + 1), 13, "normal", "#555555"))
        p.append(text(gx - 10, gy + i * cs + cs / 2 + 4, str(i + 1), 13, "normal",
                      "#555555", "end"))
    # 聚合方向: 左上方 key 信息汇入 Query (起点/终点均在网格内)
    p.append(line_arrow(gx + 0.12 * cs, gy + 0.12 * cs, gx + 1.15 * cs, gy + 2.15 * cs, 3))
    lx = 580
    swatch(lx, 426, "#fa6800", "Query token（第 3 行，第 2 列）")
    swatch(lx, 466, "#ffe6cc", "Key tokens：左上方上下文（含 Query 自身）")
    swatch(lx, 506, "#ffffff", "未参与聚合的 token")
    swatch(lx, 546, None, "Query 仅聚合左上方 token 的信息", arrow=True)

    # ---------------- (c) 16×16 掩码矩阵 (精确掩码) ----------------
    band(666, "(c) 4×4 窗口的注意力掩码矩阵（16×16）")
    mx, my, ms = 200, 740, 16
    for qi in range(16):
        Rb, rb = qi // 4, qi % 4          # query (r_i, c_i)
        for ki in range(16):
            Cb, cb = ki // 4, ki % 4      # key (r_j, c_j)
            # 与 twa_block.triangular_token_mask 一致: 可见 iff r_j≤r_i 且 c_j≤c_i
            allowed = (Cb <= Rb) and (cb <= rb)
            fill = "#ffffff" if allowed else "#3b3b45"
            p.append(f'<rect x="{mx+ki*ms}" y="{my+qi*ms}" width="{ms}" height="{ms}" '
                     f'fill="{fill}" stroke="#999999" stroke-width="0.5"/>')
    for k in (0, 4, 8, 12, 16):
        p.append(f'<line x1="{mx+k*ms}" y1="{my}" x2="{mx+k*ms}" y2="{my+256}" '
                 f'stroke="#1a3a5c" stroke-width="1.6"/>')
        p.append(f'<line x1="{mx}" y1="{my+k*ms}" x2="{mx+256}" y2="{my+k*ms}" '
                 f'stroke="#1a3a5c" stroke-width="1.6"/>')
    p.append(text(mx + 128, my - 8, "Key 索引（j）", 13, "normal", "#555555"))
    p.append(f'<text x="{mx-14}" y="{my+128}" font-family="{FONT}" font-size="13" '
             f'fill="#555555" text-anchor="middle" '
             f'transform="rotate(-90 {mx-14} {my+128})">Query 索引（i）</text>')
    p.append(text(mx + 128, my + 280, "块下三角×块内子三角：Query i 仅与自身及左上方 token 相连（屏蔽位置 softmax 前置 −∞）",
                  13.5, "normal", "#555555"))
    lx = 580
    swatch(lx, 756, "#ffffff", "0：允许参与注意力", stroke="#999999")
    swatch(lx, 796, "#3b3b45", "−∞：softmax 后权重为 0", stroke="#3b3b45")

    # ---------------- (d) 模块流程 (双残差 Post-Norm) ----------------
    band(1038, "(d) TWA 模块流程（双残差，Post-Norm）")
    by, bh = 1130, 110
    boxes = [
        (30, 130, "#8caee0", "#17334d", "输入", "特征图 H×W×C"),
        (200, 130, "#a9d1b4", "#14301f", "窗口划分", "M×M 窗口"),
        (370, 160, "#fa6800", "#ffffff", "三角掩码", "多头注意力"),
        (596, 110, "#e6d6f2", "#4a2a6a", "GroupNorm", "组归一化"),
        (726, 110, "#d6e4f7", "#17334d", "FFN", "前馈网络"),
        (902, 110, "#e6d6f2", "#4a2a6a", "GroupNorm", "组归一化"),
        (1040, 130, "#fceeba", "#5a4a10", "输出", "精炼特征图"),
    ]
    for x, w, fill, tf, l1, l2 in boxes:
        p.append(block(x, by, w, bh, fill, [l1, l2], tf, rx=12))
    for cx in (560, 860):
        p.append(f'<circle cx="{cx}" cy="{by+bh/2}" r="16" fill="#ffffff" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
        p.append(f'<line x1="{cx-8}" y1="{by+bh/2}" x2="{cx+8}" y2="{by+bh/2}" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
        p.append(f'<line x1="{cx}" y1="{by+bh/2-8}" x2="{cx}" y2="{by+bh/2+8}" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
    cy = by + bh / 2
    chain = [(160, 200), (330, 370), (530, 544), (576, 596), (706, 726),
             (836, 844), (876, 902), (1012, 1040)]
    for x1, x2 in chain:
        p.append(line_arrow(x1, cy, x2, cy))
    # 双残差 (直角折线: 主链打点 → 上方水平车道 → 落入 ⊕ 顶部; 车道避开标题条)
    lane = 1106          # 灰条底 1090 与盒顶 1130 之间
    for x_tap, cx in ((350, 560), (716, 860)):
        p.append(f'<circle cx="{x_tap}" cy="{cy}" r="3.2" fill="{INK}"/>')
        p.append(f'<polyline points="{x_tap},{cy} {x_tap},{lane} {cx},{lane} {cx},{cy-16}" '
                 f'fill="none" stroke="{INK}" stroke-width="1.6" '
                 f'stroke-dasharray="5 4" marker-end="url(#ah)"/>')

    p.append("</svg>")
    with open(os.path.join(FIGDIR, "TWA.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(p))


def main():
    for name in ("TWAU-Net.svg", "TWA.svg"):
        src = os.path.join(FIGDIR, name)
        bak = os.path.join(FIGDIR, name.replace(".svg", "_drawio_backup.svg"))
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f"backup: {bak}")
    gen_twaunet()
    gen_twa()
    for tmp in ("_embed0.png", "_embed1.png"):
        os.remove(os.path.join(FIGDIR, tmp))
    print("regenerated: TWAU-Net.svg, TWA.svg")


if __name__ == "__main__":
    main()
