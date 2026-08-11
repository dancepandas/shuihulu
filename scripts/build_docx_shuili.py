# -*- coding: utf-8 -*-
"""
《水利水电技术（中英文）》投稿稿生成脚本
按期刊模板版式将 doc/论文_基于SegFormer-B2优化的轻量化水葫芦语义分割.md 转成 Word：
  - 标题小一黑体 / 作者4号楷体 / 单位5号仿宋
  - 摘要、关键词 5号黑体标签 + 5号楷体内容（结构化【目的】【方法】【结果】【结论】）
  - 中图分类号 + 文献标志码
  - 英文标题/作者/单位/结构化Abstract/Keywords
  - 章节重排：0引言,1相关工作,2本文方法,3实验与分析,4讨论,5结论
  - 一级4号宋体/二级5号黑体/正文5号宋体单倍行距
  - 图下方中英题名(小五)；三线表(表题小五黑体/内容六号宋体)；表注六号
  - 公式居中+右对齐编号；参考文献6号中英对照
"""
import re, os
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = r"D:\chengs\9.project\shuihulu"
MD = os.path.join(ROOT, "doc", "论文_基于SegFormer-B2优化的轻量化水葫芦语义分割.md")
FIGDIR = os.path.join(ROOT, "doc", "figs")
OUT = os.path.join(ROOT, "doc", "论文_水利水电技术投稿稿.docx")

# ---------------- 基础工具 ----------------
def set_run(r, cn="宋体", en="Times New Roman", size=10.5, bold=False, italic=False,
            sub=False, sup=False):
    r.font.name = en
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if sub: r.font.subscript = True
    if sup: r.font.superscript = True
    rPr = r._element.get_or_add_rPr()
    rf = rPr.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), cn)

def para(doc, align=None, size=10.5, indent_pt=0, space=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(space); pf.space_after = Pt(space)
    if indent_pt: pf.first_line_indent = Pt(indent_pt)
    if align is not None: p.alignment = align
    return p

# ---------------- 数学渲染（LaTeX 子集 → runs） ----------------
SYM = {"alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε","varepsilon":"ε",
       "theta":"θ","lambda":"λ","mu":"μ","sigma":"σ","phi":"φ","Omega":"Ω","Sigma":"Σ",
       "Delta":"Δ","cdot":"·","times":"×","sum":"Σ","prod":"∏","in":"∈","infty":"∞",
       "pi":"π","pm":"±","leq":"≤","geq":"≥","neq":"≠","approx":"≈","partial":"∂"}
UPRIGHT = {"log","min","max","sin","cos","tan","exp","Re","Im","arg","const","mod","sgn",
           "lim","inf","sup"}

def read_group(tex, i):
    """tex[i]=='{' → (inner, next_i)"""
    d = 0; j = i
    while j < len(tex):
        if tex[j] == "{": d += 1
        elif tex[j] == "}":
            d -= 1
            if d == 0: return tex[i+1:j], j+1
        j += 1
    return tex[i+1:], len(tex)

def render_math(p, tex, size, bold=False, sub=False, sup=False):
    """把 LaTeX 子集渲染成一段 runs。"""
    def add(text, italic=True, s=False, u=False):
        if not text: return
        r = p.add_run(text)
        set_run(r, cn="Times New Roman", en="Times New Roman", size=size, bold=bold,
                italic=italic, sub=(sub or s), sup=(sup or u))
    i, n, acc = 0, len(tex), ""
    def flush(italic=True):
        nonlocal acc
        if acc:
            add(acc, italic=italic); acc = ""
    while i < n:
        c = tex[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", tex[i:])
            if m:
                cmd = m.group(1); i += len(m.group(0))
                if cmd in UPRIGHT:
                    flush(); add(cmd, italic=False)
                elif cmd in ("text", "mathbb"):
                    content, j = read_group(tex, i); i = j
                    flush(); add(content, italic=False)
                elif cmd == "frac":
                    flush()
                    num, j = read_group(tex, i)
                    den, k = read_group(tex, j); i = k
                    render_math(p, num, size, bold, sub=sub, sup=sup)
                    add("/", italic=False)
                    render_math(p, den, size, bold, sub=sub, sup=sup)
                elif cmd in SYM:
                    flush(); add(SYM[cmd], italic=True)
                else:
                    pass  # \left \right \tag 等忽略
                continue
            else:
                i += 1; continue
        elif c == "_":
            i += 1
            if i < n and tex[i] == "{":
                content, j = read_group(tex, i); i = j
            else:
                content = tex[i]; i += 1
            flush(); render_math(p, content, size, bold, sub=True, sup=sup)
        elif c == "^":
            i += 1
            if i < n and tex[i] == "{":
                content, j = read_group(tex, i); i = j
            else:
                content = tex[i]; i += 1
            flush(); render_math(p, content, size, bold, sub=sub, sup=True)
        else:
            acc += c; i += 1
    flush()

# ---------------- 行内格式（**粗体** 与 $数学$） ----------------
INLINE = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*)")

def add_inline(p, text, size=10.5, cn="宋体", en="Times New Roman", bold=False):
    for tok in INLINE.split(text):
        if not tok: continue
        if tok.startswith("$"):
            render_math(p, tok[1:-1].strip(), size, bold)
        elif tok.startswith("**"):
            r = p.add_run(tok[2:-2]); set_run(r, cn=cn, en=en, size=size, bold=True)
        else:
            r = p.add_run(tok); set_run(r, cn=cn, en=en, size=size, bold=bold)

# ---------------- 章节重排 ----------------
H1_MAP = {"1 引言": "0 引  言", "2 数据与方法": "1 数据与方法",
          "3 实验结果与分析": "2 实验结果与分析", "4 讨论": "3 讨论",
          "5 结论": "4 结论"}
SUB_MAP = {"2.1":"1.1","2.2":"1.2","2.3":"1.3","2.4":"1.4","2.5":"1.5","2.6":"1.6",
           "3.1":"2.1","3.2":"2.2","3.3":"2.3","3.4":"2.4","3.5":"2.5","3.6":"2.6",
           "4.1":"3.1","4.2":"3.2"}

def ren_h1(t): return H1_MAP.get(t, t)
def ren_h2(t):
    for k, v in SUB_MAP.items():
        if t.startswith(k + " "):
            return v + " " + t[len(k)+1:]
    return t

# ---------------- 图 / 表 ----------------
PANEL_CN = "（(a) 原图；(b) 人工真值；(c) LWH-Seg 预测；(d) SegFormer-B2-FT 预测）"
PANEL_EN = " ((a) original image; (b) manual ground truth; (c) LWH-Seg prediction; (d) SegFormer-B2-FT prediction)"
FIG_CAP = {
    "fig_framework.png": ("图1 LWH-Seg 方法总体框架", "Fig.1 Overall framework of the proposed LWH-Seg method"),
    "fig_qual_1.png": ("图2 密集分布场景分割对比" + PANEL_CN, "Fig.2 Segmentation comparison in dense distribution scenes" + PANEL_EN),
    "fig_qual_2.png": ("图3 稀疏分布场景分割对比" + PANEL_CN, "Fig.3 Segmentation comparison in sparse distribution scenes" + PANEL_EN),
    "fig_qual_3.png": ("图4 小目标场景分割对比" + PANEL_CN, "Fig.4 Segmentation comparison in small-target scenes" + PANEL_EN),
    "fig_qual_4.png": ("图5 岸基植被干扰场景分割对比" + PANEL_CN, "Fig.5 Segmentation comparison in riverside vegetation interference scenes" + PANEL_EN),
}
FIG_W = {"fig_framework.png": 15.5, "fig_qual_1.png": 16.5, "fig_qual_2.png": 16.5,
         "fig_qual_3.png": 16.5, "fig_qual_4.png": 16.5}
TAB_CAP = {
    1: ("表1 LWH-Seg 训练配置", "Table 1 Training configuration of LWH-Seg"),
    2: ("表2 各方法在水葫芦验证集上的分割精度与效率对比",
        "Table 2 Comparison of segmentation accuracy and efficiency on the water hyacinth validation set"),
    3: ("表3 LWH-Seg 逐类分割结果", "Table 3 Per-class segmentation results of LWH-Seg"),
    4: ("表4 消融实验结果", "Table 4 Ablation experiment results"),
}

def add_caption(doc, cn_txt, en_txt):
    p1 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p1.add_run(cn_txt); set_run(r, cn="黑体", en="Times New Roman", size=9)
    p2 = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p2.add_run(en_txt); set_run(r, cn="Times New Roman", en="Times New Roman", size=9)

def add_figure(doc, fname):
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run()
    r.add_picture(os.path.join(FIGDIR, fname), width=Cm(FIG_W.get(fname, 15.0)))
    cn_cap, en_cap = FIG_CAP[fname]
    add_caption(doc, cn_cap, en_cap)

def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    tb = OxmlElement("w:tcBorders")
    for edge, sz in kw.items():
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "000000")
        tb.append(el)
    tcPr.append(tb)

def add_table(doc, rows, cap_line):
    m = re.search(r"表\s*(\d)", cap_line)
    tnum = int(m.group(1)) if m else 1
    # 去掉分隔行，解析单元格
    data = []
    for row in rows:
        cells = [x.strip() for x in row.strip().strip("|").split("|")]
        if cells and re.match(r"^[-:\s]+$", cells[0]):
            continue
        data.append(cells)
    cn_cap, en_cap = TAB_CAP[tnum]
    add_caption(doc, cn_cap, en_cap)
    ncol = max(len(r) for r in data)
    tbl = doc.add_table(rows=len(data), cols=ncol)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    for ri, row in enumerate(data):
        for ci in range(ncol):
            cell = tbl.rows[ri].cells[ci]
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
            txt = row[ci] if ci < len(row) else ""
            is_hdr = (ri == 0)
            for tok in INLINE.split(txt):
                if not tok: continue
                if tok.startswith("**"):
                    rr = cell.paragraphs[0].add_run(tok[2:-2])
                    set_run(rr, cn="黑体" if is_hdr else "宋体", en="Times New Roman",
                            size=7.5, bold=True)
                else:
                    rr = cell.paragraphs[0].add_run(tok)
                    set_run(rr, cn="黑体" if is_hdr else "宋体", en="Times New Roman",
                            size=7.5, bold=is_hdr)
    # 三线表：首行上1.5pt/下0.5pt，末行下1.5pt
    for c in tbl.rows[0].cells:
        set_cell_border(c, top=12, bottom=4)
    for c in tbl.rows[-1].cells:
        set_cell_border(c, bottom=12)

def add_note(doc, text):
    p = para(doc, indent_pt=15)
    r = p.add_run(text); set_run(r, cn="宋体", en="Times New Roman", size=7.5)

def add_equation(doc, latex):
    m = re.search(r"\\tag\{(\d+)\}", latex)
    tag = m.group(1) if m else ""
    body = re.sub(r"\s*\\tag\{\d+\}\s*", "", latex).strip()
    p = para(doc)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT)
    render_math(p, body, 10.5)
    if tag:
        p.add_run("\t")
        r = p.add_run("(%s)" % tag); set_run(r, cn="宋体", en="Times New Roman", size=10.5)

# ---------------- 摘要分段 ----------------
def split_abs(text, markers):
    out = []
    pos = 0
    for mark in markers:
        idx = text.find(mark)
        if idx >= 0:
            seg = text[pos:idx + len(mark)].strip()
            if seg: out.append(seg)
            pos = idx + len(mark)
    tail = text[pos:].strip()
    if tail: out.append(tail)
    return out

CN_MARK = ["难以满足实时监测与边缘部署需求。",
           "兼顾水葫芦识别与干扰地物剔除。",
           "两阶段方案提升超过两个数量级。"]
EN_MARK = ["making real-time monitoring and edge deployment difficult.",
           "covering both water hyacinth recognition and interference-object exclusion.",
           "improved by more than two orders of magnitude over the two-stage \"detection + fine segmentation\" scheme."]
CN_TAG = ["【目的】", "【方法】", "【结果】", "【结论】"]
EN_TAG = ["[Objective]", "[Methods]", "[Results]", "[Conclusion]"]

# ---------------- 参考文献 中英对照 ----------------
REF_EN = {
    1: "JIANG Hongtao, ZHANG Hongmei. A review of controlling common waterhyacinth at home and abroad[J]. Journal of Agricultural Science and Technology, 2003, 5(3): 72-75.",
    2: "YANG Fenghui, MA Tao, CHEN Jiakuan, et al. Preliminary study on the occurrence mechanism and control countermeasures of Eichhornia crassipes disaster in Huangpu River, Shanghai[J]. Journal of Fudan University (Natural Science), 2002, 41(6): 599-603.",
    8: "WANG Tianhao. Research on water hyacinth object detection method based on YOLOv7[D]. Shandong Agricultural University, 2024.",
    16: "ZHU Bao. Intelligent recognition method and prototype system development of water hyacinth based on ground-based surveillance images[D]. East China Normal University, 2024.",
}

# ---------------- 主流程 ----------------
def main():
    md = open(MD, encoding="utf-8").read()
    doc = Document()
    # 页面与默认字体
    for sec in doc.sections:
        sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
        sec.top_margin = sec.bottom_margin = Cm(2.5)
        sec.left_margin = sec.right_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"; st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # ---- 中文标题 / 作者 / 单位 ----
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1).strip()
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run(title); set_run(r, cn="黑体", en="Times New Roman", size=24, bold=True)
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("作者姓名1，作者姓名2，……（请填写）"); set_run(r, cn="楷体", size=14)
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("（1. ××××大学 ××××学院，×× 邮编；2. ……）（请填写）"); set_run(r, cn="仿宋", size=10.5)

    # ---- 中文摘要 / 关键词 ----
    m_abs = re.search(r"## 摘要\n+(.*?)\n+---", md, re.S)
    abs_block = m_abs.group(1)
    abs_lines = [l.strip() for l in abs_block.split("\n") if l.strip()]
    cn_abs = "".join(abs_lines[:-1])
    kw_cn = abs_lines[-1]
    segs = split_abs(cn_abs, CN_MARK)
    p = para(doc)
    r = p.add_run("摘  要："); set_run(r, cn="黑体", size=10.5, bold=True)
    for tag, seg in zip(CN_TAG, segs):
        r = p.add_run(tag); set_run(r, cn="楷体", size=10.5, bold=True)
        r = p.add_run(seg); set_run(r, cn="楷体", size=10.5)
    p = para(doc)
    r = p.add_run("关键词："); set_run(r, cn="黑体", size=10.5, bold=True)
    kw_content = kw_cn.split("：", 1)[1].strip()
    r = p.add_run(kw_content); set_run(r, cn="楷体", size=10.5)
    p = para(doc)
    r = p.add_run("中图分类号：TP391.41　　文献标志码：A"); set_run(r, cn="宋体", size=9)

    # ---- 英文标题 / 作者 / 单位 / Abstract / Keywords ----
    en_title_txt = "Lightweight Semantic Segmentation of Water Hyacinth in Rivers "
    en_title_txt += "Based on Optimized SegFormer-B2"
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run(en_title_txt); set_run(r, cn="Times New Roman", en="Times New Roman", size=14, bold=True)
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("Author Name1, Author Name2, ... (please fill)"); set_run(r, size=10.5)
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("(1. ×××× University, ×××× College, Beijing 100000, China; ...)"); set_run(r, size=9)

    en_abs = re.search(r"\*\*Abstract\*\*:?\s*(.*?)\n\s*\n\s*\*\*Keywords", md, re.S).group(1).strip()
    en_segs = split_abs(en_abs, EN_MARK)
    p = para(doc)
    r = p.add_run("Abstract: "); set_run(r, en="Times New Roman", size=10.5, bold=True)
    for tag, seg in zip(EN_TAG, en_segs):
        r = p.add_run(tag + " "); set_run(r, en="Times New Roman", size=10.5, bold=True)
        r = p.add_run(seg + " "); set_run(r, en="Times New Roman", size=10.5)
    en_kw = re.search(r"\*\*Keywords\*\*:?\s*(.*?)\n\s*\n\s*---", md, re.S).group(1).strip()
    p = para(doc)
    r = p.add_run("Keywords: "); set_run(r, en="Times New Roman", size=10.5, bold=True)
    r = p.add_run(en_kw); set_run(r, en="Times New Roman", size=10.5)

    # ---- 基金/作者简介 占位（小5号） ----
    for note in ["基金项目：××××××（请填写）",
                 "作者简介：×××（出生年—），性别，职称，学位，主要从事×××研究。E-mail：",
                 "通信作者：×××（出生年—），性别，职称，学位，主要从事×××研究。E-mail："]:
        p = para(doc)
        r = p.add_run(note); set_run(r, cn="宋体", size=9)

    # ---- 正文 ----
    body_src = re.search(r"(## 1 引言\n.*?)\n## 参考文献", md, re.S).group(1)
    body_lines = body_src.split("\n")
    i, n = 0, len(body_lines)
    while i < n:
        s = body_lines[i].strip()
        if not s or s == "---":
            i += 1; continue
        m1 = re.match(r"^## (.*)$", s)
        if m1:
            p = para(doc); r = p.add_run(ren_h1(m1.group(1).strip()))
            set_run(r, cn="宋体", size=14, bold=True)
            i += 1; continue
        m2 = re.match(r"^### (.*)$", s)
        if m2:
            p = para(doc); r = p.add_run(ren_h2(m2.group(1).strip()))
            set_run(r, cn="黑体", size=10.5)
            i += 1; continue
        if s.startswith("$$"):
            eq = s[2:].strip()
            if eq.endswith("$$"): eq = eq[:-2].strip()
            add_equation(doc, eq)
            i += 1; continue
        mg = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if mg:
            add_figure(doc, os.path.basename(mg.group(2)))
            i += 1; continue
        if s.startswith("|"):
            j = i; rows = []
            while j < n and body_lines[j].strip().startswith("|"):
                rows.append(body_lines[j].strip()); j += 1
            cap = ""
            k = i - 1
            while k >= 0 and body_lines[k].strip() == "":
                k -= 1
            if k >= 0 and re.match(r"^\*\*表\s*\d", body_lines[k].strip()):
                cap = body_lines[k].strip()
            add_table(doc, rows, cap)
            i = j; continue
        if s.startswith(">"):
            add_note(doc, s[1:].strip())
            i += 1; continue
        if s.startswith("**表"):
            jj = i + 1
            while jj < n and body_lines[jj].strip() == "":
                jj += 1
            if jj < n and body_lines[jj].strip().startswith("|"):
                i += 1; continue  # 表题由表格块消费
        p = para(doc, indent_pt=21)
        add_inline(p, s, size=10.5, cn="宋体", en="Times New Roman")
        i += 1

    # ---- 参考文献（6号 中英对照） ----
    ref_src = re.search(r"## 参考文献\n(.*?)(\n---|\Z)", md, re.S).group(1)
    p = para(doc); r = p.add_run("参考文献")
    set_run(r, cn="宋体", size=14, bold=True)
    for line in ref_src.split("\n"):
        line = line.strip()
        if not line: continue
        mm = re.match(r"^\[(\d+)\]", line)
        if not mm: continue
        p = para(doc)
        r = p.add_run(line); set_run(r, cn="宋体", en="Times New Roman", size=7.5)
        num = int(mm.group(1))
        if num in REF_EN:
            p = para(doc)
            r = p.add_run(REF_EN[num]); set_run(r, cn="Times New Roman", en="Times New Roman", size=7.5)

    # ---- 声明 ----
    decs = re.findall(r"\*\[([^\]]+)\]\s*(.*?)\*", md)
    for k, v in decs:
        p = para(doc)
        r = p.add_run("[%s] %s" % (k, v)); set_run(r, cn="宋体", en="Times New Roman", size=9, italic=True)

    doc.save(OUT)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
