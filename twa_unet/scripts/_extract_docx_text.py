# -*- coding: utf-8 -*-
"""按 body 顺序导出 docx 正文 (段落 + 表格), 供审稿阅读. 只读, 不修改 docx."""
import sys

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(doc):
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def main(path, out):
    doc = Document(path)
    lines = []
    ti = 0
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            t = block.text.strip()
            if not t:
                continue
            # 图片段落
            if block._p.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
                lines.append("[图片]")
                continue
            lines.append(t)
        else:
            ti += 1
            lines.append(f"[表{ti} 开始]")
            for row in block.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                lines.append("T| " + " | ".join(cells))
            lines.append(f"[表{ti} 结束]")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    print(f"wrote {out}, {len(lines)} blocks")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
