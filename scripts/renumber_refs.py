# -*- coding: utf-8 -*-
"""按 GB/T 7714 顺序编码制：参考文献按文中首次引用顺序重编号。
旧→新映射由正文首次出现顺序推出；正文引用（含区间/并列）与文献表同步改写。
"""
import re, os

ROOT = r"D:\chengs\9.project\shuihulu"
MD = os.path.join(ROOT, "doc", "论文_基于SegFormer-B2优化的轻量化水葫芦语义分割.md")

# 旧号 → 新号（按正文首次引用顺序：1,2,3,4,20,21,5,6,7,10..15,8,9,16,17,18,19）
MAP = {1:1, 2:2, 3:3, 4:4, 20:5, 21:6, 5:7, 6:8, 7:9,
       10:10, 11:11, 12:12, 13:13, 14:14, 15:15,
       8:16, 9:17, 16:18, 17:19, 18:20, 19:21}

CITE = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")

def remap_cite(m):
    out = []
    for part in m.group(1).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.append(f"{MAP[int(a)]}-{MAP[int(b)]}")
        else:
            out.append(str(MAP[int(part)]))
    # 并列编号按新号排序，如 [16,17,18] → [16-18] 的压缩留给人工判断，这里保持逗号并列
    return "[" + ",".join(out) + "]"

def main():
    src = open(MD, encoding="utf-8").read()
    body, ref_part = src.split("## 参考文献", 1)
    # 1) 正文引用重映射（仅数字标号，不会误伤 [J] 等）
    body = CITE.sub(remap_cite, body)
    # 2) 文献表重排
    refs = {}
    for line in ref_part.split("\n"):
        mm = re.match(r"^\[(\d+)\]\s*(.+)$", line.strip())
        if mm:
            refs[int(mm.group(1))] = mm.group(2).strip()
    assert len(refs) == 21, f"expect 21 refs, got {len(refs)}"
    new_lines = []
    for old, new in sorted(MAP.items(), key=lambda kv: kv[1]):
        new_lines.append(f"[{new}] {refs[old]}")
    # 保留文献表之后的声明部分
    tail = ref_part.split("---", 1)
    tail = "---" + tail[1] if len(tail) > 1 else ""
    out = body + "## 参考文献\n\n" + "\n".join(new_lines) + "\n\n" + tail.strip() + "\n"
    open(MD, "w", encoding="utf-8").write(out)
    # 3) 校验：正文引用号集合 == 1..21
    used = set()
    for m in CITE.finditer(body):
        for part in m.group(1).split(","):
            if "-" in part:
                a, b = part.split("-")
                used.update(range(int(a), int(b) + 1))
            else:
                used.add(int(part))
    assert used == set(range(1, 22)), f"citation set mismatch: {sorted(used)}"
    print("renumbered OK; citations cover [1]..[21] in first-use order")

if __name__ == "__main__":
    main()
