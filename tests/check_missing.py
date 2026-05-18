import sys
from pathlib import Path

img_dir = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/train")
lbl_dir = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")

img_stems = {p.stem for p in img_dir.glob("*.JPG")}
lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

missing = sorted(img_stems - lbl_stems)
print(f"总图片: {len(img_stems)}, 已标注: {len(lbl_stems)}, 未标注: {len(missing)}")
print("\n未标注的图片:")
for s in missing:
    print(f"  {s}.JPG")