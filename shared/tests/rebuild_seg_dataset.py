import random
import shutil
from pathlib import Path

random.seed(42)

src_img = Path("D:/chengs/9.project/shuihulu/datasets/station")
src_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")

train_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/train")
train_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")
val_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/val")
val_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/val")

for d in [train_img, train_lbl, val_img, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        f.unlink()

lbl_stems = sorted(p.stem for p in src_lbl.glob("*.txt"))
print(f"Total labeled: {len(lbl_stems)}")

n_val = max(1, int(len(lbl_stems) * 0.2))
val_set = set(random.sample(lbl_stems, n_val))

for stem in lbl_stems:
    if stem in val_set:
        dst_img, dst_lbl = val_img, val_lbl
    else:
        dst_img, dst_lbl = train_img, train_lbl
    img_file = src_img / f"{stem}.JPG"
    lbl_file = src_lbl / f"{stem}.txt"
    if img_file.exists():
        shutil.copy2(str(img_file), str(dst_img / img_file.name))
    if lbl_file.exists():
        shutil.copy2(str(lbl_file), str(dst_lbl / lbl_file.name))

print(f"Train: {len(lbl_stems) - n_val}, Val: {n_val}")
print(f"Train images: {len(list(train_img.glob('*.JPG')))}")
print(f"Val images: {len(list(val_img.glob('*.JPG')))}")
print(f"Train labels: {len(list(train_lbl.glob('*.txt')))}")
print(f"Val labels: {len(list(val_lbl.glob('*.txt')))}")