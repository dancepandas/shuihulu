import random
import shutil
from pathlib import Path

random.seed(42)

src_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/train")
src_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")

val_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/val")
val_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/val")
val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

lbl_stems = sorted(p.stem for p in src_lbl.glob("*.txt"))
print(f"Total labeled: {len(lbl_stems)}")

val_ratio = 0.2
n_val = max(1, int(len(lbl_stems) * val_ratio))
val_stems = set(random.sample(lbl_stems, n_val))
print(f"Train: {len(lbl_stems) - n_val}, Val: {n_val}")

for stem in val_stems:
    img_file = src_img / f"{stem}.JPG"
    lbl_file = src_lbl / f"{stem}.txt"
    if img_file.exists():
        shutil.move(str(img_file), str(val_img / img_file.name))
    if lbl_file.exists():
        shutil.move(str(lbl_file), str(val_lbl / lbl_file.name))

print("Done!")
print(f"Train images: {len(list(src_img.glob('*.JPG')))}")
print(f"Val images: {len(list(val_img.glob('*.JPG')))}")
print(f"Train labels: {len(list(src_lbl.glob('*.txt')))}")
print(f"Val labels: {len(list(val_lbl.glob('*.txt')))}")