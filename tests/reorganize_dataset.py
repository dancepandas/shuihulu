import random
import shutil
from pathlib import Path

random.seed(42)

src_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images")
src_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels")

train_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/train")
train_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")
val_img = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/val")
val_lbl = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/val")

for d in [train_img, train_lbl, val_img, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

items = []
for img in sorted(src_img.glob("*.JPG")):
    lbl = src_lbl / (img.stem + ".txt")
    if lbl.exists():
        orig_name = img.name.split("-", 1)[-1]
        items.append((img, lbl, orig_name))

print(f"Total labeled: {len(items)}")

n_val = max(1, int(len(items) * 0.2))
val_set = set(random.sample(range(len(items)), n_val))

for i, (img, lbl, orig_name) in enumerate(items):
    if i in val_set:
        dst_img, dst_lbl = val_img, val_lbl
    else:
        dst_img, dst_lbl = train_img, train_lbl
    shutil.copy2(str(img), str(dst_img / orig_name))
    shutil.copy2(str(lbl), str(dst_lbl / (orig_name.rsplit(".", 1)[0] + ".txt")))

print(f"Train: {len(items) - n_val}, Val: {n_val}")
print(f"Train images: {len(list(train_img.glob('*.JPG')))}")
print(f"Val images: {len(list(val_img.glob('*.JPG')))}")
print(f"Train labels: {len(list(train_lbl.glob('*.txt')))}")
print(f"Val labels: {len(list(val_lbl.glob('*.txt')))}")