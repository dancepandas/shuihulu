import shutil
from pathlib import Path

src = Path("D:/chengs/9.project/shuihulu/datasets/station")
img_train = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/train")
img_val = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/images/val")
lbl_train = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/train")
lbl_val = Path("D:/chengs/9.project/shuihulu/datasets/hyacinth_seg/labels/val")

img_train.mkdir(parents=True, exist_ok=True)

val_stems = set(p.stem for p in lbl_val.glob("*.txt"))
train_stems = set(p.stem for p in lbl_train.glob("*.txt"))

for img in sorted(src.glob("*.JPG")):
    stem = img.stem
    if stem in val_stems:
        shutil.copy2(str(img), str(img_val / img.name))
    elif stem in train_stems:
        shutil.copy2(str(img), str(img_train / img.name))

print(f"Train images: {len(list(img_train.glob('*.JPG')))}")
print(f"Val images: {len(list(img_val.glob('*.JPG')))}")
print(f"Train labels: {len(list(lbl_train.glob('*.txt')))}")
print(f"Val labels: {len(list(lbl_val.glob('*.txt')))}")