"""
语义分割数据集按比例切 train/val（同 stem 配对 image ↔ mask）。

用法：
  python scripts/split_seg_dataset.py \
      --src-images datasets/hyacinth_ls_v10/images \
      --src-masks  datasets/hyacinth_ls_v10/masks \
      --out         datasets/hyacinth_ls_v10 \
      --ratio 0.8 --seed 42
"""
import argparse, shutil, random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-images", required=True)
    ap.add_argument("--src-masks", required=True)
    ap.add_argument("--out", required=True, help="输出根目录（含 images/{train,val} masks/{train,val}）")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    a = parse_args()
    src_img = Path(a.src_images); src_msk = Path(a.src_masks)
    out_root = Path(a.out)

    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[INFO] 源图片 {len(images)} 张")

    pairs = []
    missing = 0
    for img in images:
        m = src_msk / f"{img.stem}.png"
        if m.exists():
            pairs.append((img, m))
        else:
            missing += 1
            print(f"  [WARN] 缺 mask，跳过：{img.name}")
    if missing:
        print(f"[INFO] 跳过缺 mask 图片 {missing} 张")

    random.seed(a.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * a.ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    dirs = {
        "train_img": out_root / "images" / "train",
        "val_img":   out_root / "images" / "val",
        "train_msk": out_root / "masks" / "train",
        "val_msk":   out_root / "masks" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def copy_split(pairs_list, img_dir, msk_dir):
        for img, msk in pairs_list:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(msk, msk_dir / msk.name)

    copy_split(train_pairs, dirs["train_img"], dirs["train_msk"])
    copy_split(val_pairs,   dirs["val_img"],   dirs["val_msk"])

    print(f"\n[OK] 划分完成（seed={a.seed}, ratio={a.ratio}）")
    print(f"  train: {len(train_pairs)} 张  -> {dirs['train_img']}")
    print(f"  val:   {len(val_pairs)} 张  -> {dirs['val_img']}")


if __name__ == "__main__":
    main()