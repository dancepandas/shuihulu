from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Label Studio 导出的 YOLO 标注数据集按比例随机划分为 train/val。"
        "复制（不动源数据），输出标准 YOLO 目录结构。"
    )
    parser.add_argument("--src-images", required=True, help="源图片目录。")
    parser.add_argument("--src-labels", required=True, help="源标签目录（YOLO txt）。")
    parser.add_argument("--out", required=True, help="输出数据集根目录，例如 datasets/hyacinth5。")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集占比（默认 0.8）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）。")
    return parser.parse_args()


def validate_label_line(line: str) -> tuple[int, int] | None:
    """校验单行 YOLO 标签。返回 (class_id, 点数)；异常行返回 None。"""
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        cls = int(parts[0])
    except ValueError:
        return None
    coords = parts[1:]
    if len(coords) % 2 != 0 or len(coords) < 6:
        return None
    return cls, len(coords) // 2


def main() -> None:
    args = parse_args()

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    out_root = Path(args.out)

    if not src_images.is_dir():
        raise FileNotFoundError(f"源图片目录不存在：{src_images}")
    if not src_labels.is_dir():
        raise FileNotFoundError(f"源标签目录不存在：{src_labels}")

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[INFO] 源图片 {len(images)} 张")

    # 按 stem 配对 image ↔ label
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        lbl = src_labels / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1
            print(f"  [WARN] 缺标签，跳过：{img.name}")
    if missing:
        print(f"[INFO] 跳过缺标签图片 {missing} 张")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # 准备目录
    dirs = {
        "train_img": out_root / "images" / "train",
        "val_img": out_root / "images" / "val",
        "train_lbl": out_root / "labels" / "train",
        "val_lbl": out_root / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cls_counter: Counter[int] = Counter()
    bad_lines = 0

    def copy_split(pairs_list: list[tuple[Path, Path]], img_dir: Path, lbl_dir: Path) -> None:
        nonlocal bad_lines
        for img, lbl in pairs_list:
            shutil.copy2(img, img_dir / img.name)
            # 写校验后的标签（原样复制，但统计 + 告警异常行）
            out_lines: list[str] = []
            with lbl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parsed = validate_label_line(line)
                    if parsed is None:
                        bad_lines += 1
                        print(f"  [WARN] 异常标签行（{lbl.name}）：{line.strip()[:60]}")
                        continue
                    cls_counter[cls := parsed[0]] += 1  # noqa: B023
                    out_lines.append(line.rstrip("\n"))
            with (lbl_dir / lbl.name).open("w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    copy_split(train_pairs, dirs["train_img"], dirs["train_lbl"])
    copy_split(val_pairs, dirs["val_img"], dirs["val_lbl"])

    print(f"\n[OK] 划分完成（seed={args.seed}, ratio={args.ratio}）")
    print(f"  train: {len(train_pairs)} 张  -> {dirs['train_img']}")
    print(f"  val:   {len(val_pairs)} 张  -> {dirs['val_img']}")
    print(f"  异常标签行（已跳过）: {bad_lines}")
    print("  各类实例分布（train+val 合计）:")
    for cls in sorted(cls_counter):
        print(f"    class {cls}: {cls_counter[cls]}")


if __name__ == "__main__":
    main()
