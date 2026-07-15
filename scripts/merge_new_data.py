from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将新标注数据合并到 hyacinth5 数据集。"
        "自动把 Label Studio 导出的 1-indexed 类别 ID 转成 YOLO 的 0-indexed。"
    )
    parser.add_argument(
        "--src-images",
        nargs="+",
        default=[
            "data/DJI_202606121754_002_20260612175453840448/DJI_202606121754_002_20260612175453840448",
            "TH_DJI/data/DJI_202606121645_002_20260612164552155584_results",
        ],
        help="源图片目录（可多个，脚本会按 stem 自动匹配标签）。",
    )
    parser.add_argument(
        "--src-labels",
        default="data/project-1-at-2026-06-16-09-26-36052dfa/labels",
        help="源标签目录（YOLO txt，1-indexed）。",
    )
    parser.add_argument(
        "--out",
        default="datasets/hyacinth5",
        help="输出数据集根目录。",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=1,
        help="类别 ID 偏移量（默认 1，表示从 1-indexed 转为 0-indexed）。",
    )
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集占比（默认 0.8）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际复制文件。",
    )
    return parser.parse_args()


def validate_and_convert_label_line(line: str, offset: int) -> tuple[str, int] | None:
    """校验并转换单行 YOLO 标签。返回 (转换后的行, class_id)；异常行返回 None。"""
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
    new_cls = cls - offset
    if new_cls < 0:
        return None
    return f"{new_cls} {' '.join(coords)}", new_cls


def main() -> None:
    args = parse_args()

    src_image_dirs = [Path(p) for p in args.src_images]
    src_labels = Path(args.src_labels)
    out_root = Path(args.out)

    for src_images in src_image_dirs:
        if not src_images.is_dir():
            raise FileNotFoundError(f"源图片目录不存在：{src_images}")
    if not src_labels.is_dir():
        raise FileNotFoundError(f"源标签目录不存在：{src_labels}")

    # 汇总所有源目录的图片（以 stem 去重，后出现的目录会覆盖先出现的）
    all_images: dict[str, Path] = {}
    for src_images in src_image_dirs:
        for p in src_images.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                all_images[p.stem] = p
    images = sorted(all_images.values())
    print(f"[INFO] 源图片（去重后） {len(images)} 张")

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

    if not pairs:
        print("[ERROR] 没有匹配到任何 image-label 对，请检查路径。")
        return

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
    if not args.dry_run:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    cls_counter: Counter[int] = Counter()
    bad_lines = 0
    copied = {"train": 0, "val": 0}

    def copy_split(
        pairs_list: list[tuple[Path, Path]],
        img_dir: Path,
        lbl_dir: Path,
        split: str,
    ) -> None:
        nonlocal bad_lines
        for img, lbl in pairs_list:
            if not args.dry_run:
                shutil.copy2(img, img_dir / img.name)
            out_lines: list[str] = []
            with lbl.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parsed = validate_and_convert_label_line(line, args.class_offset)
                    if parsed is None:
                        bad_lines += 1
                        print(f"  [WARN] 异常标签行（{lbl.name}）：{line.strip()[:60]}")
                        continue
                    new_line, cls = parsed
                    cls_counter[cls] += 1
                    out_lines.append(new_line)
            if not args.dry_run:
                with (lbl_dir / lbl.name).open("w", encoding="utf-8") as f:
                    f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            copied[split] += 1

    copy_split(train_pairs, dirs["train_img"], dirs["train_lbl"], "train")
    copy_split(val_pairs, dirs["val_img"], dirs["val_lbl"], "val")

    mode = "[DRY-RUN]" if args.dry_run else "[OK]"
    print(f"\n{mode} 合并完成（seed={args.seed}, ratio={args.ratio}）")
    print(f"  train: {len(train_pairs)} 张  -> {dirs['train_img']}")
    print(f"  val:   {len(val_pairs)} 张  -> {dirs['val_img']}")
    print(f"  已复制图片: {copied['train'] + copied['val']} 张")
    print(f"  异常标签行（已跳过）: {bad_lines}")
    print("  各类实例分布（train+val 合计）:")
    for cls in sorted(cls_counter):
        print(f"    class {cls}: {cls_counter[cls]}")


if __name__ == "__main__":
    main()
