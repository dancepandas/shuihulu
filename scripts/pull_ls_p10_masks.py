"""
LS project 10 brush PNG 导出（新格式兼容版）

官方 label_studio_sdk.converter.brush.save_brush_images_from_annotation 只读 result["rle"]，
而新版本 LS 把 rle 放在 result["value"]["rle"]，导致静默无文件。
本脚本复制官方 save_brush_images_from_annotation 逻辑，但适配 value 嵌套格式。

输入：data/ls_export_p10/meta.jsonl (pull_ls_p10.py 产出)
输出：data/ls_export_p10/task-{tid}-annotation-{aid}-by-{user}-{from_name}-{labels}-{i}.png
     单通道 uint8 PNG（alpha 通道的 0/255 二值）
"""
import os, json
import numpy as np
from PIL import Image
from collections import defaultdict
from label_studio_sdk.converter.brush import decode_rle

ROOT = r"D:\chengs\9.project\shuihulu"
META = os.path.join(ROOT, "data", "ls_export_p10", "meta.jsonl")
OUT_DIR = os.path.join(ROOT, "data", "ls_export_p10")


def _sanitize_email(x):
    if isinstance(x, dict):
        x = x.get("email", "")
    return "".join(c for c in str(x) if c.isalnum() or c in "@._-")


def _decode_results_new(results):
    """从 result["value"]["rle"] 解码，返回 {layer_name: alpha mask (H,W)}"""
    layers = {}
    counters = defaultdict(int)
    for r in results:
        if (r.get("type") or "").lower() != "brushlabels":
            continue
        val = r.get("value") or {}
        rle = val.get("rle")
        if not rle:
            continue
        H, W = r.get("original_height"), r.get("original_width")
        if not (H and W):
            continue
        from_name = r.get("from_name", "label")
        labels = val.get("brushlabels") or ["no_label"]
        key = (from_name, tuple(labels))
        i = str(counters[key]); counters[key] += 1
        name = f"{from_name}-" + "-".join(labels) + f"-{i}"
        img = decode_rle(rle)
        # 官方约定：4 通道 [H, W, 4]，取 alpha
        if img.size == H * W * 4:
            layers[name] = np.reshape(img, [H, W, 4])[:, :, 3]
        elif img.size == H * W:
            layers[name] = np.reshape(img, [H, W])
        else:
            # 截断/补齐，理论上不应出现
            layers[name] = np.zeros((H, W), dtype=np.uint8)
    return layers


def save_one(task_id, annotation_id, completed_by, results, out_dir):
    """适配新 value.rle 格式的 save_brush_images_from_annotation。"""
    layers = _decode_results_new(results)
    if not layers:
        return 0
    email = _sanitize_email(completed_by)
    n = 0
    for name, alpha in layers.items():
        sanitized = name.replace("/", "-").replace("\\", "-")
        fname = os.path.join(
            out_dir,
            f"task-{task_id}-annotation-{annotation_id}-by-{email}-{sanitized}.png",
        )
        Image.fromarray(alpha).save(fname)
        n += 1
    return n


def main():
    tasks = []
    with open(META, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    print(f"[meta] {len(tasks)} tasks loaded")

    total_png = 0
    n_tasks_with_mask = 0
    for ti, t in enumerate(tasks):
        task_png = 0
        for a in t.get("annotations", []):
            task_png += save_one(
                task_id=t["id"],
                annotation_id=a["id"],
                completed_by=a.get("completed_by", ""),
                results=a.get("result", []),
                out_dir=OUT_DIR,
            )
        if task_png: n_tasks_with_mask += 1
        total_png += task_png
        if (ti + 1) % 50 == 0 or (ti + 1) == len(tasks):
            print(f"  [{ti+1}/{len(tasks)}]  tasks_with_mask={n_tasks_with_mask}  pngs={total_png}")

    print(f"\nDONE  total PNGs: {total_png}")
    print(f"  out: {OUT_DIR}")


if __name__ == "__main__":
    main()