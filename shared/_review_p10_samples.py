"""review 样本：每个标签挑 1 张 mask（带它所属 task 的原图叠加），输出 review_p10_samples.png"""
import os, glob, re
import numpy as np, cv2, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\chengs\9.project\shuihulu"
EXP = os.path.join(ROOT, "data", "ls_export_p10")
OUT = os.path.join(EXP, "review_p10_samples.png")

COLORS = {
    "water":                     (66, 165, 245),
    "shore_vegetation":          (67, 160, 71),
    "water_hyacinth":            (255, 152, 0),
    "hard_structure":            (158, 158, 158),
    "other_aquatic_vegetation":  (171, 71, 188),
}
LABELS = list(COLORS.keys())

# 用 meta.jsonl 建 task_id → image basename 映射
task_image = {}
with open(os.path.join(EXP, "meta.jsonl"), encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        task_image[t["id"]] = os.path.basename(t["data"]["image"])
print(f"loaded {len(task_image)} task→image map")

# 每个标签挑一张 mask：选像素数占整图 1%~30% 的（避免全空或几乎全图覆盖的极端）
samples = {}
for f in sorted(glob.glob(os.path.join(EXP, "task-*-label-*.png"))):
    base = os.path.basename(f).replace(".png", "")
    m = re.match(r"task-(\d+)-annotation-\d+-by-\d+-label-(.+?)-\d+$", base)
    if not m: continue
    tid, label = int(m.group(1)), m.group(2)
    if label not in COLORS or label in samples: continue
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    H, W = im.shape[:2]
    ratio = (im > 0).sum() / (H * W)
    if 0.005 < ratio < 0.35:
        samples[label] = (tid, f)

print("samples:")
for lab, (tid, f) in samples.items():
    print(f"  {lab:30s}  task={tid}  file={os.path.basename(f)}")

n = len(LABELS) + 1
fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.4))

# 第 1 张：water_hyacinth 的 task 原图作为主参考
ref_tid = samples["water_hyacinth"][0]
ref_img_path = os.path.join(EXP, "images", task_image[ref_tid])
ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
axes[0].imshow(ref_img)
axes[0].set_title(f"原图 (task {ref_tid})\n{os.path.basename(ref_img_path)[:30]}",
                  fontsize=10, fontweight="bold")
axes[0].set_xticks([]); axes[0].set_yticks([])

# 后 5 张：每张用各自 task 的原图 + 该标签 mask 叠加
for i, lab in enumerate(LABELS, 1):
    tid, mask_path = samples[lab]
    img_path = os.path.join(EXP, "images", task_image[tid])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    overlay = img.astype(np.float32).copy()
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    region = m > 0
    cnt = int(region.sum())
    col = np.array(COLORS[lab], np.float32)
    overlay[region] = (1 - 0.55) * overlay[region] + 0.55 * col
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    axes[i].imshow(overlay)
    axes[i].set_title(f"{lab}\ntask {tid}  |  {cnt:,} px  ({cnt/(H*W)*100:.1f}%)",
                      fontsize=9, color="#%02x%02x%02x" % COLORS[lab], fontweight="bold")
    axes[i].set_xticks([]); axes[i].set_yticks([])

plt.suptitle("LS project 10 — 5 个标签语义 review（橙色 = 水葫芦核心目标）",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nsaved: {OUT}")