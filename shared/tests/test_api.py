from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # 项目根
SRC = ROOT / "yolo_sam" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shuihulu_yolo_sam import Segmenter, segment_image

# --- 方式1：一键函数 ---
result = segment_image(
    image=ROOT / "datasets/crack_seg/images/val/crack_0001.jpg",
    yolo_model_path=ROOT / "runs/segment/runs/segment/crack_yolo_sam/weights/best.pt",
    sam_checkpoint=ROOT / "weights/sam_vit_h.pth",
    device="cpu",
    class_names=["crack"],
)

result.save_vis(ROOT / "runs/api_test/one_click.jpg")
print("=== one-click ===")
print(result.summary())

# --- 方式2：Segmenter 实例（复用模型，批量处理） ---
seg = Segmenter(
    yolo_model_path=ROOT / "runs/segment/runs/segment/crack_yolo_sam/weights/best.pt",
    sam_checkpoint=ROOT / "weights/sam_vit_h.pth",
    device="cpu",
    class_names=["crack"],
)

val_dir = ROOT / "datasets/crack_seg/images/val"
images = sorted(val_dir.glob("*.jpg"))[:3]

out_dir = ROOT / "runs/api_test"
out_dir.mkdir(parents=True, exist_ok=True)

for img_path in images:
    r = seg.segment(str(img_path))
    r.save_vis(out_dir / img_path.name)
    r.save_masks(out_dir / "masks", stem=img_path.stem)
    print(f"{img_path.name}: {r.summary()}")

print("\nDone!")