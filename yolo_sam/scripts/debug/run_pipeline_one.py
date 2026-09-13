"""单图完整 pipeline: GLI Otsu + YOLO 排非WH + 形态学闭运算 + YOLO-WH 交集约束"""
import sys
from pathlib import Path

ROOT = Path("D:/chengs/9.project/shuihulu/yolo_sam")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pipelines.inference import DualPipeline

IMG = ROOT / "data/fc4999996d2b4657a56be888a91cbb94.png"
OUT = ROOT / "doc/vis_pipeline_fc499.png"

pipe = DualPipeline(
    yolo_model_path="runs/hyacinth8_yolo_sam/weights/best.pt",
    device="cuda:0",
    conf=0.10,
)

result = pipe.segment(str(IMG), mode="gli")
result.save_vis(str(OUT))
print(result.summary())
print(f"Saved: {OUT}")
