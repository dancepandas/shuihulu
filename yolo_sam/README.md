# YOLO + SAM / GLI — 检测路线

工程化检测路线：**YOLOv8-seg + SAM 两阶段**精细掩膜、**YOLO + GLI 植被指数融合**、
以及 SAM→YOLO 知识蒸馏。面向部署服务与分割路线的对比基线。

## 目录

```text
app.py                  # Web 推理服务 (纯 YOLO / YOLO+GLI / YOLO+SAM, :5000)
main.py                 # YOLO+SAM CLI 推理 (单图/目录/视频)
serve.py                # YOLO+SAM HTTP API 服务 (Docker 部署用, :13000)
src/
  shuihulu_yolo_sam/    #   两阶段流水线 (Segmenter API)
  shuihulu_yolov10_seg/ #   YOLOv8-seg 基线封装
  shuihulu_distill/     #   SAM→YOLO 蒸馏损失 (KL + Dice)
configs/                # Ultralytics 数据集配置 (dataset_hyacinth5~v10_seg.yaml, runtime.yaml)
scripts/
  train_yolo_sam.py / val_yolo_sam.py     # YOLOv8-seg 训练/验证
  predict_yolo_sam.py                     # 两阶段推理 + 面积估算
  build_yolo_v10.py / split_dataset.py / merge_new_data.py
  build_seg_dataset.py / build_seg_dataset2.py   # YOLO 数据集构建
  preanno_v6.py / preanno_v9_ls.py        # YOLO 预标注
  gli_v6_pipeline.py / exg_v6_pipeline.py # GLI / ExG 植被指数融合 pipeline
  pipelines/            # gli_v8_pipeline / inference / run_v8 (一站式训练)
  debug/                # GLI/SAM/YOLO 各版对比调试脚本
  download_weights.py
gli_onnx_model/         # GLI 路线 ONNX 模型 + 说明
doc/                    # GLI_Pipeline.md / SAM 蒸馏论文初稿 / pipeline 可视化图
temp/                   # TH_DJI / temp_compare_5x3 / temp_gli_yolo_wh_intersect_20 (临时件)
```

## 使用

```bash
# CLI 两阶段推理
python yolo_sam/main.py --source <图片或目录> \
    --yolo-model runs/segment/.../best.pt --sam-checkpoint weights/sam_vit_h.pth

# HTTP API (Docker: docker-compose up -d)
python yolo_sam/serve.py --yolo /models/hyacinth5_best.pt --sam /models/sam_vit_h.pth --port 13000

# Web 对比界面
python yolo_sam/app.py    # → http://localhost:5000

# YOLOv8-seg 训练
python yolo_sam/scripts/train_yolo_sam.py --data yolo_sam/configs/dataset_hyacinth8_seg.yaml
```

- 权重：`weights/sam_vit_h.pth`、`weights/yolov8m-seg.pt`、`weights/yolo26n.pt`（原地未动）
- 训练产物：`runs/hyacinth*_yolo_sam/`、`runs/segment/`（原地未动）
- 数据集：`datasets/hyacinth8/9`、`hyacinth_v10`（Ultralytics 格式）
- 注意：脚本约定从**项目根**运行（相对路径 `runs/`、`datasets/`、`yolo_sam/configs/` 从根解析）
