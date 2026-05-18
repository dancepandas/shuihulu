# Shuihulu YOLOv8 + SAM 水葫芦面积识别

基于论文《基于YOLOv7的水葫芦目标检测方法研究》中的 YOLO + SAM 两阶段组合方案复现：

- **第一阶段 (YOLOv8-seg)**：目标检测，输出水葫芦边界框
- **第二阶段 (SAM)**：以边界框为提示，生成像素级精细掩膜
- **面积估算**：从掩膜计算水葫芦覆盖面积

## 目录结构

```text
configs/
  dataset_hyacinth_seg.yaml   # 数据集配置
  runtime.yaml                # 运行参数（含 SAM 配置）
scripts/
  train_yolo_sam.py           # YOLOv8 训练（迁移学习 + 冻结 P3）
  predict_yolo_sam.py         # YOLO+SAM 两阶段推理 + 面积估算
  val_yolo_sam.py             # YOLO+SAM 验证（IoU 评估）
  train.py                    # 纯 YOLOv8-seg 基线训练
  val.py                      # 纯 YOLOv8-seg 验证
  predict.py                  # 纯 YOLOv8-seg 推理
  download_weights.py         # 权重下载
src/
  shuihulu_yolo_sam/
    __init__.py
    model.py                  # YOLO+SAM 两阶段流水线核心
  shuihulu_yolov10_seg/
    __init__.py
    model.py                  # YOLOv8-seg 基线
requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 下载模型权重

```bash
# 下载 YOLOv8-seg 预训练权重
python scripts/download_weights.py --model yolov8n-seg.pt

# 下载 SAM ViT-H 权重（论文指定版本）
python scripts/download_weights.py --model sam_vit_h.pth
```

内置权重：`yolov8n-seg.pt` / `yolov8s-seg.pt` / `yolov8m-seg.pt` / `yolov8l-seg.pt` / `yolov8x-seg.pt` / `sam_vit_h.pth` / `sam_vit_l.pth` / `sam_vit_b.pth`

## 数据集准备

按 Ultralytics 实例分割目录组织：

```text
datasets/
  hyacinth_seg/
    images/
      train/
      val/
    labels/
      train/
      val/
```

修改 `configs/dataset_hyacinth_seg.yaml` 中的路径和类别名。

## 训练 YOLOv8（第一阶段）

```bash
python scripts/train_yolo_sam.py \
  --model weights/yolov8n-seg.pt \
  --data configs/dataset_hyacinth_seg.yaml
```

训练策略（与论文一致）：
- COCO 预训练权重迁移学习
- 冻结 FPN P3 层权重（`--no-freeze-p3` 可取消）
- SGD 优化器，640×640 输入，100 轮，早停 50 轮
- 可选遗传算法超参数搜索：`--tune --tune-iterations 300`

## 两阶段推理 + 面积估算

```bash
python scripts/predict_yolo_sam.py \
  --yolo-model runs/segment/hyacinth_yolo_sam/weights/best.pt \
  --sam-checkpoint weights/sam_vit_h.pth \
  --source path/to/image_or_folder \
  --save \
  --pixels-per-meter 100
```

输出：可视化叠加图 + 各目标掩膜 + 覆盖面积（像素数或平方米）。

## 验证

```bash
python scripts/val_yolo_sam.py \
  --yolo-model runs/segment/hyacinth_yolo_sam/weights/best.pt \
  --sam-checkpoint weights/sam_vit_h.pth \
  --data configs/dataset_hyacinth_seg.yaml
```

## 模型架构说明

| 阶段 | 模型 | 职责 |
|------|------|------|
| Step 1 | YOLOv8-seg | Backbone(C2f) + Neck(FPN+PAN) + Head(Anchor-Free)，输出边界框 |
| Step 2 | SAM (ViT-H) | 零样本分割，接收边界框提示，输出像素级掩膜 |

SAM 无需针对水葫芦重新训练，利用其零样本学习能力直接精细分割。
