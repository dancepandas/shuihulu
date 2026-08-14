# Shuihulu — 河道水葫芦语义分割与识别

面向河道水葫芦实时监测的语义分割 / 目标识别项目。核心为**轻量化单阶段语义分割方法 LWH-Seg**（基于 SegFormer-B2 优化），并内置 **YOLOv8-seg + SAM 两阶段**与 **纯 YOLO / YOLO+GLI 融合**等对比方案与 Flask 推理服务。

- 论文（中文核心）：基于 SegFormer-B2 优化的河道水葫芦轻量化语义分割方法
- 推理速度：约 7.4 ms/图（RTX 2060），支持机载/边缘部署

### 两套语义分割标签体系

| 体系 | 用途 | 类别 |
|------|------|------|
| **6 类（旧）** | 论文正文、LWH-Seg（segformer_b2_ls+v9） | 背景 / 船只 / 桥梁 / 岸基建筑 / 水葫芦 / 树木 |
| **5 类（新）** | v10 起重训基座（segformer_b2_ls_v10/v11） | water 水体 / water_hyacinth 水葫芦 / hard_structure 硬结构 / shore_vegetation 岸基植被 / other_aquatic_vegetation 其他水生植物 |

> 5 类新体系面向水葫芦监测重新定义：将旧「船只/桥梁/岸基建筑」合并为 `hard_structure`（桥墩、堤坝、护岸等硬结构），「树木」细分为 `shore_vegetation`（岸基植被）与 `other_aquatic_vegetation`（其他水生植物），并新增水体 `water` 作为背景。核心目标始终是 `water_hyacinth`（水葫芦）。

## 方法概览

### 本文方法 LWH-Seg（SegFormer-B2 单阶段）

以轻量级层次化 Transformer 网络 **SegFormer-B2**（MiT-B2 编码器 + 全 MLP 解码器，27.35 M 参数）为分割主干，三项针对性设计：

- **数据层面**：多源标注融合 + 针对性数据增强（多尺度、多光照）
- **损失层面**：类别加权交叉熵 + Dice 联合损失（缓解水葫芦类别不平衡）
- **训练策略**：两段式训练（第一阶段冻结编码器、第二阶段联合微调）

单次前向直接输出六类像素级分割结果，无需检测框提示。

### 对比基线

| 方案 | 说明 |
|------|------|
| SegFormer-B2-FT | 直接微调基线（标准交叉熵 + 基础增强） |
| YOLOv8-seg + SAM | 两阶段方案：YOLOv8-seg 定位水葫芦框，SAM 生成像素级掩膜 |
| 纯 YOLO / YOLO+GLI | 目标检测基线（GLI 为植被指数融合） |

## 目录结构

```text
app.py                          # Flask 单文件 Web 推理服务（纯YOLO / YOLO+GLI / YOLO+SAM / SegFormer / SegFormer-v10 五模式）
main.py / serve.py              # 服务入口
configs/
  dataset_hyacinth_v10_seg.yaml # 数据集配置（V5–V10 多版本；v10 为 5 类新体系）
  runtime.yaml                  # 运行参数
scripts/
  segformer_train.py            # SegFormer-B2 语义分割训练（6 类，论文方法）
  segformer_train_v10.py        # SegFormer-B2 训练（5 类新标签，640×640 + 两段式 + 加权 CE+Dice）
  eval_v10_testdata.py          # v10 模型在测试数据上的可视化验证
  eval_paper_models.py          # 论文方法评测
  preanno_v10_ls.py             # SegFormer-v10 对 LS project 10 未标注任务笔刷预标注
  pull_ls_p10.py                # 拉 LS project 10 元数据 + 原图
  pull_ls_p10_masks.py          # 解码 value.rle → 每标签 PNG
  pull_all_p10.py               # 一键拉全部 completed 标注 → 整理成训练集 v11
  build_hyacinth_ls_v10.py      # 合并 PNG → 5 类索引 mask
  split_seg_dataset.py          # 语义分割 train/val 划分
  build_seg_dataset.py          # 数据集构建
  train_yolo_sam.py             # YOLOv8-seg 训练（两阶段第一阶段）
  predict_yolo_sam.py           # YOLO+SAM 两阶段推理 + 面积估算
  val_yolo_sam.py               # YOLO+SAM 验证
  gli_v6_pipeline.py            # GLI 植被指数融合 pipeline
  preanno_v9_ls.py              # Label Studio 笔刷预标注（旧 6 类）
  download_weights.py           # 权重下载
  build_docx_shuili.py          # 期刊投稿稿 docx 生成
  fig1_framework.py             # 方法框架图生成
src/
  shuihulu_yolo_sam/            # YOLO+SAM 两阶段流水线
  shuihulu_yolov10_seg/         # YOLOv8-seg 基线
  shuihulu_distill/             # 蒸馏损失等
```

## 安装

```bash
pip install -r requirements.txt
# 可选：前端依赖（docx 文档生成）
npm install
```

## 数据集

按语义分割目录组织：

```text
datasets/
  hyacinth_seg_p7p9/            # SegFormer 训练集（project-7/9 笔刷标注，6 类）
    images/{train,val}/
    masks/{train,val}/          # 0=背景 1=船只 2=桥梁 3=岸基建筑 4=水葫芦 5=树木
  hyacinth_ls_v10/              # LS project 10 笔刷标注（5 类新体系，304 张）
    images/{train,val}/         # 243 train / 61 val
    masks/{train,val}/          # 0=water 1=water_hyacinth 2=hard_structure 3=shore_vegetation 4=other_aquatic_vegetation
  hyacinth_ls_v11/              # LS project 10 人工修正后（5 类新体系，674 张）
    images/{train,val}/         # 539 train / 135 val
    masks/{train,val}/
  hyacinth_v10/                 # YOLOv8-seg 数据集（Ultralytics 格式）
    images/{train,val}/
    labels/{train,val}/
```

## 训练 SegFormer-B2（本文方法）

```bash
python scripts/segformer_train.py
```

- 主干：`weights/segformer_b2`（预训练），6 类，512×512
- 优化器：AdamW（lr 6e-5，wd 0.01），余弦退火，80 epoch
- 输出：`runs/segformer/segformer_b2_v2`（直接微调基线）
- 本文完整方案（多源融合 + 联合损失 + 两段式训练）对应 `runs/segformer/segformer_b2_ls+v9`

### 新标签体系 v10（LS project 10 笔刷标注，5 类）

```bash
python scripts/segformer_train_v10.py
```

- 数据集：`datasets/hyacinth_ls_v10/{images,masks}/{train,val}`（243 train / 61 val）
- 类别：`0 water, 1 water_hyacinth, 2 hard_structure, 3 shore_vegetation, 4 other_aquatic_vegetation`
- 主干：`nvidia/segformer-b2-finetuned-ade-512-512`（ADE20K 语义分割预训练；本地无则自动从 HF 下载）
- 输入 640×640，batch=1 + 梯度累积 4 步
- 损失：类别加权 CE + Dice 联合（水葫芦权重最高）
- 策略：两段式（冻结编码器 20 epoch → 解冻联合微调 60 epoch）
- 输出：`runs/segformer/segformer_b2_ls_v10`

### 构建 v10/v11 数据集（从 LS 导出）

**v10（首次 304 张）分步：**
```bash
python scripts/pull_ls_p10.py            # 拉元数据 + 原图 → data/ls_export_p10/
python scripts/pull_ls_p10_masks.py      # 解码 value.rle → 每标签 PNG
python scripts/build_hyacinth_ls_v10.py  # 合并成 5 类索引 mask → datasets/hyacinth_ls_v10/
python scripts/split_seg_dataset.py      # 8:2 切 train/val
```

**v11（人工修正后 674 张）一键：**
```bash
python scripts/pull_all_p10.py           # 拉全部 completed → 解码 → 合并 → 切分 → datasets/hyacinth_ls_v11/
```

> 用 v11 重训时，改 `segformer_train_v10.py` 里 `DATASET_DIR` 为 `hyacinth_ls_v11`、`OUTPUT_DIR` 为 `segformer_b2_ls_v11` 即可。

### 标注闭环（预标注 → 人工修正 → 重训）

```bash
python scripts/preanno_v10_ls.py   # SegFormer-v10 对 LS project 10 未标注任务做笔刷预标注
# （在 Label Studio 人工修正/接受预标注）
python scripts/pull_all_p10.py     # 拉回人工修正后的 completed 标注 → 新数据集
# （改 DATASET_DIR 重训，迭代）
```

## Web 推理服务

```bash
python app.py
```

五种模式（POST /predict，`model` 字段）：

- **纯 YOLO**（`yolo`）：YOLOv8-seg 输出水葫芦边界框
- **YOLO+GLI**（`gli`）：YOLO 检测 + GLI 植被指数融合
- **YOLO+SAM**（`sam`）：YOLOv8-seg 定位 + SAM 精细掩膜（两阶段）
- **SegFormer**（`segformer`）：单次前向输出六类像素级分割（论文方法，512×512）
- **SegFormer-v10**（`segformer_v10`）：单次前向输出五类像素级分割（新标签体系，640×640）

## 评测指标

- mAcc / mIoU / 水葫芦类别 IoU（WH IoU）
- 参数量 / 模型体积 / 单图推理时间
