# Shuihulu — 河道水葫芦语义分割与识别

面向河道水葫芦监测的语义分割 / 目标识别项目。代码按**三条模型路线**组织，各自独立目录：

| 目录 | 路线 | 模型 | 状态 |
|------|------|------|------|
| [`twa_unet/`](twa_unet/) | TWAU-Net | U-Net + 三角窗注意力（TWA），5 类新标签 | **现行论文**（《水利水电技术》投稿，Stage5 复审 Minor Revision） |
| [`segformer/`](segformer/) | LWH-Seg | SegFormer-B2 单阶段语义分割（6 类旧体系 + v10 起 5 类新体系） | 旧论文基线 |
| [`yolo_sam/`](yolo_sam/) | 检测路线 | YOLOv8-seg + SAM 两阶段、GLI 植被指数融合、SAM 蒸馏 | 工程服务/对比基线 |
| [`shared/`](shared/) | 数据集工具 | Label Studio 拉取/解码/合并/划分（v10/v11 数据集两条分割路线共用） | 公共 |

## 目录约定

- **代码与文档**在四个目录内；**数据与产物原地不动**：
  - `datasets/` 各版本数据集（hyacinth_seg / hyacinth_ls_v10 / **hyacinth_ls_v11** 等）
  - `data/` Label Studio 原始导出与中间件
  - `runs/` 训练产物（checkpoints、日志；已 gitignore）
  - `weights/` 共享预训练权重（SAM、YOLOv8、SegFormer-B2）
- 脚本大多以**项目根为工作目录**运行（相对路径 `runs/`、`datasets/` 从根解析）。

## 快速入口

```bash
# TWAU-Net 训练（现行论文方法）
cd twa_unet && PYTHONPATH=scripts python train.py --direction lower --window-size 4

# SegFormer-B2 训练（旧论文方法）
python segformer/scripts/segformer_train.py        # 6 类
python segformer/scripts/segformer_train_v10.py    # 5 类新标签

# YOLO+SAM 推理服务（CPU Docker 见 Dockerfile）
python yolo_sam/main.py --source <图片>             # CLI
python yolo_sam/serve.py --yolo <pt> --sam <pth>    # HTTP API

# Web 对比服务（按路线拆分）
python yolo_sam/app.py      # YOLO/GLI/SAM 三模式   → :5000
python segformer/app.py     # SegFormer 两模式       → :5001
```

## 安装

```bash
pip install -r requirements.txt
npm install   # 可选：docx 文档生成依赖
```

详细说明见各路线目录下的 README。
