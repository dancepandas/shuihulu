# LWH-Seg — SegFormer-B2 语义分割（旧论文路线）

以轻量级层次化 Transformer **SegFormer-B2**（MiT-B2 编码器 + 全 MLP 解码器）为分割主干的
单阶段水葫芦语义分割方法（旧论文《基于 SegFormer-B2 的水葫芦轻量化语义分割研究》）。
现行论文已由 [twa_unet/](../twa_unet/) 取代，本路线作为基线与对比保留。

## 两套标签体系

| 体系 | 训练脚本 | 类别 |
|------|----------|------|
| 6 类（旧） | `scripts/segformer_train.py` | 背景/船只/桥梁/岸基建筑/水葫芦/树木，512×512 |
| 5 类（新） | `scripts/segformer_train_v10.py` 及 finetune 系列 | water/water_hyacinth/hard_structure/shore_vegetation/other_aquatic_vegetation，640×640 |

## 目录

```text
app.py                  # Web 推理服务 (SegFormer / SegFormer-v10 两模式, :5001)
scripts/
  segformer_train.py                # 6 类训练 (论文方法)
  segformer_train_v10.py            # 5 类新标签训练 (两段式 + 加权 CE+Dice)
  segformer_finetune_7only.py       # 仅用 project7 数据微调
  segformer_finetune_v11_fullscreen.py  # v11 全幅微调 (TWA 论文的 B6 基线)
  eval_v10_testdata.py              # v10 模型在测试数据上的可视化验证
  eval_paper_models.py              # 旧论文各方法评测
  preanno_v10_ls.py                 # SegFormer-v10 对 LS project 10 笔刷预标注
  cover_annotations_v11b.py         # v11b 标注覆盖可视化
  _dryrun_v10_pred.py / _view_v11b.py   # 调试/预览
  build_docx_shuili.py              # 旧论文 md → docx
  build_figs_nature.py / fig1_framework.py / renumber_refs.py   # 旧论文图件/参考文献
doc/                    # 旧论文 md/docx + 投稿稿 + 附件
  figs/_archive_segformer/          # 旧论文图档
outputs/                # figures/ + v10_test_vis/ + v11b_test_vis/
```

## 训练

```bash
python segformer/scripts/segformer_train.py       # 6 类, 主干 weights/segformer_b2, 输出 runs/segformer/segformer_b2_v2
python segformer/scripts/segformer_train_v10.py   # 5 类, ADE20K 预训练, 输出 runs/segformer/segformer_b2_ls_v10
```

- 数据：`datasets/hyacinth_seg_p7p9`（6 类）/ `datasets/hyacinth_ls_v10`、`hyacinth_ls_v11`（5 类）
- 预训练权重：`weights/segformer_b2`、`weights/segformer_b2_seg`（原地未动）

## Web 推理

```bash
python segformer/app.py   # → http://localhost:5001
```

- `segformer`：6 类旧体系（`runs/segformer/segformer_b2_ls+v9`，512×512）
- `segformer_v10`：5 类新体系（`runs/segformer/segformer_b2_ls_v11b`，640×640）
