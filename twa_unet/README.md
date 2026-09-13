# TWAU-Net — U-Net + 三角窗注意力（现行论文路线）

基于 U-Net 与三角窗注意力机制（Triangular Window Attention, TWA）的水葫芦语义分割方法。
投稿目标：《水利水电技术（中英文）》。5 类标签体系（water / water_hyacinth / hard_structure / shore_vegetation / other_aquatic_vegetation），数据集 `datasets/hyacinth_ls_v11`（内部 681 张，论文口径 2300 余张）。

## 目录

```text
model/                  # 模型定义 (自 runs/twa_u_net/model 移入版本控制)
  twa_u_net.py          #   TWAUNet 主网络 (ResNet34 编码器 + 4 跳跃连接嵌 TWA)
  twa_block.py          #   TWA 模块 (三角掩码窗口注意力, 双残差)
  attn_modules.py       #   CBAM / Triplet 注意力 (基线用)
train.py                # 训练入口 (产物输出至 runs/twa_u_net/<tag>)
scripts/
  baseline_seg_data.py / baseline_seg_train_v11.py / baseline_seg_eval.py
                        # 表2 基线 (B0-B5) 数据/训练/评测
  baseline_seg_train.py # 旧版基线训练 (6类)
  run_baselines_v11.bat / continue_after_b5.sh   # 基线批量训练
  _perclass_eval.py / _perclass_eval_b023.py / _reeval_A5_check.py
                        # 逐类 IoU / 混淆矩阵 / Wilcoxon 复测
  build_vis_analysis.py # 图7 场景对比 + 图3 TWA 热图
  gen_attn_x4_figs.py   # 图4/图5 CBAM×4 / Triplet×4 热图
  _regen_attn_figs.py   # 仅重绘 3 张热图的最小入口
  gen_analysis_figs.py  # 图6 消融柱状图 / 图8 场景 IoU 柱状图
  gen_structure_svgs.py # 图1/图2 结构图 SVG
  make_hero_pred.py     # hero 图 (流程图用)
  build_docx_twaunet.py # md → 投稿 docx 流水线 (勿直接重跑, 会覆盖手改)
  _extract_docx_text.py # docx 正文只读导出 (审稿用)
doc/
  论文_基于U-Net与三角窗注意力机制的水葫芦语义分割.md   # md 源稿 (已滞后于 docx 手改)
  基于 U-Net 与三角窗注意力机制的水葫芦语义分割方法.docx  # ★ 唯一权威稿 (作者手改版)
  review/               # Stage3-5 审稿报告 + _manuscript_extract.txt
  figs/                 # 图1-图8 png/pdf + hero/ + 结构图/
  《水利水电技术（中英文）》中文稿件格式模板+2023+(1).docx
```

## 训练

```bash
cd twa_unet
PYTHONPATH=scripts python train.py --direction lower --window-size 4 --epochs 80 --tag ablation_A5_lower_win4_v11
```

- 数据：`datasets/hyacinth_ls_v11`（640×640，5 类）
- 损失：类别加权 CE + Dice（λ=0.5）；AdamW lr 6e-5 + 余弦退火
- 训练产物（checkpoint/日志）输出到项目根 `runs/twa_u_net/<tag>/`

## 论文工作流注意

- **docx 为唯一权威稿**，md 已滞后于手改；重跑 `build_docx_twaunet.py` 会覆盖手改，非作者明确要求不要重跑。
- 审稿流程见 `doc/review/Stage5_复审报告.md`（当前判定 Minor Revision，待改项 C1-C6/M1/M2/M7）。
- 已知构建脚本 bug：`build_docx_twaunet.py` 的 `render_math()` 不处理 `\sqrt`（C6）。
