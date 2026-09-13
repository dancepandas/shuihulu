figure_id: fig1_framework
core_claim: 单阶段 SegFormer-B2 以一次前向同时完成水葫芦与五类干扰地物的六类像素级分割，数据/损失/训练三类针对性设计共同支撑该能力，且无需检测框提示
backend: matplotlib   # 环境无 TikZ/LaTeX 中文链；matplotlib 可复现、三格式矢量、中文友好（skill §M reproducible matplotlib fallback）
style_reference: CS 会议论文结构图惯例（PlotNeuralNet 式三维特征块 / SegFormer 原文图式布局），仅借鉴构图与立体块画法，matplotlib 原创重绘
panels:
  - id: input
    type: image
    defends: "输入为真实无人机河道影像（512×512）"
  - id: encoder_stack
    type: schematic_3d_blocks
    defends: "MiT-B2 编码器四阶段画为三维特征块：空间尺寸递减（1/4→1/32）、通道深度递增（C=64→512）、块数 ×3/×4/×6/×3"
  - id: decoder
    type: schematic
    defends: "全 MLP 解码器：逐层 MLP → 上采样对齐 → 融合，四支路汇合"
  - id: score_slabs
    type: schematic_3d_blocks
    defends: "六类分数图画为 6 片类色薄板，衔接解码器与输出掩膜"
  - id: output
    type: image
    defends: "单前向输出真实六类像素级分割掩膜（背景/船只/桥梁/岸基建筑/水葫芦/树木）"
  - id: designs
    type: compact_markers
    defends: "三类针对性设计（数据增强/联合损失/两段式训练）以箭头汇入主干"
archetype: schematic_composite
hierarchy:
  overview: "左→右主链：输入影像 → MiT-B2 四阶段特征块 → MLP 解码汇合 → 六类分数薄板 → 六类掩膜"
  deviation: "输出面板嵌入真实分割结果（六类彩色掩膜叠加），论证像素级分割能力"
  relationship: "三个设计模块以箭头汇入解码器，表明其作用位置"
hero_panel: encoder_stack   # 三维特征块组视觉最重
export: [pdf, svg, png]
source_data: datasets/hyacinth_ls/images/train/ls_0385.jpg + runs/segformer/segformer_b2_ls+v9 推理
stats_on_figure: "各阶段 1/4~1/32 · C=64/128/320/512 · ×3/×4/×6/×3；输出侧 27.35 M 参数 · 109.5 MB · 7.4 ms/图；框架示意无需误差棒"
