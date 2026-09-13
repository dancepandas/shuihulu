# Stage 3 评审报告汇编 — TWAU-Net 水葫芦语义分割论文

- **评审对象**：`doc/论文_基于U-Net与三角窗注意力机制的水葫芦语义分割.md`（投稿 docx：`doc/论文_水利水电技术投稿稿_TWAU-Net.docx`）
- **目标期刊**：《水利水电技术（中英文）》
- **评审日期**：2026-09-03
- **流水线**：academic-pipeline Stage 2.5（完整性预检）+ Stage 3（五方评审），full 模式
- **评审方式**：6 个独立代理并行评审，互不通气；编辑合成见 `Stage3_编辑决定与修订路线图.md`
- **说明**：报告三（方法学）由该代理以英文输出，按原样归档未译

---

## 0. 作者预决策与主编核对事实（合成输入）

### 0.1 作者预决策（2026-09-03，评审返回后用户确认）

1. **去掉"轻量化"定位**：该定位沿袭自上一篇 SegFormer 论文，本文不再强调。标题删"轻量化"，摘要/关键词/结论/正文统一调整；效率论述只保留模块级相对表述（TWA 注意力计算量约为方形窗口的 1/4、时延低于 CBAM）。
2. **方向性表述降格**：只陈述"三角窗注意力在结构上引入方向敏感性、实验中下三角配置最优"，删除"下三角聚合左上上下文与叶缘自左上向右下走向一致"之类的因果机理解释（2.2 末段、3.4 其二相应改写）。
3. **图 2 热图解释按真实模式改写**（见 0.2）。

### 0.2 主编代理对两图的独立核对

- **图 1（fig_compare.png）**：30 个单场景 IoU 标注与正文矩阵完全一致（此前"图待重出"事项已完成）；但金框标注有错——第 1/3/4 行金框落在 U-Net-R18 / U-Net+Triplet / U-Net-R18 上，而按图题"每行 IoU 最高者标金框"，五行最高者均为 TWAU-Net（0.773/0.887/0.690/0.901/0.769），金框应全部落在 TWAU-Net 列。出图脚本"每行最高"判定疑似使用了旧一轮数值，需修正后重出。
- **图 2（fig_attention_twa.png）实际模式**：s1/s2（H/4、H/8）细粒度高响应落在水葫芦斑块、岸边植被与硬质结构边缘（硬质结构列中桁架桥与码头轮廓被细碎亮斑勾出），开阔水面整体暗、仅右侧岸带亮；s3/s4（H/16、H/32）响应收敛为大块语义区块——桥梁/码头/建筑群、岸边林带、连片植被。岸边植被带列中部细碎水葫芦斑块在深层为低响应。正文"深层聚焦于水葫芦主体及其与背景的交界"与图不符。

### 0.3 环境约束

- 作者单卡 RTX 2060 6GB，算力有限；
- WebSearch 不可用，"补文献"类建议只能给方向由作者自查；
- 本地 `runs/` 目录为早期实验轮次，与论文数字无关（论文数字为作者服务器实测权威值）；
- 投稿 docx 已含英文题名/结构化英文摘要/英文关键词/中英双语图表题、公式编号 (1)-(4)、引言编号 0 起；评审代理只看了 md 源稿，此类意见在 docx 中已解决。

---

## 1. 报告一：完整性核查（Stage 2.5 预检）

### 核查结论

**PASS**（无 P0 问题；共 2 项 P1、13 项 P2）

补充说明：正文 [n] 引用严格按 1→17 递增、无前向引用/缺号/重号，文末 17 条与正文一一对应；图 1/图 2、表 1/2/3 均被正文引用，且 `doc/figs/fig_compare.png` 与 `doc/figs/fig_attention_twa.png` 文件实际存在，相对路径可解析；公式掩码条件与 2.2 节"下三角关注左上方"的文字描述经坐标约定验算数学一致；摘要/引言/3.3/3.4/3.5/结论中的核心指标（0.7807、0.8729、0.5962、0.7487、27.20 M、34.52 ms、681 张）全部互相一致。

### 问题清单

**P1**

1. [3.4 表 3 vs 3.3 表 2] A0（Vanilla U-Net，Params 24.40 M）与 B1（U-Net ResNet34，Params 24.44 M）名义为同一配置，但三处数字均有出入：mIoU 0.7505 vs 0.7487（+0.0018）、WH IoU 0.5917 vs 0.5962（−0.0045）、Params 24.40 vs 24.44（−0.04 M）。参数量对固定结构是确定值、不随随机种子变化，0.04 M 的差异提示两表背后实现或统计口径不同；指标差异若为不同种子的重复训练，则属 135 张验证集的正常波动范围。且表 3 未注明 A0 的骨干网络，读者无从将 A0 与 B1 对应 — 建议在表 3 注或 3.4 正文中明确说明 A0 与表 2 B1 的关系（同配置独立重复训练，差异在随机波动内；或实现差异的具体来源），核实统一参数量口径，并在表 3 补注"A0：U-Net (ResNet34)，与表 2 B1 同配置"。
2. [2.2 / 2.3 display 公式] md 源稿 4 个 display 公式均无编号（docx 已加 (1)-(4)，md 源稿同步即可）。

**P2（13 项择要）**

1. "计算量约为方形窗口注意力的四分之一"为渐近近似：精确占比 ((M+1)/2M)²，M=4 时 39.1%、M=8 时 31.6%，"≈ M⁴/4"仅当 M≫1 成立。
2. [4 结论] "在保持较低参数量的同时"："较低"缺参照系，与 3.3"与 CBAM 的 24.44 M 基本相当"不完全呼应。
3. [2.2] "分别对应 token 网格上沿主对角线的两个相反聚合方向"与下一句语义重复。
4. [2.1] 公式 (1) 中 d_k、ConvBlock_{k-1} 下标体系及 d_4（bottleneck 地位）未在文中定义。
5. [引言 [9-12]] 该区间标注为"R-CNN 系列"，但 [10] YOLACT 不属 R-CNN 家族，区间覆盖不严格。
6. [图 1 图题] "最优结果以金框标出"未指明"逐场景行、按水葫芦类 IoU 判定"。
7. [3.5] 图 2 的 5 列与图 1 的 5 个场景的对应/顺序关系未说明。
8. [表 2 注] 时延测试未注明输入尺寸（应补 640×640）。
9. [3.5] 场景级 IoU 为 3 位小数，全文主指标均为 4 位，不统一。
10. [表 2] mAcc 列在摘要、3.3 与结论中均未被提及或分析。
11. [摘要] "高于……的 U-Net"未注明骨干（ResNet34）。
12. [参考文献] 会议论文条目缺出版地与出版者（如 [13] 可补"Cham: Springer"）。
13. [全文] 无英文标题/摘要/关键词（docx 已解决）。

### 验算附录（关键项）

- 3.3 提升量：0.7807−0.5962=0.1845（18.45 pp）✓；0.8729−0.7487=0.1242（12.42 pp）✓；vs CBAM：0.0315（3.15 pp）✓、0.0165（1.65 pp）✓；34.52<52.72 ✓；27.20−24.44=+2.76 M（+11.3%）✓
- 3.5 场景矩阵反推最优基线：0.773−0.003=0.770、0.887−0.034=0.853、0.690−0.066=0.624、0.901−0.008=0.893、0.769−0.042=0.727，与"领先幅度由 0.003 至 0.066 不等"及分组叙述一致 ✓
- 表 1 训练集占比加和 100.00 ✓、验证集 100.00 ✓；546+135=681 ✓、546/135≈4.04≈"约 4:1" ✓
- 表 3 A5 与表 2 Ours 三项完全一致 ✓；2.3 的 5.78%/3.40% 与表 1 一致 ✓
- 掩码有效元素 [M(M+1)/2]²：M=4 时 100/256=39.1%、M=8 时 1296/4096=31.6% → "≈M⁴/4"为渐近近似
- 掩码方向：按 i=rM+c、行自上而下列自左向右递增，A^lower 条件 r_j≤r_i 且 c_j≤c_i 恰为"query 自身+左上方（含同行左侧、同列上方）"，与 2.2 文字及摘要一致 ✓；A^upper 同理 ✓
- A0 vs B1：ΔmIoU +0.0018、ΔWH −0.0045、ΔParams −0.04 M → 需论文说明

---

## 2. 报告二：主编 EIC（总体 62/100）

### 总体评价

本文以无人机航拍水葫芦影像的像素级识别为切入点，将超分辨率领域的三角窗注意力迁移至 U-Net 分割框架，并配套自建 681 张五类标注数据集，选题正合本刊"智慧水利/遥感监测"栏目对"问题来自河湖管理实际"的取稿偏好，对比与消融实验完整，可信度尚可。但稿件存在一处一眼即穿的硬伤——标题、摘要、关键词、结论四处使用"轻量化"，而表 2 中 TWAU-Net 参数量 27.20 M 与模型体积 103.93 MB 均为全表最大、时延 34.52 ms 仅列第四，标题与数据自相矛盾；叠加数据集采集信息空白、业务落地论述薄弱，当前状态达不到直接送外审的标准，须实质性修改后复审。

### 优点

1. [引言 第 1 段] 需求链条完整：从"阻塞航道、影响行洪排灌"到"面积核算与打捞调度需要的是像素级的覆盖边界，而非'有无'的定性判断"，把方法选择的必然性立在业务需求上。
2. [摘要、引言末段] 创新点表达集中，主编 5 分钟内可完整抓住贡献。
3. [2.3 末句、3.2] 对比公平性交代到位：损失与基线一致、相同划分、ImageNet 预训练，基线覆盖无注意力/通道-空间注意力/自注意力三代方案。
4. [3.4 表 3、3.5 图 2] 消融同时覆盖方向形式与窗口尺寸两个设计自由度，并用去几何先验的注意力热图支撑机理叙述，完成度高于本刊同类来稿平均水平。
5. [3.5 末段] 作者主动限定可视化结论边界，对单场景 IoU 波动不做过当解读。

### 问题清单

**P0**

1. [标题/摘要/关键词/结论] "轻量化"与表 2 数据正面冲突；结论"在保持较低参数量的同时"更与 3.3 自相矛盾，结论末句自认"下一步……进一步压缩模型"等于承认当前并未轻量化。**建议改题**《基于 U-Net 与三角窗注意力机制的水葫芦语义分割方法》，全文"轻量化"统一改为"高精度/精细化"叙事，效率论据仅保留模块级相对表述。
2. [全文] 缺英文题录与双语图表题（**docx 已解决**）。
3. [3.1] 681 张影像未交代采集河段/地域、时间跨度、无人机型号、飞行高度与 GSD、相机参数——外审必问，须补齐。

**P1**

4. 业务价值止步定性：建议增加业务量化环节或"业务应用分析"小节。
5. 水利行业文献仅 [1][3][5]，建议补 3-5 条国内水利类近三年河湖遥感/水葫芦监测文献（含本刊）。
6. [3.4 末句] A5 选型理由一句带过有"挑指标"观感，建议在 3.1 前置声明选型准则。
7. 单一自建数据集、无跨河段/跨时相泛化验证，未提供混淆矩阵（"水葫芦"与"其他水生植被"类间混淆恰是最易被质疑处）。
8. 推理效率仅 RTX 2060 FP32 单点（与 P0-1 改题联动可降级为补充说明）。

**P2**

9. [表 3] 缺 mAcc/Size/Latency 列，与表 2 统计口径不齐。
10. [引言、2.2] "叶缘以斜向边缘为主""与叶缘自左上向右下的纹理走向一致"缺影像统计或文献支撑，属动机假设。
11. [图 1/图 2] 录用后需按本刊制版要求提供高清图源并核对图内中文标注清晰度。

### 决定预判

**Major Revision**。理由：选题与数据正中栏目，有录用基础；"轻量化"标题硬伤贯穿四处且与表 2 直接矛盾，属实质性修改；英文摘要（docx 已解决）、数据集采集信息、业务量化分析三项需补齐后方可送外审。

外审可能攻击点：轻量化名不副实；单一数据集且采集信息缺失、无泛化验证、创新说服力有限；A4 mIoU 高于最终模型而选型准则未前置、方向性动机无实证支撑。

改投意见：**不必改投**。仅当作者坚持轻量化技术路线（轻量骨干 + Jetson/CPU 端实测）时，改投《水利信息化》或《中国农村水利水电》更贴合；备选《人民长江》。

### 评分

创新性 65/100 | 期刊契合度 80/100 | 完成度 70/100 | 总体 62/100

---

## 3. 报告三：评审 1 方法学（总体 58/100；原文英文，按原样归档）

**P1**

- [2.1 Eq. (1) / 2.2 / Fig. 2] The recursive resolution is self-inconsistent: according to f₁∈R^{H/4×W/4×64} (consistent with the "s1·H/4" row label in Fig. 2) and d_{k-1}=CB(Concat(Up(d_k), f̃_k)), k=1..4 (with d₄ as the bottleneck), the first three levels of Up align with H/16, H/8, and H/4 respectively; but at k=1, Up(d₁) is H/2, which cannot be concatenated with f̃₁ at H/4; and "4 times 2× upsampling" starting from H/32 can only reach H/2, not an H×W output. Recommendation: verify against the code and rewrite — if the last hop is same-resolution concatenation Concat(d₁, f̃₁) (no Up) with ×4 upsampling by the segmentation head, correct Eq. (1) and change to "3 times + head upsampling"; if f₁ is actually the H/2 feature from conv1, correct 2.1 and the row label in Fig. 2.
- [2.2 / 3.3] "Reduced computation to about a quarter" is too strong for M=4: the effective element ratio = (1+1/M)²/4, which is 39.1% at M=4 and 31.6% at M=8, only asymptotically approaching 25%. Provide the exact values and weaken the attribution wording in 3.3.
- [Table 2/3] Missing two key controls: (a) no square window attention comparison with the same framework, same M — the core claim "triangular better than square" is only compared among internal TWA variants, and the latency advantage in 3.3 is compared against CBAM and cannot be attributed to the triangular mask; (b) lightweight Transformer segmentation baselines like SegFormer-B0 are missing. Add one training run each, and add a FLOPs column (thop, zero cost).
- [Table 3 A0 / Table 2 B1] Unexplained discrepancy between two rows of the same configuration and exposure of variance: the 0.18pp mIoU difference is the run-to-run variance of a single training, while key differences of the same magnitude like A1 vs A2 (0.22pp), A4 vs A5 mIoU (0.39pp) lack statistical support under a single seed. Clarify the seed/implementation relationship; add 2 seeds each for A1/A2/A4/A5 and report mean±std, or support the directional conclusions with per-image paired tests.
- [3.4 末句] A5 selection is potentially "post-hoc": A4's mIoU (0.8768) is higher than A5's (0.8729), and the composite criterion is given after seeing the results; also "highest IoU reaches 0.7807, highest mIoU reaches 0.8768" stitches together the optimal values of two configurations (no single configuration achieves both). Pre-declare WH IoU as the primary metric; add latency and per-class IoU for A4/A5; rewrite the stitching sentence.
- [3.1-3.3] Single validation set (135 images) used for both selection and reporting, single seed, no significance testing. Minimum-cost reinforcement: (a) zero training — per-image paired Wilcoxon/bootstrap 95% CI on 135 images (Ours vs B2); (b) zero training — FLOPs, per-class IoU, boundary metrics; (c) light retraining — 2 more seeds for Ours/B2 (and A4/A5); (d) stratified split of the 135 images into val/test (or an external test set from different sections/seasons). If none are executed, 3.3 and Section 4 must disclose limitations and weaken the conclusion to "on this validation set…".
- [3.1] The splitting protocol is not described, posing a leakage risk: it is not stated whether the 681 images are randomly split 4:1 per image or grouped by flight runs; near-duplicates among consecutive UAV frames would systematically inflate validation metrics; acquisition parameters (resolution/GSD/seasons/regions) are also not given.
- [Title/Abstract/Conclusion "lightweight"] vs Table 2: 27.20M and 103.93MB are the largest among all methods; 34.52ms≈29FPS is only quasi-real-time. Change to "accuracy-efficiency trade-off/quasi-real-time" wording (作者已预决策采纳改题).

**P2**

- [2.2] The effective region is a proper subset of the attention matrix lower triangular {j≤i} ("row and column each ≤" product-order region, block-staircase shaped, M=4 with 100 elements vs 136 in the lower triangle); "the name describes its triangular shape in the attention matrix" should be changed to "the block-triangular region within the lower triangle of the attention matrix"; whether r, c are 0-based is not stated.
- [2.1/2.2/2.3] Symbol completeness: the initialization of d₄ (bottleneck) and the input to the segmentation head are undefined; the Q/K/V projection, scaling, softmax, and 8-head dimension splitting within Attn are not given; the normalization approach for w_c=1/p_c is not specified; the four equations are unnumbered in md (docx 已解决).
- [3.4 A3/A4] The structure of "dual-direction concatenation" (parallel concat then projection or addition, channel changes) is not described and can only be inferred indirectly from 27.20→29.29M, making it irreproducible.
- [2.2/3.4] The directional mechanism argument is weak: for a "\"-oriented leaf edge, the upper-left (lower-triangle aggregation) and lower-right (upper-triangle aggregation) neighborhoods of a query are both on that edge, making the priors of the two directions symmetric; A1 is only 1.7pp higher than A2, the same magnitude as the noise in A0/B1. At zero training cost, compute a gradient dominant-direction histogram of water hyacinth boundaries, or weaken to a hypothetical statement.
- [3.1] Defining mAcc as "the arithmetic mean of recall for each class" does not match common notation (usually mean recall). Rename to mRecall or provide the formula.
- [表 2 注/3.2] Latency protocol incomplete: warm-up count, repetitions, statistics (mean/median), input confirmation of 640×640, eval+no_grad, CUDA synchronization; batch=2 BN statistics noise; whether encoder pretrained BN is frozen is not described. Supplement the protocol; if possible, add FP16/TensorRT latency.
- [3.5 图 2] Incomplete normalization: the formula for "column sum divided by uniform attention baseline" is not given (masked uniform vs. mask-free 1/M²); the colorbar has no numerical scale; whether "same colorbar across scales" is cross-scale or per-scale is not stated; the upsampling interpolation method for overlaying on grayscale images is not specified. The row/column labels and heatmap morphology are acceptable.
- [3.5] "Water hyacinth pixel ratio of about 1.3%" has no source: it is a single-sample ground truth statistic with no basis in the paper. Note "statistically calculated from that sample's ground truth" and provide a small table of WH ratios for the 5 typical scenarios.
- [3.1-3.4] Metric completeness: no complete per-class IoU table (which could explain DeepLabV3+'s collapse to 0.382 in hard scenes); no boundary metrics (boundary IoU/F1/HD95); no convergence curves for 80 epochs.
- [摘要/3.3] Reference bias: "increased by 18.45 and 12.42 percentage points" is relative to the weakest B1; provide both against the strongest baseline CBAM in the abstract.
- [全文] No code/data availability statement, no random seed, no implementation framework and baseline source.

**评分**：Innovation 65/100 | Technical Rigor 57/100 | Reproducibility 55/100 | Overall 58/100

总评：transferring the triangular window and improving WH IoU has practical value, and the mask definition is mathematically self-consistent; however, single validation set + single seed + missing key comparisons constitute a complete evidence chain, the "lightweight" positioning is inconsistent with the data. Major revision recommended, prioritizing zero-training-cost supplements (paired tests, FLOPs, per-class IoU, boundary metrics, Fig. 1 gold-frame fix) and 1-2 control training runs (square window attention, SegFormer-B0).

---

## 4. 报告四：评审 2 领域（总体 75/100）

### 总体评价

论文选题切中河湖管理业务痛点，引言"光谱指数→检测/实例分割→语义分割"的演进脉络清晰，消融实验对"迁移是否有效、方向孰优、窗口尺寸"三问形成了完整闭环，且对 TWA 来源的交代总体诚实。主要不足在于：贡献表述与文献[17]的界定存在自相矛盾、"轻量化"定位与自身实测数据不自洽、方法章无图且公式未编号、水利业务向文献缺位，属修改后可达发表水平的应用型论文。

### 优点

1. 引言技术演进梳理准确，"检测框不携带像素级覆盖信息""阈值难以跨河段泛化"等转折论证贴合水利业务真实约束。
2. 对 TWA 的来源交代诚实，消融实验（表 3）与三问一一对应，问题—实验—结论闭环完整。
3. 对比实验设计规范，统一损失与训练条件并专门声明"性能差异仅来自网络结构本身"。
4. 图 2 的热图归一化设计考究（"除以均匀注意力基线，以消除下三角掩码本身的几何梯度"），避免把掩码先验误读为学到的聚焦；3.5 对典型样本局限的自觉说明体现严谨态度。
5. 摘要具备四要素且数字恰当；公式推导正确；有效信息密度高。

### 问题清单

**P0**

1. [摘要、引言第 4 段、4 结论] 贡献表述与文献[17]自相矛盾且夸大：引言已写明三角掩码由 Ray 等[17]提出，主要工作第一条却是"**提出**基于主对角线的下三角与右上三角两种掩码形式"。改为："将 TWA 迁移至语义分割并在 U-Net 的 4 个跳跃连接处适配嵌入；通过消融实验确定分割任务中下三角方向与窗口尺寸 M=4 的最优组合"；摘要"所提 TWA 模块"改"经适配嵌入的 TWA 模块"。
2. [标题/摘要/关键词"轻量化" vs 表 2] 二选一：(a) 标题删"轻量化"（作者已采纳）；(b) 保留则必须补参照系并改写结论"保持较低参数量"句。
3. [表 2 B1 与表 3 A0] 口径不一致且未说明；此问题不解决，两表无法互校，审稿必被质询。

**P1**

1. ResNet34、CBAM、Triplet Attention 全文无一处引文，对比方法无法溯源——补三者原始文献（ResNet、Woo 等 CBAM、Misra 等 Triplet Attention）；AdamW、GroupNorm、ImageNet 建议联引。
2. 公式无编号（docx 已解决）；z=Attn(Q,K,V;A^lower) 信息量低可并入正文。
3. "1 引言"应改"0 引言"（docx 已解决）。
4. 方法章全无插图：建议补网络总体结构图（标注 4 个 TWA 嵌入位置）与 M=4 三角掩码示意图；2.2 开头加白话铺垫。
5. 动机与评价指标错位：引言以叶缘/桥墩/船体三种方向性纹理立论，但全文仅报 WH IoU 与 mIoU，无逐类 IoU；补逐类 IoU 或收缩动机。
6. 数据集采集与标注信息不足（机型/相机/航高/GSD/河段/时间跨度/标注工具与质检）。
7. 结论重复摘要且缺局限性分析；按"方法要点+适用条件+局限+展望"改写。
8. 文献三方向缺位：(a) 水利业务语境（"清四乱"、河长制遥感监测、水面漂浮物识别）；(b) 2021—2025 水葫芦/入侵水生植物遥感识别更新；(c) 轻量化语义分割（随改题可弱化）。

**P2（22 条择要）**

引言两处三层长句拆分；"由…转变为由…"杂糅；token/query/key 中文注释；2.1 递推只到 d0、未说明如何恢复 H×W；"两种掩码的名称描述其在注意力矩阵中的三角形状"不准确（应为块三角/网格三角）；"式中/其中"不统一；3.1"涵盖/包含"重复；6e-5 改 6×10⁻⁵；表 2 表题混入英文、"Ours"改"本文方法"、"Vanilla"改"基准 U-Net（无 TWA）"、表头按 GB 3101；小数位统一 4 位；2.1 的 f1-f4 与图 2 的 s1-s4 符号不统一；"掩码/掩膜"混用；WH 缩写未解释；[8] 软件宜标 [CP]；[13] 缺 Proceedings of 与 LNCS 卷次；摘要末句"验证了…对性能的影响"空泛；Dice 损失未给出处；3.4 应直面 A4 mIoU 更高补取舍理由；"四分之一"可选补方窗对照。

### 评分

文献综述 76/100 | 领域贡献 74/100 | 写作与规范 74/100 | 总体 75/100

说明：领域贡献按"模块迁移+应用验证"评定——三处表述修正后，其"迁移+系统消融+水利场景验证"的贡献对《水利水电技术（中英文）》读者成立且有实用价值。

---

## 5. 报告五：评审 3 交叉视角（总体 65/100）

### 总体评价

论文选题精准对位水利业务真实痛点，需求刻画在同类工作中属上乘。但价值链条在实验环节断裂：引言承诺的打捞处置、面积核算、生态调度三项业务产出，实验止步于验证集 IoU，全文没有出现一个业务语言的指标（面积误差、米制边界偏差、吞吐量），也没有局限性交代。补上零成本的面积误差/边界偏差分析与一段业务衔接讨论后，该文才能真正立住。

### 优点

1. 需求刻画准确：正确识别了检测类方法与打捞调度业务之间的真实鸿沟。
2. 标签体系有业务意识："其他水生植被"与"岸边植被"的区分正是打捞台账中最易混淆的两类。
3. 可视化按业务场景分层，诚实标注"示意性典型样本、单场景波动大"。
4. 实测环境交代可直接换算采购决策（RTX 2060 6GB、batch=1、FP32）。
5. 正视类别不均衡，加权 CE + Dice 符合小面积入侵物种监测的真实数据分布，且与基线一致保证公平。

### 问题清单

**P0**

- [3.1+3.3] 实验终点止于 IoU，业务承诺无一被验证：建议补两项零成本分析：①逐张面积估算相对误差分布（中位数、P90，按真值面积分桶 <100 / 100–1000 / >1000 m²）；②预测与真值边界的 HD95 或平均对称边界距离，乘以 GSD 换算为米。
- [4 结论] 全文无局限性段落：建议结论前增加 150–250 字（单一河段、跨季节未验证、640×640 切片推理边界效应、"其他水生植被"类内多样性覆盖有限，纯文字即可）。

**P1**

- "轻量化"主张与证据不符（作者已预决策采纳；可限定为"注意力计算的轻量化"）。
- 34.52 ms 未翻译成业务吞吐：单卡 RTX 2060 每小时约可处理 10 万张 640×640 切片，可支撑一个巡检架次当日批处理；边缘端部署需量化前置。
- 压缩路径顺序次优：先 FP16/INT8 量化与 TensorRT/RKNN 导出（103.93 MB FP32 INT8 后约 26 MB），再谈换轻量骨干（换骨干后 TWA 增益需重新验证）。

**P2**

- 识别结果与河长制/河湖管理平台数据链路未讨论（切片预测→回投 CGCS2000→连通域矢量化→叠加河道管理范围线与打捞台账生成工单→多期复飞面积时序）。
- 业务约束可作先验：水域岸线矢量约束后处理（可压低岸边植被带误检）、NDVI 作弱监督先验、打捞台账作主动学习选片（方向性建议）。
- 数据集采集条件零交代（同 EIC P0-3）。

### 评分

业务相关性 72/100 | 应用价值 65/100 | 讨论充分性 48/100 | 总体 65/100

---

## 6. 报告六：魔鬼代言人

### 最强反驳

论文以"轻量化"为题，但按其自己的表 2，参数量（27.20M）与模型体积（103.93MB）最大的恰是 TWAU-Net，时延也高于 DeepLabV3+ 与两个 U-Net 基线；结论更自认将改用 MobileNet、ShuffleNet"进一步压缩模型"——标题主张被论文自身数据证伪。支撑核心叙事的方向性解释同样站不住："叶缘自左上向右下分布"无任何方向统计佐证，且被论文自己的水平翻转与随机旋转增强在训练分布层面直接抵消；A1 与 A2 的 mIoU 差距（0.22 个百分点）与论文内部两次名义上同为 U-Net 的结果差（0.18 个百分点）处于同一量级，"下三角优于右上三角"完全可能落在单次训练噪声之内。而"下三角 vs 方形窗口"这一最关键对照从未做过——增益究竟来自三角方向性，还是仅仅来自"加了任何注意力加正则"，论文无法回答。审稿人只需指出这三点，标题、机理与实验设计三根支柱同时动摇。

### 问题清单

**CRITICAL C-1 "轻量化"标题与表 2 全面矛盾**
- 辩护充分性：难以辩护（就措辞而言）。效率优势只在 CBAM 一个参照下成立。
- 补强：零训练成本——标题删"轻量化"（作者已采纳）；可选补 SegFormer-B0 或 MobileNetV3-UNet 一次训练。

**CRITICAL C-2 方向性机理为事后叙事，且与自身数据增强自相矛盾**
- (a) 无叶缘方向统计支撑"自左上向右下"前提；(b) 水平/垂直翻转与随机旋转使固定对角先验在训练分布中失去意义；(c) A1 与 A2 差 0.22pp 与 A0/B1 差 0.18pp 同量级，n=1 对 n=1。
- 论文可用辩护：引言把方向比较框定为待检验问题（以实验而非机理为选型依据）；3.4 为实测数据；下三角保留 token 自身信息，方向缺损可由多层堆叠部分恢复。
- 补强：①因果断言降格为假设（作者已采纳，零成本）；②真值边界方向直方图（结构张量，约半天，无训练）；③A1/A2 各补 2 种子。

**MAJOR M-1 消融选型是"用最终指标反推"的事后标准**
- 方向×窗口网格不完整：方向比较只在 M=8 做，M=4 缺 upper——"下三角在最终窗口尺寸下仍最优"从未被检验。
- 辩护：以 WH IoU 优先符合应用逻辑，A5 对 A4 的 WH 优势达 6.75pp 远大于 mIoU 0.39pp 的让步，参数少 2.09M，A5 位于帕累托前沿。
- 补强：零成本在 3.1 前置声明"以 WH IoU 为首要选型指标"；低成本补 upper-M=4 一次训练。

**MAJOR M-2 A0 与 B1 数字不一致，暴露主表与消融表非同批产出**
- 补强：加一句话注明实现差异来源（零成本）；或反向利用——在 3.4 注明"两处数字差异可作为运行间波动的粗略量级（约 0.2pp mIoU）"。

**MAJOR M-3 统计稳健性：单次训练、无方差、验证集兼作选型集**
- 辩护：协议对称；头部差距远超内部波动量级（对 CBAM mIoU 优势 1.65pp 约为内部同构差异 0.18pp 的 9 倍）。
- 补强：零成本逐图配对检验（135 张验证图上 Ours vs CBAM、A1 vs A2 的配对 bootstrap/Wilcoxon，仅需已存预测）。

**MAJOR M-4 数据集规模与单一性，采集元数据完全缺失**（同前）；验证集 WH 占比 3.40% 低于训练 5.78%，存在分布漂移且未讨论。

**MAJOR M-5 图 2 热图证据力不足，缺对照组**
- 列和把 query-key 相对几何完全平均掉，原则上不可能显示"方向性"；"细粒度叶缘状"是任何训练网络浅层的标配输出；实测"硬质结构为主"列 s3/s4 的亮区主要落在桥梁钢结构与码头面域而非水葫芦，与"聚焦于水葫芦主体及其与背景的交界"表述不符（与主编核对一致）；"方向性特征"三字超出该图能证明的范围；无 A0 或方窗对照热图。
- 辩护："内容性聚焦"的说法站得住，均匀基线归一化是少见的严谨做法。
- 补强：零成本删改措辞；低成本补一列 A0 或方窗同规格热图（对已有 checkpoint 挂钩子重跑推理，无需重训）。

**MAJOR M-6 相对 CFAT 的增量性：A+B 迁移，且缺方向性同族对照**（axial attention、strip pooling、Swin 移位窗口）。低成本补 U-Net+方形窗口注意力可一石三鸟（同时回应 C-2、M-5）。

**MINOR m-1** "1.3%"无出处 → 改为"图 1 该硬质结构场景样本的人工真值中水葫芦像素占比仅约 1.3%"（零成本）。
**MINOR m-2** DeepLabV3+ 异常低（0.7294<0.7415）且无讨论 → 补一句解释（小样本下 ASPP 大空洞率更易欠拟合）或给 B4 单独 lr 扫描。
**MINOR m-3** 时延归因链断裂与 CBAM 时延反常（1/4 是相对方窗而非 CBAM；CBAM +25ms 可能源于实现细节）→ 改写 3.3 该句并补测量协议（零成本）。

### 专项：最强论断的最小补证清单

核心论断："下三角方向契合叶缘斜向纹理走向，是 TWAU-Net 性能优势的来源"。若审稿人要求补证：①真值边界方向直方图（约半天，零训练）；②A1、A2 各 3 种子（2-3 天 GPU）；③补 upper-M=4 与 U-Net+方形窗口各 1 次（1 天）；④逐图配对检验（零成本）。若只允许做一项：第 ④ 项。

### 被忽视的替代解释

1. 增益来自"泛注意力+正则"而非"三角方向性"（最直接的对照从未跑过）。
2. 数据集特异性偏倚：无人机朝向与水流漂移方向使叶缘恰以"\"向为主，换河段可能翻转。
3. 少数样本驱动：验证集 WH 仅 3.40% 且高度聚集，优势可能由少数几张水葫芦大图主导。
4. 时延差异的实现假象：CBAM +25ms、TWA +6.8ms 可能主要反映算子实现质量。

### 观察（非缺陷）

1. 图 2"列和除以均匀基线"归一化是同类可视化中少见的严谨做法，建议在正文明确表述为方法学细节。
2. 3.5 末段对选择性呈现风险有自觉，是全文防御姿态最好的段落。
3. 摘要与表 2 如实列出对己不利的 Params/Size/Latency——这使"轻量化"争议停留在措辞层面而非数据诚信层面。
