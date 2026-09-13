const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, LevelFormat,
} = require("docx");

const C = {
  primary: "1B6B4A", secondary: "2E8B57", accent: "3CB371",
  light: "E8F5E9", blue: "1565C0", dark: "1A1A2E", gray: "666666",
  white: "FFFFFF", border: "C8E6C9",
};

const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 160 }, children: [new TextRun({ text, font: "Microsoft YaHei", bold: true, size: 34, color: C.primary })] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 120 }, children: [new TextRun({ text, font: "Microsoft YaHei", bold: true, size: 28, color: C.secondary })] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 80 }, children: [new TextRun({ text, font: "Microsoft YaHei", bold: true, size: 24, color: C.dark })] });

const p = (text) => new Paragraph({ spacing: { after: 100, line: 360 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, color: C.dark })] });
const boldP = (text) => new Paragraph({ spacing: { after: 80, line: 360 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, bold: true, color: C.dark })] });
const highlight = (text) => new Paragraph({ spacing: { before: 120, after: 120, line: 360 }, indent: { left: 360, right: 360 }, border: { left: { style: BorderStyle.SINGLE, size: 12, color: C.accent, space: 8 } }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, color: C.primary, italics: true })] });
const bullet = (text, ref) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 340 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, color: C.dark })] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const spacer = (pts = 200) => new Paragraph({ spacing: { before: pts } });

const cell = (text, opts = {}) => {
  const isHeader = opts.header || false;
  return new TableCell({
    width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
    borders: { top: { style: BorderStyle.SINGLE, size: 1, color: C.border }, bottom: { style: BorderStyle.SINGLE, size: 1, color: C.border }, left: { style: BorderStyle.SINGLE, size: 1, color: C.border }, right: { style: BorderStyle.SINGLE, size: 1, color: C.border } },
    shading: isHeader ? { fill: C.primary, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT, children: [new TextRun({ text, font: "Microsoft YaHei", size: 20, bold: isHeader, color: isHeader ? C.white : C.dark })] })],
  });
};
const row = (cells) => new TableRow({ children: cells });

// COVER
const cover = [
  spacer(3600),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "水葫芦智能识别模型", font: "Microsoft YaHei", size: 52, bold: true, color: C.primary })] }),
  spacer(200),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "研发进展报告", font: "Microsoft YaHei", size: 32, color: C.gray })] }),
  spacer(400),
  new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.accent, space: 8 } } }),
  spacer(600),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "2026年5月 — 2026年8月", font: "Microsoft YaHei", size: 24, color: C.gray })] }),
];

// 1. OVERVIEW
const overview = [
  h1("一、项目概述"),
  p("本项目面向河道水葫芦（Water Hyacinth）入侵监测需求，基于深度学习技术构建了一套从数据标注、模型训练到生产部署的完整智能识别系统。项目自 2026 年 5 月 18 日启动，至 8 月上旬完成核心算法研发与工程化交付，历时约三个月。"),
  p("技术路线选用 YOLOv8m-seg 实例分割模型作为前端检测器，Meta SAM（Segment Anything Model）作为后端像素级精修模块，并创新性地引入 GLI（Green Leaf Index）植被指数实现训练数据的自动标注与自迭代。整套方案仅需 RGB 可见光影像，无需多光谱传感器，完全适配消费级无人机。"),
  p("截至报告撰写时，模型已迭代四个大版本（V1→V2→V3→V4），训练数据从零增长至 2000+ 张，检测精度（mAP50）从 0.472 提升至高精度收敛。模型已导出为 ONNX 格式，可在纯 CPU 环境下运行，满足生产部署要求。"),
];

// 2. PHASE 1 - 5.18-5.25
const phase1 = [
  h1("二、第一阶段：技术选型与基线搭建"),
  h2("时间：2026年5月18日 — 5月25日"),
  h3("2.1 完成的工作"),
  p("项目启动后，首先对水葫芦检测的技术路线进行了方案论证与可行性验证。考虑到水葫芦目标大小不一、形态不规则、常与河岸植被混杂等特点，确定采用两阶段级联架构：前端使用 YOLOv8m-seg 进行快速定位与粗分割，后端使用 SAM 进行像素级精细分割。"),
  boldP("技术原理 — YOLOv8m-seg"),
  p("YOLOv8 是 Ultralytics 于 2023 年发布的目标检测模型，其 seg 变体在检测框之外增加了一个并行的分割头，可同步输出目标边界框与粗粒度掩膜。模型主干网络为 CSPDarknet，颈部采用 FPN+PAN 特征金字塔融合多尺度特征，检测头为 Anchor-Free 解耦设计。m 版本参数量 2,722 万，FLOPS 104.7G，在精度与速度之间取得良好平衡。"),
  boldP("技术原理 — SAM"),
  p("SAM（Segment Anything Model）是 Meta AI 发布的通用图像分割大模型，基于 ViT-H 编码器（6.32 亿参数），在海量多样化分割数据上训练，具备强大的零样本泛化能力——即使从未见过水葫芦图像，也能根据一个简单的边界框提示（Box Prompt）生成高质量分割结果。"),
  p("在确定技术路线后，完成了 Python 开发环境搭建（ultralytics、segment-anything、PyTorch 等核心依赖），加载 COCO 预训练权重建立了基线模型 V1。为防止小样本过拟合，冻结了 FPN 特征金字塔的 P3 层（小目标检测层）。V1 未使用任何水葫芦专用训练数据，mAP50 仅为 0.472，但它验证了预训练权重对水葫芦场景具有基本的特征响应能力。"),
  h3("2.2 阶段产出"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [3000, 3000, 3026],
    rows: [
      row([cell("事项", { header: true }), cell("内容", { header: true }), cell("状态", { header: true })]),
      row([cell("技术方案"), cell("YOLOv8m-seg + SAM 两阶段级联"), cell("已确定")]),
      row([cell("开发环境"), cell("Python + PyTorch + ultralytics + segment-anything"), cell("已搭建")]),
      row([cell("基线模型 V1"), cell("COCO 预训练，冻结 P3 层，mAP50 = 0.472"), cell("已完成")]),
    ],
  }),
];

// 3. PHASE 2 - 5.25-6.15
const phase2 = [
  pageBreak(),
  h1("三、第二阶段：人工标注与模型冷启动"),
  h2("时间：2026年5月25日 — 6月15日"),
  h3("3.1 完成的工作"),
  p("基线模型 V1 的精度（mAP50 = 0.472）远未达到实用要求。深度学习模型从零到可用，必须经过领域数据微调。但水葫芦的像素级分割标注是一项高成本工作——标注员需要在每张无人机影像上逐像素描出水葫芦的轮廓，单张耗时 30 至 60 分钟。"),
  p("本阶段的策略是：先集中精力完成少量（约 200 张）极高精度的 polygon 级人工标注，用这批高质量标签训练第一个可用的水葫芦检测模型，为后续的自动标注方法提供质量基准。"),
  boldP("具体工作内容："),
  bullet("搭建 Label Studio 标注平台，设计标注规范与质量控制流程", "b1"),
  bullet("定义 5 类目标体系：Water Hyacinth（水葫芦）、Boat（船只）、Bridge（桥梁）、Structure（建筑）、Tree（树木）。多类别设计的目的不仅是扩展功能，更是为了在自动标注管线中利用非 WH 类排除干扰——河岸树木和绿色船只在颜色上与 WH 高度相似，单靠植被指数无法区分", "b1"),
  bullet("完成 191 张无人机航拍影像的人工 polygon 标注，覆盖小/中/大三种尺寸水葫芦，涵盖日间晴光、阴天、阴影等多种光照条件", "b1"),
  bullet("实施迁移学习训练：SGD 优化器（动量 0.937，基础学习率 0.01），遗传算法自动搜索最优超参数组合，13 种数据增强（HSV 调整、旋转、平移、Mosaic、Copy-Paste、RandAugment、随机擦除、翻转、缩放等），早停机制（50 轮无提升则终止）", "b1"),
  h3("3.2 训练结果 — V2 模型"),
  p("V2 的 mAP50 达到 0.773，相比 V1 的 0.472 提升 64%。这一结果验证了两个关键假设："),
  bullet("少量高质量人工标注（约 200 张）对模型冷启动是充分且必要的。高质量优先于大批量", "b1"),
  bullet("多类别联合训练有助于模型学习更具判别力的特征表示，间接提升了 WH 类的检测精度", "b1"),
  h3("3.3 阶段产出"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [3000, 3000, 3026],
    rows: [
      row([cell("事项", { header: true }), cell("内容", { header: true }), cell("状态", { header: true })]),
      row([cell("标注平台"), cell("Label Studio，支持 polygon 标注与格式互转"), cell("已搭建")]),
      row([cell("人工标注数据"), cell("191 张，5 类 polygon 级标注"), cell("已完成")]),
      row([cell("V2 模型"), cell("mAP50 = 0.773（+64%），5 类实例分割"), cell("已训练")]),
    ],
  }),
];

// 4. PHASE 3 - 6.15-6.30
const phase3 = [
  pageBreak(),
  h1("四、第三阶段：GLI 自动标注方法研究"),
  h2("时间：2026年6月15日 — 6月30日"),
  h3("4.1 问题背景"),
  p("V2 模型虽然精度可观，但仅靠 191 张训练数据，其泛化能力存在明显上限——对新河道、新光照条件下的水葫芦，检出率和边界精度均不稳定。扩充训练数据是提升模型泛化能力的必然选择，但继续依赖人工标注在经济上不可行（191 张已耗费约 100 人时）。必须找到一种自动化的标注方法。"),
  boldP("技术原理 — GLI 植被指数"),
  p("GLI（Green Leaf Index，绿叶指数）= (2G - R - B) / (2G + R + B)。其物理原理基于绿色植被的光谱反射特征：叶绿素在绿光波段（约 550nm）有反射峰，在红光（约 650nm）和蓝光（约 450nm）因吸收而反射率低。GLI 通过归一化比值放大这一差异——植被区域 GLI 值高，非植被区域 GLI 值低。与 NDVI 不同，GLI 仅需 RGB 三通道，完全适配消费级无人机。"),
  h3("4.2 方法对比实验"),
  p("自动标注的核心思路是：先找到图中所有绿色植被（GLI），然后剔除其中的非水葫芦部分（YOLO 识别的船只/桥梁/建筑/树木），剩余即为水葫芦候选。但具体实施中存在大量参数选择——用哪种植被指数？用固定阈值还是自适应阈值？识别到植被后如何排除干扰？"),
  p("本阶段开展了系统的对比实验："),
  bullet("植被指数选型：在 GLI、ExG（超绿指数）、CIVE（颜色植被指数）、COM2、VSSI 五种指数中进行对比。GLI 在阴影、高光、浑浊水体等困难场景下表现最稳定，确定为最终方案", "b1"),
  bullet("阈值策略：固定阈值（如 GLI > 0.12）在不同光照下表现差异大——阴影漏检、强光误检。Otsu 自适应阈值逐图寻找最优分割点，一致性和鲁棒性均优于固定阈值", "b1"),
  bullet("管线架构设计：确定了五步串行管线——GLI 二值化 → YOLO 排除非 WH → 形态学闭运算合并碎片 → YOLO-WH 交集约束过滤误检 → 轮廓标签入库", "b1"),
  bullet("参数筛选：50 余轮对比实验确定关键参数——GLI-YOLO 重叠阈值 0.25、形态学闭运算核 25×25 迭代 2 次、最小轮廓面积 50px、YOLO 置信度阈值 0.10", "b1"),
  h3("4.3 阶段产出"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [3000, 3000, 3026],
    rows: [
      row([cell("事项", { header: true }), cell("内容", { header: true }), cell("状态", { header: true })]),
      row([cell("植被指数方案"), cell("GLI + Otsu 自适应阈值，50+ 轮实验验证"), cell("已确定")]),
      row([cell("自动标注管线"), cell("GLI→排干扰→闭运算→交集约束，5 步流程"), cell("已设计")]),
      row([cell("关键参数"), cell("wh_overlap=0.25, close=25x25x2, min_area=50, conf=0.10"), cell("已定版")]),
    ],
  }),
];

// 5. PHASE 4 - 7.01-7.15
const phase4 = [
  pageBreak(),
  h1("五、第四阶段：自迭代训练"),
  h2("时间：2026年7月1日 — 7月15日"),
  h3("5.1 V3 — GLI 自动标注首次扩量"),
  p("基于 V2 模型 + GLI 管线，对新增的无人机航拍影像进行自动标注，生成约 800 张伪标签。经人工抽检（抽检比例约 10%）确认标签质量合格后，将这批伪标签与原 191 张人工标签合并，训练集扩充至约 1,000 张。"),
  p("V3 阶段的核心发现：GLI 固定阈值在阴影和强光场景下表现不稳定——阴影区 GLI 整体偏低导致漏检，强光水面反光导致误检。这一问题在 V4 中通过引入 Otsu 自适应阈值得到解决。V3 模型的泛化能力相比 V2 显著增强，对大面积、暗色水葫芦的识别能力明显提升。"),
  h3("5.2 V4 — Otsu 自适应阈值与互约束精标"),
  p("V4 在前代基础上引入三项关键改进："),
  boldP("改进一：Otsu 自适应阈值"),
  p("替代 V3 的固定 GLI 阈值。Otsu 算法通过最大化类间方差，为每张影像独立寻找 GLI 直方图的最优分割点。无论影像是在正午强光还是傍晚阴影下拍摄，植被与非植被的 GLI 分界点都能自适应调整，从根本上解决了固定阈值的场景适应性缺陷。"),
  boldP("改进二：YOLO-WH 交集约束"),
  p("这是自动标注管线中最关键的一道安全阀。GLI 只能回答「哪里是绿色植被」，无法区分绿色植被到底是水葫芦还是河岸树木。YOLO 的水葫芦检测结果（class=3）虽然边界不够精细，但类别判断相对可靠——它知道某个区域到底是不是水葫芦。"),
  p("约束规则：GLI 候选连通域与 YOLO-WH 检测区域的重叠度必须 > 25%，否则视为绿色水体误检并丢弃。这个双保险机制的效果立竿见影——浅绿色河段、水下藻类等 GLI 植被误检被有效滤除，而真正的水葫芦区域因 YOLO 通常能检出至少一部分而得以保留。"),
  boldP("改进三：形态学碎片合并"),
  p("YOLO 排除非 WH 类掩膜可能将一块完整水葫芦切割为多个碎片——例如水葫芦中间停了一艘小船，YOLO 的 Boat 掩膜就会把水葫芦切出一个洞。形态学闭运算（25×25 椭圆核，迭代 2 次）将相邻碎片重新连接，恢复水葫芦区域的完整性。"),
  p("V4 训练数据扩展至 2000+ 张，覆盖多种河道类型和光照条件，mAP50 达到稳定收敛。至此，人工标注占比已降至总数据的 8%——仅需对 GLI 标注结果进行少量抽检纠偏即可维持高质量。"),
  h3("5.3 工程优化"),
  p("在自迭代训练过程中，同步解决了若干工程问题："),
  bullet("零拷贝架构：图片不再复制到训练目录，标签直接写入图片所在文件夹，解决磁盘溢出问题", "b1"),
  bullet("自动 train/val 划分：按 8:2 比例自动划分训练集与验证集", "b1"),
  bullet("一键训练脚本：将 GLI 伪标签生成 + 数据集划分 + YOLO 训练整合为单条命令", "b1"),
  bullet("Python 版本兼容：Path.walk() 替换为 os.walk()，向下兼容 Python < 3.12", "b1"),
  h3("5.4 版本迭代总览"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [800, 2600, 2600, 3026],
    rows: [
      row([cell("版本", { header: true, width: 800 }), cell("训练数据", { header: true, width: 2600 }), cell("标注方式", { header: true, width: 2600 }), cell("关键改进", { header: true, width: 3026 })]),
      row([cell("V1", { width: 800 }), cell("COCO 预训练（零 WH 数据）", { width: 2600 }), cell("COCO 公开数据集", { width: 2600 }), cell("迁移学习基线，mAP50 = 0.472", { width: 3026 })]),
      row([cell("V2", { width: 800 }), cell("191 张", { width: 2600 }), cell("Label Studio 人工 polygon", { width: 2600 }), cell("冷启动成功，mAP50 = 0.773（+64%）", { width: 3026 })]),
      row([cell("V3", { width: 800 }), cell("约 1,000 张（+800 伪标签）", { width: 2600 }), cell("GLI 固定阈值 + V2 排除干扰", { width: 2600 }), cell("自动标注验证可行，泛化能力增强", { width: 3026 })]),
      row([cell("V4", { width: 800 }), cell("2,000+ 张", { width: 2600 }), cell("GLI Otsu + YOLO-WH 互约束", { width: 2600 }), cell("自适应阈值 + 交集约束 + 碎片合并，高精度收敛", { width: 3026 })]),
    ],
  }),
  spacer(200),
  highlight("四个版本逐级递进，人工标注占比从 100% 降至 8%，模型精度持续上升。验证了「以少量精标注冷启动 → 以自动标注扩量 → 模型进化反哺标签质量」这条路径的可行性。"),
];

// 6. PHASE 5 - 7.15-8.10
const phase5 = [
  pageBreak(),
  h1("六、第五阶段：生产部署与工程化"),
  h2("时间：2026年7月15日 — 8月10日"),
  h3("6.1 ONNX 模型导出与 CPU 推理适配"),
  p("模型训练依赖 GPU（NVIDIA CUDA），但业主生产环境为 CPU 服务器，无法安装 PyTorch、CUDA 等重量级依赖。为此，将 YOLO 模型从 PyTorch 格式导出为 ONNX（Open Neural Network Exchange）开放格式："),
  bullet("导出参数：imgsz=640, opset=12，文件体积 104 MB", "b1"),
  bullet("推理引擎仅需 onnxruntime（CPU 版）+ opencv-python + numpy，三个轻量 pip 包，安装体积约 200 MB", "b1"),
  bullet("无需 CUDA Toolkit、无需 cuDNN、无需 NVIDIA 驱动，支持 Windows / Linux / ARM 跨平台", "b1"),
  h3("6.2 推理性能测试"),
  p("在 i7-13700KF（16 核，3.4GHz）CPU 上完成了完整的性能基准测试："),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [3400, 2000, 1800, 1826],
    rows: [
      row([cell("环节", { header: true, width: 3400 }), cell("平均耗时", { header: true, width: 2000 }), cell("P95 耗时", { header: true, width: 1800 }), cell("占比", { header: true, width: 1826 })]),
      row([cell("YOLO ONNX 推理（含预处理 + NMS 后处理）", { width: 3400 }), cell("138 ms", { width: 2000 }), cell("155 ms", { width: 1800 }), cell("21%", { width: 1826 })]),
      row([cell("GLI 植被掩膜生成（计算 + Otsu + 开运算）", { width: 3400 }), cell("约 120 ms", { width: 2000 }), cell("—", { width: 1800 }), cell("18%", { width: 1826 })]),
      row([cell("形态学处理与轮廓提取（闭运算 + findContours）", { width: 3400 }), cell("约 280 ms", { width: 2000 }), cell("—", { width: 1800 }), cell("43%", { width: 1826 })]),
      row([cell("YOLO-WH 交集约束过滤", { width: 3400 }), cell("约 110 ms", { width: 2000 }), cell("—", { width: 1800 }), cell("17%", { width: 1826 })]),
      row([cell("单图总耗时", { width: 3400 }), cell("约 650 ms", { width: 2000 }), cell("约 900 ms", { width: 1800 }), cell("吞吐约 1.5 fps", { width: 1826 })]),
    ],
  }),
  spacer(100),
  p("瓶颈在形态学运算（占 43%），在更高性能服务器 CPU 上可通过多核并行进一步缩短。模型加载仅需 0.18 秒，运行时内存不超过 500 MB。对于非实时批量处理场景，当前性能已完全满足需求。"),
  h3("6.3 生产推理脚本"),
  p("构建了完整的 Python 推理脚本（scripts/pipelines/inference.py），支持两种模式："),
  bullet("SAM 模式（YOLO + SAM）：YOLO 定位 → SAM Box Prompt 精修 → 像素级 mask 输出，适合对分割精度要求高的场景", "b1"),
  bullet("GLI 模式（YOLO + GLI）：GLI Otsu 植被掩膜 → YOLO 排非 WH → 闭运算 → YOLO-WH 交集约束，轻量级，无需 SAM 依赖，生产环境推荐", "b1"),
  p("同时完成了 Docker 容器化部署方案，编写了完整的生产部署文档（GLI_Pipeline.md），涵盖环境配置、模型加载、API 调用示例与错误排查指南。"),
  h3("6.4 阶段产出"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [3000, 3000, 3026],
    rows: [
      row([cell("事项", { header: true }), cell("内容", { header: true }), cell("状态", { header: true })]),
      row([cell("ONNX 模型"), cell("best.onnx, 104 MB, opset=12, CPU 可运行"), cell("已导出")]),
      row([cell("推理性能"), cell("约 650ms/图（i7-13700KF），吞吐约 1.5 fps"), cell("已验证")]),
      row([cell("生产脚本"), cell("inference.py，支持 SAM 和 GLI 双模式"), cell("已交付")]),
      row([cell("部署文档"), cell("GLI_Pipeline.md，含完整代码与部署指南"), cell("已交付")]),
      row([cell("Docker 方案"), cell("Dockerfile + docker-compose，一键部署"), cell("已交付")]),
    ],
  }),
];

// 7. CURRENT WORK
const current = [
  pageBreak(),
  h1("七、当前工作（截至 2026 年 8 月）"),
  p("核心算法研发与生产部署已基本完成。当前阶段的工作重心从技术攻坚转向质量打磨与产品化，主要包括以下三个方面："),
  h2("7.1 场景覆盖扩充"),
  p("当前训练数据主要来源于单一河道的多次航拍，场景多样性有限——不同季节的植被状态、不同河道的水体颜色、不同天气的光照条件，都可能导致模型在新场景下性能下降。正在收集新的无人机航拍数据，目标是将训练集覆盖范围扩展至："),
  bullet("不同河道类型（宽河道、窄河道、城市内河、乡村河流）", "b1"),
  bullet("不同季节（夏季茂盛期、秋季枯萎期）", "b1"),
  bullet("不同光照条件（正午强光、阴天散射光、傍晚低照度）", "b1"),
  h2("7.2 管线参数场景化调优"),
  p("GLI 管线的参数（Otsu 策略、闭运算核大小、wh_overlap 阈值）是在现有数据集上通过 50+ 轮实验确定的全局最优值，但不一定是每个具体场景的局部最优。正在根据业主反馈，针对高浊度水体、大面积阴影区等特殊场景进行参数微调，目标是在保持整体精度的前提下，降低特定场景的误检率和漏检率。"),
  h2("7.3 交付文档与材料完善"),
  bullet("整理模型研发全流程技术文档，形成标准化交付物", "b1"),
  bullet("编写客户端操作手册，覆盖软件安装、模型加载、单图/批量推理、结果解读等环节", "b1"),
  bullet("配合项目验收，准备汇报材料与演示案例", "b1"),
];

// 8. FUTURE PLANS
const future = [
  h1("八、后续计划"),
  p("后续工作围绕三个方向展开：减依赖（降低部署复杂度）、提效率（加速推理）、建平台（产品化）。三条线可并行推进。"),
  h2("8.1 减依赖 — SAM 模型蒸馏"),
  p("当前 SAM ViT-H 模型权重文件 2.4GB，推理时需要加载到内存，且与 YOLO 分属两个独立模型，增加了部署复杂度和维护成本。计划通过知识蒸馏（Knowledge Distillation）技术，将 SAM 的精细分割能力迁移到 YOLO 自身的分割头上，实现单模型端到端推理。"),
  bullet("预期效果：去除 SAM 依赖后，部署体积从约 2.5GB 缩减至约 200MB，推理时间从约 650ms 降至约 150ms", "b1"),
  bullet("当前进展：已完成技术调研与实验方案设计，论文初稿已撰写，待训练数据进一步扩充后启动蒸馏训练", "b1"),
  h2("8.2 提效率 — 边缘端部署与实时推理"),
  p("当前推理在桌面级 CPU 上约 1.5 fps，适合离线批量处理但不满足实时视频流需求。计划从两个方向提升推理效率："),
  bullet("边缘端适配：将 ONNX 模型部署到 Jetson Orin、RK3588 等 ARM 边缘计算设备，实现无人机端侧实时推理，摆脱对地面站或云端的算力依赖", "b1"),
  bullet("管线加速：将 GLI 形态学运算改为 GPU 加速或近似算法，目标将单图耗时降至 200ms 以内，吞吐达到 5+ fps，支持无人机直播视频流逐帧检测", "b1"),
  h2("8.3 建平台 — 水葫芦监测管理系统"),
  p("当前模型以命令行脚本形式运行，结果以图片和文本方式输出。为满足河道管理部门的日常使用需求，计划搭建一套 Web 端监测管理平台："),
  bullet("数据管理：支持历史检测结果的存储、查询与可视化回放", "b1"),
  bullet("多期对比：加载同一河道不同时期的检测结果，自动计算水葫芦覆盖面积的时间变化趋势，生成对比报告", "b1"),
  bullet("自动预警：水葫芦覆盖面积超过预设阈值时自动推送预警通知", "b1"),
  spacer(300),
  h2("8.4 后续工作优先级"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [800, 3200, 2200, 1600, 1226],
    rows: [
      row([cell("优先级", { header: true, width: 800 }), cell("工作项", { header: true, width: 3200 }), cell("预期收益", { header: true, width: 2200 }), cell("预估周期", { header: true, width: 1600 }), cell("依赖条件", { header: true, width: 1226 })]),
      row([cell("P0", { width: 800 }), cell("场景覆盖扩充 + 管线参数调优", { width: 3200 }), cell("提升新场景下的模型鲁棒性", { width: 2200 }), cell("持续进行", { width: 1600 }), cell("新批次航拍数据", { width: 1226 })]),
      row([cell("P1", { width: 800 }), cell("SAM 模型蒸馏", { width: 3200 }), cell("部署体积 -92%，推理加速 -77%", { width: 2200 }), cell("3~4 周", { width: 1600 }), cell("训练数据 > 3000 张", { width: 1226 })]),
      row([cell("P2", { width: 800 }), cell("边缘端部署适配", { width: 3200 }), cell("摆脱地面站依赖，实时巡检", { width: 2200 }), cell("4~6 周", { width: 1600 }), cell("硬件设备到位", { width: 1226 })]),
      row([cell("P3", { width: 800 }), cell("Web 监测管理平台", { width: 3200 }), cell("产品化交付，降低使用门槛", { width: 2200 }), cell("6~8 周", { width: 1600 }), cell("前后端开发资源", { width: 1226 })]),
    ],
  }),
];

// APPENDIX
const appendix = [
  pageBreak(),
  h1("附录：GLI 管线关键技术参数"),
  p("以下参数是 GLI 自动标注与生产推理管线的核心配置，经过 50 余轮对比实验验证。在更换应用场景或数据来源时，建议重新评估并微调。"),
  new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: [2600, 1600, 4826],
    rows: [
      row([cell("参数", { header: true, width: 2600 }), cell("默认值", { header: true, width: 1600 }), cell("说明", { header: true, width: 4826 })]),
      row([cell("GLI 二值化", { width: 2600 }), cell("Otsu 自适应", { width: 1600 }), cell("逐图自动寻找最优分割阈值，替代固定 GLI 阈值", { width: 4826 })]),
      row([cell("YOLO 置信度阈值", { width: 2600 }), cell("0.10", { width: 1600 }), cell("低阈值优先保证召回，误检由 GLI 互约束统一过滤", { width: 4826 })]),
      row([cell("GLI-YOLO 重叠阈值", { width: 2600 }), cell("0.25", { width: 1600 }), cell("候选域与 YOLO-WH 重叠度下限。提高则更精确，降低则更高召回", { width: 4826 })]),
      row([cell("形态学闭运算", { width: 2600 }), cell("25x25, 2 次迭代", { width: 1600 }), cell("椭圆核，合并被 YOLO 排除掩膜切开的相邻 WH 碎片", { width: 4826 })]),
      row([cell("形态学开运算", { width: 2600 }), cell("5x5 椭圆核", { width: 1600 }), cell("二值化后和闭运算前各一次，去噪点", { width: 4826 })]),
      row([cell("最小轮廓面积", { width: 2600 }), cell("50 px", { width: 1600 }), cell("低于此值的连通域丢弃，过滤微型碎片", { width: 4826 })]),
      row([cell("排除类别", { width: 2600 }), cell("Boat/Bridge/Structure/Tree", { width: 1600 }), cell("全部非 WH 类从植被掩膜中剔除", { width: 4826 })]),
      row([cell("YOLO 模型", { width: 2600 }), cell("YOLOv8m-seg", { width: 1600 }), cell("2,722 万参数，CSPDarknet 主干，FPN+PAN 颈部", { width: 4826 })]),
      row([cell("SAM 模型", { width: 2600 }), cell("ViT-H（可选）", { width: 1600 }), cell("6.32 亿参数，零样本通用分割，用于精修", { width: 4826 })]),
      row([cell("输入分辨率", { width: 2600 }), cell("640x640", { width: 1600 }), cell("Letterbox 等比例填充，自适应原始宽高比", { width: 4826 })]),
      row([cell("检测类别", { width: 2600 }), cell("5 类", { width: 1600 }), cell("WH(3) / Boat(0) / Bridge(1) / Structure(2) / Tree(4)", { width: 4826 })]),
      row([cell("导出格式", { width: 2600 }), cell("ONNX, opset=12", { width: 1600 }), cell("104 MB，onnxruntime CPU 推理，无 PyTorch 依赖", { width: 4826 })]),
    ],
  }),
];

// BUILD
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 34, bold: true, font: "Microsoft YaHei", color: C.primary }, paragraph: { spacing: { before: 400, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Microsoft YaHei", color: C.secondary }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: "Microsoft YaHei", color: C.dark }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => ({
      reference: `b${i}`,
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    })),
  },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: cover,
    },
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 4 } }, children: [new TextRun({ text: "水葫芦智能识别模型 · 研发进展报告", font: "Microsoft YaHei", size: 18, color: C.gray })], })], }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "- ", font: "Microsoft YaHei", size: 18, color: C.gray }), new TextRun({ children: [PageNumber.CURRENT], font: "Microsoft YaHei", size: 18, color: C.gray }), new TextRun({ text: " -", font: "Microsoft YaHei", size: 18, color: C.gray })], })], }) },
      children: [
        ...overview, pageBreak(),
        ...phase1,
        ...phase2,
        ...phase3,
        ...phase4,
        ...phase5,
        ...current,
        ...future,
        ...appendix,
      ],
    },
  ],
});

const outPath = "D:/chengs/9.project/shuihulu/doc/项目介绍_水葫芦智能检测系统.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  console.log("Saved: " + outPath);
  console.log("Size: " + (buf.length / 1024).toFixed(1) + " KB");
});
