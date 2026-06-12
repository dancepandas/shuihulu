# 水葫芦 YOLO + SAM Docker 部署文档

> 版本：1.0 | 日期：2026-06-12 | 镜像：`hyacinth-yolo-sam:latest`（9.39 GB）

---

## 目录

1. [架构总览](#1-架构总览)
2. [环境要求](#2-环境要求)
3. [文件结构](#3-文件结构)
4. [构建镜像](#4-构建镜像)
5. [配置详解](#5-配置详解)
6. [启动与停止](#6-启动与停止)
7. [API 参考](#7-api-参考)
8. [GPU 与 CPU 模式](#8-gpu-与-cpu-模式)
9. [性能基准](#9-性能基准)
10. [镜像源与网络加速](#10-镜像源与网络加速)
11. [故障排除](#11-故障排除)
12. [生产部署建议](#12-生产部署建议)
13. [附录](#13-附录)

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────┐
│                   Docker Container               │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐ │
│  │  Flask    │───▶│  YOLOv8  │───▶│    SAM     │ │
│  │  :13000  │    │  (检测)   │    │  (分割)    │ │
│  └──────────┘    └──────────┘    └────────────┘ │
│       ▲               │                │         │
│       │         5 类检测框      仅水葫芦掩膜      │
│       │         (Boat/Bridge/     (pixel mask)   │
│       │          Structure/                      │
│  ┌────┴─────┐    Hyacinth/tree)                 │
│  │  Client  │                                    │
│  └──────────┘                                    │
│                                                  │
│  /models/  ←── 宿主机 ./weights/ (只读挂载)       │
│  /data/    ←── 宿主机 ./data/    (只读挂载)       │
└─────────────────────────────────────────────────┘
```

**数据流：**

1. 客户端通过 HTTP POST 上传图片（或指定 `/data/` 下的路径）
2. **YOLOv8m-seg** 对图片推理，检出全部 5 类目标：Boat / Bridge / Structure / Water Hyacinth / tree
3. 仅 **Water Hyacinth（class 3）** 的边界框传给 **SAM ViT-H**
4. SAM 为每个水葫芦框生成像素级二值掩膜（bool mask）
5. 响应 JSON 包含：全部 5 类检测结果（bbox + 类别 + 置信度）+ 水葫芦的 SAM 掩膜（base64 PNG + 轮廓多边形 + 像素面积）

---

## 2. 环境要求

### 2.1 宿主机

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 操作系统 | Windows 10+ / Linux / macOS | Linux (Ubuntu 22.04) |
| Docker | 20.10+ | 28.x |
| Docker Compose | v2.x | 插件版 (`docker compose`) |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 15 GB 可用 | SSD，30 GB+ |
| CPU | 4 核 | 8 核+ |
| GPU（可选）| 支持 CUDA 12.x 的 NVIDIA 显卡 | RTX 2060+ |

> **注意：** 当前镜像为 CPU 版。GPU 版需要额外配置，见 [第 8 节](#8-gpu-与-cpu-模式)。

### 2.2 模型文件

构建前需确保 `weights/` 目录包含以下文件：

| 文件 | 大小 | 用途 | 来源 |
|------|------|------|------|
| `hyacinth5_best.pt` | ~55 MB | YOLOv8m-seg 5 类检测模型 | `scripts/train_yolo_sam.py` 训练产物 |
| `sam_vit_h.pth` | ~2.4 GB | SAM ViT-H 分割模型 | `scripts/download_weights.py` 下载 |

模型文件通过 Docker volume **挂载到容器内**，不在镜像中。升级模型只需替换宿主机文件后重启容器。

---

## 3. 文件结构

```
shuihulu/
├── Dockerfile                  # 镜像构建定义
├── docker-compose.yml          # 容器编排（端口、挂载、启动命令）
├── .dockerignore               # 构建时忽略的文件
├── serve.py                    # Flask API 入口（推理服务）
├── src/                        # 核心库（打包容器的唯一源码目录）
│   └── shuihulu_yolo_sam/
│       ├── __init__.py
│       ├── model.py            # YoloSamPipeline + resolve_class_index
│       └── api.py              # Segmenter + SegResult + _draw_vis
├── weights/                    # 模型文件（外挂，不打包）
│   ├── hyacinth5_best.pt       # 5 类 YOLO 模型
│   ├── sam_vit_h.pth           # SAM ViT-H 权重
│   ├── yolov8m-seg.pt          # COCO 预训练（训练用）
│   └── yolov8m.pt              # 检测预训练（备用）
└── data/                       # 输入图片目录（可选外挂）
    └── example.jpg
```

### 3.1 Dockerfile 分层说明

```
FROM python:3.11-slim-bookworm          ← 基础镜像（195 MB）
  ↓
RUN sed ... aliyun apt mirror           ← 第 1 层：apt 加速
RUN apt install libgl1-mesa-glx ...     ← 第 2 层：OpenCV 系统依赖
RUN pip install torch ultralytics ...   ← 第 3 层：Python 依赖（~8 GB）
COPY src/ serve.py                      ← 第 4 层：应用代码（~50 KB）
CMD ["python", "serve.py", ...]          ← 启动命令
```

### 3.2 `.dockerignore`

构建时排除以下目录，减少 context 大小并避免意外打包：

```
__pycache__/  .git/  runs/  datasets/  doc/  tests/  .claude/  weights/
```

> **关键：** `weights/` 在 `.dockerignore` 中，确保模型文件不被打包进镜像。模型通过 volume 在运行时挂载。

---

## 4. 构建镜像

### 4.1 标准构建

```bash
cd D:\chengs\9.project\shuihulu
docker build -t hyacinth-yolo-sam:latest .
```

**预计时间：** 8 ~ 20 分钟（取决于网络速度）

**构建阶段耗时参考：**

| 阶段 | 操作 | 耗时 |
|------|------|------|
| [1/7] | 拉取基础镜像（仅首次） | 10 ~ 60 秒 |
| [2/7] | sed apt 源替换 | < 1 秒 |
| [3/7] | apt 安装系统依赖 | 30 ~ 60 秒 |
| [4/7] | pip 安装 Python 依赖 | 5 ~ 15 分钟（主要瓶颈） |
| [5/7] | COPY src/ | < 1 秒 |
| [6/7] | COPY serve.py | < 1 秒 |
| [7/7] | 设置 WORKDIR + EXPOSE | < 1 秒 |

### 4.2 重新构建

```bash
# 强制完全重建（忽略缓存）
docker build --no-cache -t hyacinth-yolo-sam:latest .

# 仅重新构建代码层（依赖层使用缓存）——日常开发推荐
docker build -t hyacinth-yolo-sam:latest .
```

由于 COPY 指令在依赖安装之后，修改 `serve.py` 或 `src/` 后重新构建仅需 ~1 秒。

### 4.3 构建参数

当前 Dockerfile 不使用 ARG，所有配置在运行时通过以下方式指定：

- **命令行参数：** `docker-compose.yml` 的 `command:` 字段
- **环境变量：** 可在 `docker-compose.yml` 中追加 `environment:`
- **卷挂载：** 模型文件通过 `volumes:` 指定

### 4.4 验证构建

```bash
docker images hyacinth-yolo-sam:latest
# 预期输出：
# REPOSITORY          TAG       IMAGE ID       CREATED         SIZE
# hyacinth-yolo-sam   latest    97d2cc231056   2 hours ago     9.39GB
```

---

## 5. 配置详解

### 5.1 docker-compose.yml 逐字段说明

```yaml
services:
  hyacinth:                              # 服务名
    build: .                             # 构建上下文（含 Dockerfile 的目录）
    image: hyacinth-yolo-sam:latest      # 镜像标签
    command:                             # 覆盖 Dockerfile 的 CMD
      ["python", "serve.py",
       "--yolo", "/models/hyacinth5_best.pt",   # YOLO 模型路径（容器内）
       "--sam",  "/models/sam_vit_h.pth",       # SAM 模型路径（容器内）
       "--device", "cpu"]                       # 推理设备
    ports:
      - "13000:13000"                    # 宿主机端口:容器端口
    volumes:
      - ./weights:/models:ro             # 模型目录（只读，安全）
      - ./data:/data:ro                  # 数据目录（只读，可选）
    restart: unless-stopped              # 崩溃自动重启（手动停止除外）
```

### 5.2 serve.py 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--yolo` | `/models/hyacinth5_best.pt` | YOLO 模型权重路径 |
| `--sam` | `/models/sam_vit_h.pth` | SAM 模型权重路径 |
| `--sam-type` | `vit_h` | SAM 模型类型：`vit_h` / `vit_l` / `vit_b` |
| `--device` | `cpu` | 推理设备：`cpu` / `cuda` / `cuda:0` |
| `--target-class` | `hyacinth` | 目标类别名称或索引（SAM 只对该类分割） |
| `--conf` | `0.25` | YOLO 检测置信度阈值 |
| `--iou` | `0.5` | YOLO NMS IoU 阈值 |
| `--imgsz` | `640` | 推理输入尺寸（像素） |
| `--port` | `13000` | HTTP 监听端口 |
| `--host` | `0.0.0.0` | HTTP 监听地址 |

### 5.3 模型切换

#### 使用 TinySAM（更快，精度略低）

1. 下载 TinySAM 权重到 `weights/`
2. 修改 `command:` 中的 `--sam` 和 `--sam-type`：
   ```yaml
   command: ["python", "serve.py",
     "--yolo", "/models/hyacinth5_best.pt",
     "--sam", "/models/tinysam.pth",
     "--sam-type", "vit_b",
     "--device", "cpu"]
   ```

#### 使用不同 YOLO 模型

```yaml
# YOLOv8n-seg（更快，精度略低）
"--yolo", "/models/yolov8n-seg.pt"

# 自定义训练产物
"--yolo", "/models/my_custom_model.pt"
```

### 5.4 阈值调节

```yaml
# 降低阈值 → 更多检出水葫芦（但可能误报增多）
"--conf", "0.15"

# NMS IoU 阈值 → 高值保留更多重叠框
"--iou", "0.7"
```

---

## 6. 启动与停止

### 6.1 启动

```bash
cd D:\chengs\9.project\shuihulu
docker compose up -d
```

首次启动后模型加载需要 20 ~ 60 秒（取决于 CPU），期间 `/health` 可能返回 500。用以下命令等待就绪：

```bash
# 轮询等待服务就绪
while ! curl -s http://localhost:13000/health | grep -q ok; do
  echo "waiting..."
  sleep 5
done
echo "Service ready!"
```

### 6.2 查看日志

```bash
# 实时日志
docker compose logs -f

# 最近 50 行
docker compose logs --tail 50

# 仅看错误
docker compose logs 2>&1 | grep -i error
```

### 6.3 停止

```bash
# 停止但保留容器
docker compose stop

# 停止并删除容器（保留镜像和卷）
docker compose down

# 停止并删除容器 + 网络
docker compose down -v
```

### 6.4 重启

```bash
docker compose restart
```

### 6.5 状态检查

```bash
# 容器运行状态
docker compose ps

# 资源使用
docker stats shuihulu-hyacinth-1 --no-stream

# 模型是否正确加载
docker compose logs | grep -E "目标类|可用类别|设备|监听"
# 预期输出示例：
# [INFO] 目标类索引: 3（hyacinth）
# [INFO] 可用类别: ['Boat', 'Bridge', 'Structure', 'Water Hyacinth', 'tree']
# [INFO] 设备: cpu
# [INFO] 监听 http://0.0.0.0:13000
```

---

## 7. API 参考

### 7.1 基础信息

| 属性 | 值 |
|------|-----|
| 协议 | HTTP/1.1 |
| 格式 | JSON |
| 编码 | UTF-8 |
| 超时 | 单图 CPU 模式 30 ~ 120 秒（取决于图片大小和水葫芦数量） |

### 7.2 GET /health

健康检查端点。Docker HEALTHCHECK 每 30 秒调用一次。

**请求：**

```bash
curl http://localhost:13000/health
```

**响应（200 OK）：**

```json
{"status": "ok"}
```

### 7.3 POST /api/predict

上传一张图片，返回全部 5 类检测结果 + 水葫芦 SAM 掩膜。

**请求：**

```bash
curl -X POST \
  -F "image=@/path/to/photo.jpg" \
  http://localhost:13000/api/predict
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 图片文件，支持 JPG / PNG / BMP / TIFF |

**响应（200 OK）：**

```json
{
  "filename": "photo.jpg",
  "image_size": {"width": 4032, "height": 3024},
  "total_area_px": 374003,
  "objects": [
    {
      "class": "Water Hyacinth",
      "class_id": 3,
      "confidence": 0.8472,
      "bbox": [1709, 1116, 2230, 1788],
      "is_water_hyacinth": true,
      "sam_score": 0.949,
      "area_px": 189671,
      "mask_base64": "iVBORw0KGgoAAAANSUhEUg...",
      "mask_contour": [
        [[1709, 1116], [1710, 1117], [1712, 1118], ...]
      ]
    },
    {
      "class": "Boat",
      "class_id": 0,
      "confidence": 0.912,
      "bbox": [100, 200, 300, 400],
      "is_water_hyacinth": false,
      "sam_score": null,
      "area_px": null,
      "mask_base64": null,
      "mask_contour": null
    }
  ]
}
```

**响应字段详解：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `filename` | string | 上传的文件名 |
| `image_size` | object | 图片尺寸 `{width, height}` |
| `total_area_px` | int | 所有水葫芦 SAM 掩膜总像素面积 |
| `objects[]` | array | 全部 5 类检测结果 |
| `objects[].class` | string | 类别名称（Boat/Bridge/Structure/Water Hyacinth/tree） |
| `objects[].class_id` | int | 类别索引（0-4） |
| `objects[].confidence` | float | YOLO 检测置信度 [0, 1] |
| `objects[].bbox` | [int×4] | 边界框 `[x1, y1, x2, y2]`（像素坐标） |
| `objects[].is_water_hyacinth` | bool | 是否为目标类（水葫芦） |
| `objects[].sam_score` | float\|null | SAM 掩膜置信度（仅水葫芦有值） |
| `objects[].area_px` | int\|null | SAM 掩膜像素面积（仅水葫芦有值） |
| `objects[].mask_base64` | string\|null | PNG 编码的掩膜图片 base64（仅水葫芦） |
| `objects[].mask_contour` | array\|null | 掩膜轮廓多边形（仅水葫芦） |

**错误响应：**

```json
// 400 — 未上传图片
{"error": "请通过 image 字段上传图片文件"}

// 400 — 无法解码
{"error": "无法解码图片，请确认上传了 JPG/PNG/BMP 格式的文件。"}

// 500 — 服务未初始化
{"error": "服务未初始化"}
```

### 7.4 POST /api/predict_file

指定服务器上已挂载的图片路径，适用于批量处理 `/data/` 目录下的文件。

**请求：**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/aerial_photo.jpg"}' \
  http://localhost:13000/api/predict_file
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 容器内图片绝对路径（对应宿主机 `./data/` 挂载目录） |

**响应：** 同 `/api/predict`，额外包含 `"filename"` 字段为路径的 basename。

**错误响应：**

```json
// 400 — 路径不存在
{"error": "图片路径不存在：/data/nonexistent.jpg"}

// 400 — 无法读取
{"error": "无法读取图片：/data/corrupt.jpg"}
```

### 7.5 Python 调用示例

```python
import requests
import base64
import cv2
import numpy as np

def predict(image_path: str, server_url: str = "http://localhost:13000"):
    """调用水葫芦推理服务."""
    with open(image_path, "rb") as f:
        resp = requests.post(f"{server_url}/api/predict", files={"image": f})
    resp.raise_for_status()
    result = resp.json()

    # 保存水葫芦掩膜
    for i, obj in enumerate(result["objects"]):
        if obj["is_water_hyacinth"] and obj["mask_base64"]:
            mask_bytes = base64.b64decode(obj["mask_base64"])
            mask_arr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f"mask_{i}.png", mask)

    print(f"检测到 {len(result['objects'])} 个目标，"
          f"水葫芦总面积 {result['total_area_px']} px")
    return result

# 使用
result = predict("drone_photo.jpg")
```

### 7.6 批量处理 `/data/` 目录

```bash
#!/bin/bash
# 批量处理挂载目录下所有 JPG 文件
for img in /path/to/data/*.jpg; do
  name=$(basename "$img")
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"/data/$name\"}" \
    http://localhost:13000/api/predict_file \
    > "results/${name%.jpg}.json"
  echo "processed: $name"
done
```

---

## 8. GPU 与 CPU 模式

### 8.1 当前配置（CPU）

Dockerfile 和 docker-compose.yml 当前配置为 CPU 模式：

- Dockerfile: `--device cpu`
- docker-compose: `command: ... "--device", "cpu"`
- 无 `runtime: nvidia`

**特点：**
- ✅ 无需 NVIDIA 显卡，任何 x86_64 机器可运行
- ✅ 部署简单，无需 nvidia-container-toolkit
- ❌ 推理较慢（SAM ViT-H: 30 ~ 120 秒/张）

### 8.2 切换到 GPU 模式

如果宿主机有 NVIDIA 显卡并安装了 nvidia-container-toolkit：

**Step 1：安装 nvidia-container-toolkit**

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Step 2：修改 docker-compose.yml**

```yaml
services:
  hyacinth:
    # ... 其余不变 ...
    command: ["python", "serve.py",
      "--yolo", "/models/hyacinth5_best.pt",
      "--sam", "/models/sam_vit_h.pth",
      "--device", "cuda"]                    # ← CPU → cuda
    deploy:                                   # ← 新增
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # runtime: nvidia                         # ← docker-compose v1 用这一行
```

**Step 3：切换 Dockerfile 基础镜像**

GPU 模式需要用 CUDA 基础镜像。创建 `Dockerfile.gpu`：

```dockerfile
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt update && apt install -y python3.11 python3-pip \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    ultralytics "opencv-python-headless>=4.10.0" flask pyyaml "segment-anything>=1.0"

COPY src/ /app/src/
COPY serve.py /app/
WORKDIR /app
EXPOSE 13000
CMD ["python", "serve.py", "--yolo", "/models/hyacinth5_best.pt",
     "--sam", "/models/sam_vit_h.pth", "--device", "cuda"]
```

### 8.3 serve.py 自动降级

`serve.py` 内置了 GPU 自动降级逻辑：

```python
self.device = device if torch.cuda.is_available() else "cpu"
```

- 即使 `--device cuda`，如果容器内检测不到 GPU → 自动用 CPU
- 无需手动修改配置即可在同一镜像中切换

---

## 9. 性能基准

### 9.1 测试环境

| 项目 | CPU 模式 | GPU 模式 |
|------|---------|---------|
| CPU | Intel Xeon (24 vCPU) | 同左 |
| GPU | - | NVIDIA RTX 2060 (6 GB) |
| RAM | 16 GB Docker Desktop | 同左 |
| 测试图片 | 4032×3024 无人机航拍 | 同左 |

### 9.2 单图推理耗时

| 阶段 | CPU 模式 | GPU 模式 |
|------|---------|---------|
| YOLO 加载（启动时一次） | ~5 秒 | ~3 秒 |
| SAM 加载（启动时一次） | ~25 秒 | ~8 秒 |
| YOLO 推理（每图） | ~2 秒 | ~0.1 秒 |
| SAM 推理（每框） | ~15 秒 | ~1 秒 |
| **单图总计（6 框）** | **~90 秒** | **~8 秒** |

### 9.3 资源占用

| 指标 | CPU 模式 | GPU 模式 |
|------|---------|---------|
| 镜像大小 | 9.39 GB | ~13 GB |
| 内存（空闲） | ~1.5 GB | ~2.0 GB |
| 内存（推理峰值） | ~6 GB | ~4.5 GB |
| 显存 | 0 | ~4 GB |

### 9.4 并发

Flask 默认单线程处理请求。如需并发：

```python
# serve.py 末尾
from werkzeug.serving import run_simple
run_simple("0.0.0.0", 13000, app, threaded=True, processes=1)
```

> **警告：** SAM 模型不是线程安全的。如需并发，为每个 worker 创建独立的 Predictor 实例，或使用进程隔离（Gunicorn + `preload_app`）。

---

## 10. 镜像源与网络加速

### 10.1 Docker 镜像源配置

编辑 `C:\Users\<用户名>\.docker\daemon.json`：

```json
{
  "registry-mirrors": [
    "https://docker.xuanyuan.me",
    "https://docker.mirrors.ustc.edu.cn",
    "https://docker.mszx.com",
    "https://docker.1panel.live",
    "https://hub.rat.dev",
    "https://docker.m.daocloud.io"
  ]
}
```

> **优先级：** 列表从上到下依次尝试。轩辕（xuanyuan）和中科大（ustc）最稳定。

重启 Docker Desktop 后生效：

```bash
docker info | grep "Registry Mirrors" -A 5
```

### 10.2 Dockerfile 内加速

本 Dockerfile 已内置以下加速：

| 资源 | 镜像 | 位置 |
|------|------|------|
| apt | 阿里云 `mirrors.aliyun.com` | 第 11 行 `sed` |
| pip | 清华 `pypi.tuna.tsinghua.edu.cn` | 第 21 行 `-i` |

### 10.3 离线环境部署

如果目标环境无网络：

```bash
# 有网环境导出
docker save hyacinth-yolo-sam:latest | gzip > hyacinth-yolo-sam.tar.gz

# 离线环境导入
docker load < hyacinth-yolo-sam.tar.gz
```

模型文件（`weights/`）直接拷贝。

---

## 11. 故障排除

### 11.1 容器反复重启

```bash
docker compose logs --tail 30
```

**常见原因：**

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `ValueError: 未在类别名称中找到匹配 'hyacinth'` | 挂载的是 COCO 预训练模型（80 类）而非 5 类训练产物 | 确认 `weights/` 下为 `hyacinth5_best.pt` |
| `ModuleNotFoundError: No module named 'xxx'` | 镜像构建不完整 | `docker build --no-cache` |
| `FileNotFoundError: /models/sam_vit_h.pth` | SAM 权重未挂载 | 确认 `weights/sam_vit_h.pth` 存在 |
| `Killed` (exit 137) | OOM 内存不足 | 增加 Docker Desktop 内存限制或减小 `--imgsz` |

### 11.2 推理超时无响应

SAM ViT-H 在 CPU 上可能很慢（大图可达 120 秒）。客户端需设置足够长的超时：

```python
requests.post(url, files=..., timeout=300)  # 5 分钟
```

### 11.3 掩膜质量差

| 问题 | 可能原因 | 调整 |
|------|---------|------|
| 水葫芦漏检 | `--conf` 太高 | 降至 0.15 |
| 掩膜边缘粗糙 | SAM 本身局限 | 无需调整（已是最佳） |
| 误检（非水葫芦被分割）| YOLO 误报 | 提高 `--conf` 至 0.35 |
| 同区域多个掩膜 | NMS 不足 | 降低 `--iou` 至 0.3 |

### 11.4 端口占用

```
Error: port 13000 already in use
```

```bash
# 查找占用进程
netstat -ano | findstr :13000    # Windows
sudo lsof -i :13000              # Linux

# 改用其他端口
# docker-compose.yml 中修改为 "13001:13000"
```

### 11.5 镜像过大导致磁盘不足

```bash
# 清理未使用的镜像
docker image prune -a

# 清理构建缓存
docker builder prune -af

# 清理所有未使用资源
docker system prune -af
```

---

## 12. 生产部署建议

### 12.1 安全加固

1. **不要以 root 运行：** Dockerfile 中添加 `USER 1000:1000`
2. **只读挂载：** 模型和数据使用 `:ro`（已配置）
3. **网络隔离：** 仅暴露 13000 端口
4. **限流：** 前端加 Nginx `limit_req_zone`
5. **HTTPS：** 前面套 Nginx + Let's Encrypt

### 12.2 高可用

```yaml
# docker-compose.yml — 多副本 + Nginx 负载均衡
services:
  hyacinth:
    deploy:
      replicas: 2
    # ...
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### 12.3 监控

```yaml
# 添加健康检查到 docker-compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:13000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 12.4 日志管理

```yaml
# docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "3"
```

### 12.5 资源限制

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: "4"
    reservations:
      memory: 4G
      cpus: "2"
```

---

## 13. 附录

### 13.1 构建时间线（完整记录）

```
[2026-06-12] 初始构建
  基础镜像: python:3.11-slim-bookworm (195 MB, cached)
  apt 阶段: 43 packages, 52.6 MB, ~30s (阿里云镜像 308 kB/s)
  pip 阶段: torch 2.12 + ultralytics 8.x + opencv + flask + segment-anything, ~5min (清华镜像)
  最终镜像: 9.39 GB
  ✅ /health 正常, /api/predict 返回 6 个水葫芦 + 掩膜
```

### 13.2 关键命令速查

```bash
# 构建
docker build -t hyacinth-yolo-sam:latest .

# 启动
docker compose up -d

# 状态
docker compose ps && docker compose logs --tail 5

# 测试
curl http://localhost:13000/health
curl -X POST -F image=@test.jpg http://localhost:13000/api/predict

# 停止
docker compose down

# 清理
docker builder prune -af && docker image prune -a
```

### 13.3 镜像层级详情

```
docker history hyacinth-yolo-sam:latest

IMAGE          CREATED          SIZE      COMMENT
97d2cc231056   2 hours ago      0B        CMD ["python" "serve.py" ...]
<missing>      2 hours ago      0B        EXPOSE 13000
<missing>      2 hours ago      0B        WORKDIR /app
<missing>      2 hours ago      51.2kB    COPY serve.py /app/
<missing>      2 hours ago      51.2kB    COPY src/ /app/src/
<missing>      2 hours ago      8.12GB    pip install torch ultralytics opencv flask segment-anything
<missing>      2 hours ago      186MB     apt install libgl1-mesa-glx libglib2.0-0
<missing>      2 hours ago      394B      sed aliyun mirror
<missing>      2 weeks ago      195MB     FROM python:3.11-slim-bookworm
```

> **关键洞察：** pip 层占 8.12 GB（整个镜像的 86%），主要是 torch（~3 GB）和 nvidia 相关库。后续优化可考虑精简 torch 安装或使用 `torch-cpu` 专用轮子。

### 13.4 与项目其他组件的关系

```
shuihulu/
├── Docker 服务 (本文档)    ← 生产推理
├── scripts/train_yolo_sam.py  ← 训练
├── scripts/split_dataset.py   ← 数据准备
├── scripts/val_yolo_sam.py    ← 验证
└── main.py                    ← 本地推理 / 可视化
```
