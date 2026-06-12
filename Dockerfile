# 水葫芦 YOLOv8 + SAM 推理服务（CPU 版）
# 构建: docker build -t hyacinth-yolo-sam .
# 运行: docker-compose up -d

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Debian apt 使用阿里云镜像加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 系统依赖（opencv 需要 libgl）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch + 应用依赖（清华 pip 镜像，torch 含 CUDA 但也兼容 CPU）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    torch torchvision \
    ultralytics \
    "opencv-python-headless>=4.10.0" \
    flask \
    pyyaml \
    "segment-anything>=1.0"

COPY src/ /app/src/
COPY serve.py /app/

WORKDIR /app
EXPOSE 13000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:13000/health')" || exit 1

CMD ["python", "serve.py", \
     "--yolo", "/models/hyacinth5_best.pt", \
     "--sam", "/models/sam_vit_h.pth", \
     "--device", "cpu"]
