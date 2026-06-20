#!/bin/bash
# ============================================================
# 水葫芦 V8 一站式训练脚本
# 用法: bash scripts/run_v8.sh
#
# 步骤:
#   1. 解压 data/ 下所有 zip 文件
#   2. GLI(>0.12) + V7 伪标签流水线
#   3. YOLOv8m-seg 训练 (从 V7 继续微调)
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── 配置 ──────────────────────────────────────
MODEL="runs/hyacinth7_yolo_sam/weights/best.pt"
DATA_CONFIG="configs/dataset_hyacinth8_seg.yaml"
EPOCHS=100
BATCH=8
IMGSZ=640
DEVICE=0
WORKERS=4
PROJECT="runs/segment"
NAME="hyacinth8_yolo_sam"
# ───────────────────────────────────────────────

echo "============================================"
echo "  Water Hyacinth V8 Training Pipeline"
echo "============================================"
echo ""

# ── Step 1: 解压数据 ──────────────────────────
echo "[1/3] Unzipping data..."
ZIP_COUNT=$(find data -maxdepth 1 -name "*.zip" | wc -l)
if [ "$ZIP_COUNT" -gt 0 ]; then
    for f in data/*.zip; do
        echo "  extracting: $(basename "$f")"
        unzip -o "$f" -d data/ > /dev/null
    done
    echo "  done. $(find data -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l) images"
else
    echo "  no zip files found, skipping"
fi
echo ""

# ── Step 2: GLI Pipeline ──────────────────────
echo "[2/3] Running GLI+V7 pseudo-label pipeline..."
python -u scripts/gli_v8_pipeline.py
echo ""

# ── Step 3: 训练 ───────────────────────────────
echo "[3/3] Training YOLOv8m-seg V8..."
python scripts/train_yolo_sam.py \
    --model "$MODEL" \
    --data "$DATA_CONFIG" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$PROJECT" \
    --name "$NAME"
echo ""

echo "============================================"
echo "  V8 Training Complete!"
echo "  Model: $PROJECT/$NAME/weights/best.pt"
echo "============================================"
