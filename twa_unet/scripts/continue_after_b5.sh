#!/bin/bash
# 等 B5+B6 完成后启动 B3 (CBAM) + B4 (Triplet) 训练
# 当前 master_pipeline 已经会跑 B5 → B6

cd D:/chengs/9.project/shuihulu
export PYTHONPATH=twa_unet/scripts

echo "=== [CONTINUE $(date +%Y-%m-%d_%H:%M:%S)] 等待 B6 完成 ==="
# 等 segformer 进程结束 (B6)
while pgrep -f "segformer_finetune_v11_fullscreen" > /dev/null; do
  sleep 60
done

echo "=== B3 重跑 (CBAM 修复后) ==="
python twa_unet/scripts/baseline_seg_train_v11.py --arch unet --backbone resnet34 --attn cbam --epochs 80 --batch 2 --accum 2 --tag unet_resnet34_cbam_v11 2>&1 | tee runs/baselines/B3_unet_resnet34_cbam.log

echo "=== B4 重跑 (Triplet 修复后) ==="
python twa_unet/scripts/baseline_seg_train_v11.py --arch unet --backbone resnet34 --attn triplet --epochs 80 --batch 2 --accum 2 --tag unet_resnet34_triplet_v11 2>&1 | tee runs/baselines/B4_unet_resnet34_triplet.log

echo "=== [CONTINUE DONE $(date +%Y-%m-%d_%H:%M:%S)] B3+B4 完成 ==="