@echo off
REM TWAU-Net 训练完后,排队训练 6 个基线 (全部基于 v11 数据集, 5 类, 640x640)
REM 顺序: B1 → B2 → B3 → B4 → B5 → B6
REM 每个 80 epoch, 单卡串行,预计总耗时 ~15-18 小时

cd /d D:\chengs\9.project\shuihulu

set PYTHONPATH=twa_unet\scripts;segformer\scripts

echo === B1 U-Net ResNet18 (80ep, vanilla) ===
python twa_unet\scripts\baseline_seg_train_v11.py --arch unet --backbone resnet18 --attn none --epochs 80 --batch 2 --accum 2 --tag unet_resnet18_v11
if errorlevel 1 goto :err

echo === B2 U-Net ResNet34 (80ep, vanilla) ===
python twa_unet\scripts\baseline_seg_train_v11.py --arch unet --backbone resnet34 --attn none --epochs 80 --batch 2 --accum 2 --tag unet_resnet34_v11
if errorlevel 1 goto :err

echo === B3 U-Net ResNet34 + CBAM (80ep) ===
python twa_unet\scripts\baseline_seg_train_v11.py --arch unet --backbone resnet34 --attn cbam --epochs 80 --batch 2 --accum 2 --tag unet_resnet34_cbam_v11
if errorlevel 1 goto :err

echo === B4 U-Net ResNet34 + Triplet Attn (80ep) ===
python twa_unet\scripts\baseline_seg_train_v11.py --arch unet --backbone resnet34 --attn triplet --epochs 80 --batch 2 --accum 2 --tag unet_resnet34_triplet_v11
if errorlevel 1 goto :err

echo === B5 DeepLabV3+ ResNet18 (80ep) ===
python twa_unet\scripts\baseline_seg_train_v11.py --arch deeplabv3p --backbone resnet18 --attn none --epochs 80 --batch 2 --accum 2 --tag dlv3p_resnet18_v11
if errorlevel 1 goto :err

echo === B6 SegFormer-B2 v11 ===
python segformer\scripts\segformer_finetune_v11_fullscreen.py --epochs 80 --batch 2 --accum 2 --tag segformer_b2_v11
if errorlevel 1 goto :err

echo === 全部基线训练完成 ===
exit /b 0

:err
echo !!! 基线训练出错 !!!
exit /b 1