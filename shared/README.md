# shared — 数据集构建工具（两条分割路线共用）

Label Studio 标注的拉取、RLE 解码、多源合并、train/val 划分工具。
v10/v11 数据集同时被 [twa_unet/](../twa_unet/) 与 [segformer/](../segformer/) 使用，故工具不归属单一路线。

## 目录

```text
pull_ls_p10.py              # 拉 LS project 10 元数据 + 原图 → data/ls_export_p10/
pull_ls_p10_masks.py        # 解码 value.rle → 每标签 PNG
pull_all_p10.py             # 一键: 拉全部 completed → 解码 → 合并 → 切分 → datasets/hyacinth_ls_v11/
pull_v12.py / dl_latest.py  # v12 拉取 / 最新导出
download_p7o_p9_images.py / pull_7_quanmianji.py / import_quanmianji.py   # p7/p9/全面积数据
build_hyacinth_ls_v10.py    # 合并 PNG → 5 类索引 mask → datasets/hyacinth_ls_v10/
build_seg_dataset_p7p9.py   # p7/p9 → 6 类分割数据集
ls_to_dataset.py / split_seg_dataset.py   # 格式转换 / train/val 划分
_probe_*.py / _p7_*.py / _inspect_p7.py / _decode_check.py / _verify_rle.py
                            # LS 导出探查/校验 (一次性调试)
_upload_v9.py / _review_p10_samples.py / _count_p10.py
download_dataset.py         # kaggle 公开数据集下载 (参考用)
test.py                     # cv2 可视化 demo
tests/                      # LS 导出检查 / 数据集重组 / yolo_sam 管线测试
```

## 典型闭环（预标注 → 人工修正 → 重训）

```bash
python segformer/scripts/preanno_v10_ls.py   # 模型预标注推送到 LS
# (Label Studio 人工修正)
python shared/pull_all_p10.py                # 拉回 → 新数据集
# (改训练脚本 DATASET_DIR 重训)
```
