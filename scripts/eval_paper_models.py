"""论文评测脚本 — 统一验证集 (datasets/hyacinth_seg/val, 71张/68含WH) 上评测三个模型:
  - 本文方法:    runs/segformer/segformer_b2_ls+v9  (针对性优化 SegFormer-B2)
  - 直接微调基线: runs/segformer/segformer_b2_v2    (标准交叉熵直接微调)
  - 两阶段基线:  runs/hyacinth8_yolo_sam + SAM ViT-H

输出: 各方法的 WH IoU / mIoU / 参数量 / 推理耗时 (对应论文表 1)。
"""
import os, time, torch, torch.nn as nn, numpy as np, cv2
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

ROOT = r"D:\chengs\9.project\shuihulu"
VAL_IMG = os.path.join(ROOT, "datasets", "hyacinth_seg", "images", "val")
VAL_MASK = os.path.join(ROOT, "datasets", "hyacinth_seg", "masks", "val")
SZ = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
files = sorted(f for f in os.listdir(VAL_IMG) if f.lower().endswith((".jpg", ".png", ".jpeg")))


def eval_segformer(path):
    """SegFormer 评测: 返回 per-class IoU (混淆矩阵口径), mAcc, mIoU, WH IoU 与 GPU 推理耗时"""
    model = SegformerForSemanticSegmentation.from_pretrained(path).to(dev).eval()
    nparams = sum(p.numel() for p in model.parameters())
    cm = np.zeros((6, 6), np.int64)  # 混淆矩阵 gt x pred
    times = []
    with torch.no_grad():
        for f in files:
            img = cv2.cvtColor(cv2.imread(os.path.join(VAL_IMG, f)), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            r = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)
            t0 = time.perf_counter()
            logits = nn.functional.interpolate(model(pixel_values=t).logits,
                                               size=(H, W), mode="bilinear", align_corners=False)
            times.append(time.perf_counter() - t0)
            pred = logits[0].argmax(0).cpu().numpy()
            gt = np.squeeze(cv2.imread(os.path.join(VAL_MASK, os.path.splitext(f)[0] + ".png"),
                                       cv2.IMREAD_GRAYSCALE))
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            for g in range(6):
                for p in range(6):
                    cm[g, p] += ((gt == g) & (pred == p)).sum()
    valid = [i for i in range(6) if cm[i].sum() > 0]
    mAcc = float(np.mean([cm[i, i] / cm[i].sum() for i in valid])) * 100
    ious = {}
    for i in range(6):
        if cm[i].sum() > 0:
            ious[i] = cm[i, i] / (cm[i].sum() + cm[:, i].sum() - cm[i, i] + 1e-6)
    mIoU = float(np.mean(list(ious.values()))) * 100
    return mAcc, mIoU, ious.get(4, 0.0), nparams, np.mean(times[2:]) * 1000


def eval_yolo_sam(yolo_pt, sam_ckpt, conf=0.10):
    """YOLO+SAM 两阶段评测: WH IoU (实例掩膜合并为语义掩膜) 与 SAM 编码耗时"""
    yolo = YOLO(yolo_pt)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(dev)
    pred = SamPredictor(sam)
    inter = union = 0.0; n_det = wh_gt = 0; t_enc = []
    with torch.no_grad():
        for f in files:
            img = cv2.cvtColor(cv2.imread(os.path.join(VAL_IMG, f)), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            gt = np.squeeze(cv2.imread(os.path.join(VAL_MASK, os.path.splitext(f)[0] + ".png"),
                                       cv2.IMREAD_GRAYSCALE))
            g = (gt == 4)
            if g.sum() == 0:
                continue
            wh_gt += 1
            r = yolo(img, conf=conf, iou=0.5, imgsz=640, verbose=False)[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            cls = boxes.cls.cpu().numpy().astype(int)
            wh_boxes = boxes.xyxy.cpu().numpy()[cls == 3]  # 水葫芦类
            if len(wh_boxes) == 0:
                continue
            n_det += 1
            t0 = time.perf_counter(); pred.set_image(img); t_enc.append(time.perf_counter() - t0)
            merged = np.zeros((H, W), np.uint8)
            for b in wh_boxes:
                m, _, _ = pred.predict(box=b[None, :], multimask_output=False)
                mm = cv2.resize(m[0].astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST) > 0
                merged[mm] = 1
            inter += ((merged > 0) & g).sum(); union += ((merged > 0) | g).sum()
    return inter / (union + 1e-6), n_det, wh_gt, np.mean(t_enc) * 1000


if __name__ == "__main__":
    print("=== 统一验证集: hyacinth_seg/val (%d张) ===" % len(files))
    for tag, p in [("本文(ls+v9)", r"runs\segformer\segformer_b2_ls+v9"),
                   ("直接微调(v2)", r"runs\segformer\segformer_b2_v2")]:
        mAcc, mIoU, wh, np_, ms = eval_segformer(os.path.join(ROOT, p))
        print(f"[{tag}] mAcc={mAcc:.1f}%  mIoU={mIoU:.1f}%  WH IoU={wh:.3f} "
              f"| params={np_/1e6:.2f}M  {ms:.1f}ms/图")
    wh_iou, n_det, wh_gt, enc_ms = eval_yolo_sam(
        os.path.join(ROOT, "runs", "hyacinth8_yolo_sam", "weights", "best.pt"),
        os.path.join(ROOT, "weights", "sam_vit_h.pth"))
    print(f"[两阶段YOLO+SAM] WH IoU={wh_iou:.3f} (检出{n_det}/{wh_gt}张) SAM编码={enc_ms:.0f}ms/图")
