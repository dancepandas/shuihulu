"""SAM → YOLOv8-seg 知识蒸馏损失函数。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 掩膜蒸馏损失 (KL Divergence + Dice Loss)
# ---------------------------------------------------------------------------
def mask_distillation_loss(
    student_masks: torch.Tensor,  # (N, H, W) logits or probs
    teacher_masks: torch.Tensor,  # (N, H, W) binary (0/1)
    temperature: float = 4.0,
) -> torch.Tensor:
    """掩膜层蒸馏损失：KL 散度 + Dice Loss 的组合。

    Parameters
    ----------
    student_masks : (N, H, W) tensor
        学生模型输出的掩膜 logits。
    teacher_masks : (N, H, W) tensor
        教师模型（SAM）输出的二值掩膜。
    temperature : float
        KL 散度的温度参数。

    Returns
    -------
    loss : scalar tensor
    """
    if student_masks.numel() == 0 or teacher_masks.numel() == 0:
        return torch.tensor(0.0, device=student_masks.device)

    # Resize teacher to match student if needed
    if teacher_masks.shape[-2:] != student_masks.shape[-2:]:
        teacher_masks = F.interpolate(
            teacher_masks.unsqueeze(1).float(),
            size=student_masks.shape[-2:],
            mode="nearest",
        ).squeeze(1)

    # KL divergence on softened distributions
    student_soft = F.log_softmax(
        torch.stack([1 - student_masks, student_masks], dim=1) / temperature, dim=1
    )
    teacher_soft = F.softmax(
        torch.stack([1 - teacher_masks, teacher_masks], dim=1).float() / temperature, dim=1
    )
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Dice loss
    student_prob = torch.sigmoid(student_masks)
    intersection = (student_prob * teacher_masks.float()).sum(dim=(1, 2))
    union = student_prob.sum(dim=(1, 2)) + teacher_masks.float().sum(dim=(1, 2))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = (1 - dice).mean()

    return kl_loss + dice_loss


# ---------------------------------------------------------------------------
# 关系蒸馏损失 (Pairwise IoU Frobenius Norm)
# ---------------------------------------------------------------------------
def relation_distillation_loss(
    student_masks: torch.Tensor,  # (N, H, W) logits or probs
    teacher_masks: torch.Tensor,  # (N, H, W) binary (0/1)
) -> torch.Tensor:
    """关系层蒸馏损失：实例间成对 IoU 矩阵的 Frobenius 范数。

    Parameters
    ----------
    student_masks : (N, H, W) tensor
    teacher_masks : (N, H, W) tensor

    Returns
    -------
    loss : scalar tensor
    """
    N = student_masks.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=student_masks.device)

    # Resize teacher to match student if needed
    if teacher_masks.shape[-2:] != student_masks.shape[-2:]:
        teacher_masks = F.interpolate(
            teacher_masks.unsqueeze(1).float(),
            size=student_masks.shape[-2:],
            mode="nearest",
        ).squeeze(1)

    student_prob = torch.sigmoid(student_masks)
    teacher_bin = teacher_masks.float()

    def pairwise_iou(masks: torch.Tensor) -> torch.Tensor:
        """计算成对 IoU 矩阵。"""
        # masks: (N, H, W)
        masks_flat = masks.view(N, -1)  # (N, H*W)
        intersection = masks_flat @ masks_flat.T  # (N, N)
        area = masks_flat.sum(dim=1, keepdim=True)  # (N, 1)
        union = area + area.T - intersection
        iou = intersection / (union + 1e-6)
        return iou

    R_teacher = pairwise_iou(teacher_bin)
    R_student = pairwise_iou(student_prob)

    return F.mse_loss(R_student, R_teacher)


# ---------------------------------------------------------------------------
# 边界框匹配 (用于将教师检测框与学生检测框配对)
# ---------------------------------------------------------------------------
def match_boxes_by_iou(
    teacher_boxes: torch.Tensor,  # (M, 4) xyxy
    student_boxes: torch.Tensor,  # (N, 4) xyxy
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """基于 IoU 匹配教师框与学生框。

    Returns
    -------
    teacher_indices : (K,) long tensor — 匹配到的教师框索引
    student_indices : (K,) long tensor — 对应的学生框索引
    """
    if teacher_boxes.numel() == 0 or student_boxes.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=teacher_boxes.device),
            torch.zeros(0, dtype=torch.long, device=student_boxes.device),
        )

    # Compute pairwise IoU
    iou_matrix = box_iou(teacher_boxes, student_boxes)  # (M, N)

    # Greedy matching: for each teacher box, find best student box
    matched_t, matched_s = [], []
    iou_copy = iou_matrix.clone()

    for _ in range(min(len(teacher_boxes), len(student_boxes))):
        if iou_copy.numel() == 0:
            break
        best_idx = iou_copy.argmax().item()
        t_idx = best_idx // iou_copy.shape[1]
        s_idx = best_idx % iou_copy.shape[1]
        if iou_copy[t_idx, s_idx] < iou_threshold:
            break
        matched_t.append(t_idx)
        matched_s.append(s_idx)
        iou_copy[t_idx, :] = 0
        iou_copy[:, s_idx] = 0

    return (
        torch.tensor(matched_t, dtype=torch.long, device=teacher_boxes.device),
        torch.tensor(matched_s, dtype=torch.long, device=student_boxes.device),
    )


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组边界框的成对 IoU。"""
    # boxes1: (M, 4), boxes2: (N, 4)  xyxy format
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (M, N, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (M, N, 2)
    wh = (rb - lt).clamp(min=0)  # (M, N, 2)

    inter = wh[:, :, 0] * wh[:, :, 1]  # (M, N)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)
