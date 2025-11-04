"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



  # NWD          multiUAV中C=20，UAVS中用80
def NWD(pred, target, eps=1e-7, constant=80, img_size=(640, 512)):#快速版实现
    """
    Optimized NWD calculation with absolute coordinates.
    NWD计算时用的绝对坐标，如果不然，wasserstein_2算出来全趋近于0，最后取指数得到的全是1
    Args:
        pred (Tensor): Predicted boxes in normalized [x_center, y_center, width, height] format.
        target (Tensor): Target boxes in normalized [x_center, y_center, width, height] format.
        eps (float): Small value for numerical stability.
        constant (float): Scaling factor (originally 12.8, typically set to 5-10).
        img_size (tuple): Image (height, width) for normalization.
        基于 (dcx)^2 + (dcy)^2 + (dw)^2 + (dh)^2 
    Returns:
        Tensor: NWD scores (higher means more similar).
    """
    h, w = img_size
    
    # Vectorized absolute coordinate conversion
    scale = torch.tensor([w, h, w, h], device=pred.device, dtype=pred.dtype)
    pred_abs = pred * scale
    target_abs = target * scale
    
    # Center distance (squared L2)
    center_diff = pred_abs[:, :2] - target_abs[:, :2]
    center_distance = (center_diff ** 2).sum(dim=1) + eps
    
    # Width-height distance
    wh_diff = pred_abs[:, 2:] - target_abs[:, 2:]
    wh_distance = (wh_diff ** 2).sum(dim=1) / 4 + eps
    
    # Combined Wasserstein distance
    wasserstein_2 = center_distance + wh_distance
    
    # NWD = exp(-sqrt(wasserstein_2)/constant)
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


def NWD_cost(pred, target, eps=1e-7, constant=20, img_size=(640, 512)):
    """
    Compute pairwise NWD between all predicted and target boxes.
    
    Args:
        pred (Tensor): Predicted boxes [num_pred, 4] (normalized cxcywh format).
        target (Tensor): Target boxes [num_target, 4] (normalized cxcywh format).
        eps (float): Small value for numerical stability.
        constant (float): Scaling factor (typically 5-10).
        img_size (tuple): Image (height, width) for denormalization.
    
    Returns:
        Tensor: NWD matrix [num_pred, num_target], where higher values indicate more similarity.
    """
    h, w = img_size
    # Convert to absolute coordinates [num_pred, 4] and [num_target, 4]
    scale = torch.tensor([w, h, w, h], device=pred.device, dtype=pred.dtype)
    pred_abs = pred * scale  # [num_pred, 4]
    target_abs = target * scale  # [num_target, 4]
    
    # Compute center distances [num_pred, num_target]
    center_diff = pred_abs[:, None, :2] - target_abs[None, :, :2]  # Broadcasting
    center_distance = (center_diff ** 2).sum(dim=-1) + eps  # [num_pred, num_target]
    
    # Compute width-height distances [num_pred, num_target]
    wh_diff = pred_abs[:, None, 2:] - target_abs[None, :, 2:]  # Broadcasting
    wh_distance = (wh_diff ** 2).sum(dim=-1) / 4 + eps  # [num_pred, num_target]
    
    # Combined Wasserstein distance
    wasserstein_2 = center_distance + wh_distance  # [num_pred, num_target]
    
    # NWD = exp(-sqrt(wasserstein_2)/constant)
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)  # [num_pred, num_target]