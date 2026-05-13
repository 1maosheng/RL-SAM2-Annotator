import numpy as np
from PIL import Image

GRID_SIZE = None  # 将在训练时从config设置，或者作为参数传入


def set_grid_size(size):
    global GRID_SIZE
    GRID_SIZE = size


def action_to_coord(action_idx, img_size):
    """将离散动作索引 (0..grid_size^2-1) 转换为图像坐标 (x,y)"""
    if GRID_SIZE is None:
        raise ValueError("请先调用 set_grid_size()")
    row = action_idx // GRID_SIZE
    col = action_idx % GRID_SIZE
    cell_w = img_size / GRID_SIZE
    cell_h = img_size / GRID_SIZE
    x = int((col + 0.5) * cell_w)
    y = int((row + 0.5) * cell_h)
    x = max(0, min(x, img_size - 1))
    y = max(0, min(y, img_size - 1))
    return x, y


def calculate_iou(pred_mask, gt_mask):
    """二值掩码 IoU"""
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def calculate_miou(pred_mask, gt_mask):
    """二分类 mIoU（缺陷类 + 背景类）"""
    iou_defect = calculate_iou(pred_mask, gt_mask)
    iou_bg = calculate_iou(1 - pred_mask, 1 - gt_mask)
    return (iou_defect + iou_bg) / 2.0


def build_state_tensor(image, prev_masks, img_size, max_hist):
    """
    构建状态张量：RGB图像 + 历史掩码
    返回 torch.Tensor (C, H, W)
    """
    from torch import from_numpy
    # 调整图像尺寸
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image.astype(np.uint8)).resize((img_size, img_size)))
    img_norm = image.astype(np.float32) / 255.0  # (H,W,3)

    # 处理历史掩码
    masks_resized = []
    for m in prev_masks[-max_hist:]:
        if m.shape[0] != img_size or m.shape[1] != img_size:
            m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((img_size, img_size))) / 255.0
        else:
            m = m.astype(np.float32)
        masks_resized.append(m)
    while len(masks_resized) < max_hist:
        masks_resized.append(np.zeros((img_size, img_size), dtype=np.float32))

    # 拼接
    state = np.concatenate([img_norm] + [m[..., np.newaxis] for m in masks_resized], axis=-1)
    state = from_numpy(state).permute(2, 0, 1).float()
    return state