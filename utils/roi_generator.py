import torch
import numpy as np
import cv2

def generate_rois_from_gt(gt_boxes, image_size, num_negative_samples=128):
    """
    Generate ROIs from ground truth boxes + random negative samples
    (Used during training)
    
    Args:
        gt_boxes: Ground truth boxes [N, 4] in format [x1, y1, x2, y2]
        image_size: Tuple (height, width)
        num_negative_samples: Number of random background proposals
    
    Returns:
        rois: Tensor [M, 5] where M = N + num_negative_samples
    """
    H, W = image_size
    rois = []
    
    # Add ground truth boxes
    for box in gt_boxes:
        rois.append([0] + box.tolist())
    
    # Add random negative samples
    for _ in range(num_negative_samples):
        x1 = np.random.randint(0, W - 20)
        y1 = np.random.randint(0, H - 20)
        x2 = np.random.randint(x1 + 10, min(x1 + 200, W))
        y2 = np.random.randint(y1 + 10, min(y1 + 200, H))
        rois.append([0, x1, y1, x2, y2])
    
    rois_tensor = torch.tensor(rois, dtype=torch.float32)
    
    # Normalize coordinates to [0, 1] range
    rois_tensor[:, 1] /= W  # x1
    rois_tensor[:, 2] /= H  # y1
    rois_tensor[:, 3] /= W  # x2
    rois_tensor[:, 4] /= H  # y2
    
    return rois_tensor


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_rois_to_gt(rois, gt_boxes, gt_labels, image_size=(512, 512), pos_iou_thresh=0.5, neg_iou_thresh=0.1):
    """
    Match ROIs to ground truth boxes based on IoU
    
    Args:
        rois: Tensor [N, 5] - proposals in normalized coords (can be on CUDA)
        gt_boxes: Tensor [M, 4] - ground truth boxes in absolute coords
        gt_labels: Tensor [M] - ground truth labels
        image_size: Tuple (height, width) for denormalization
        pos_iou_thresh: IoU threshold for positive samples
        neg_iou_thresh: IoU threshold for negative samples
    
    Returns:
        matched_labels: Tensor [N] - labels for each ROI (-1=ignore, 0=background, >0=class)
        matched_boxes: Tensor [N, 4] - matched GT boxes (normalized)
    """
    # Move tensors to CPU for numpy operations
    rois_cpu = rois.cpu()
    gt_boxes_cpu = gt_boxes.cpu()
    gt_labels_cpu = gt_labels.cpu()
    
    H, W = image_size
    
    num_rois = rois_cpu.shape[0]
    matched_labels = torch.full((num_rois,), -1, dtype=torch.long)
    matched_boxes = torch.zeros((num_rois, 4), dtype=torch.float32)
    
    for i, roi in enumerate(rois_cpu):
        # Denormalize ROI for IoU computation
        roi_norm = roi[1:].numpy()  # Remove batch index [x1, y1, x2, y2] normalized
        roi_box = roi_norm * np.array([W, H, W, H])  # Convert to absolute
        
        max_iou = 0
        max_idx = -1
        
        # Find best matching GT box
        for j, gt_box in enumerate(gt_boxes_cpu):
            iou = compute_iou(roi_box, gt_box.numpy())
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        # Assign label based on IoU
        if max_iou >= pos_iou_thresh:
            matched_labels[i] = gt_labels_cpu[max_idx]
            # Store normalized GT boxes
            gt_box_abs = gt_boxes_cpu[max_idx]
            matched_boxes[i] = gt_box_abs / torch.tensor([W, H, W, H], dtype=torch.float32)
        elif max_iou < neg_iou_thresh:
            matched_labels[i] = 0  # Background
            # For background, use the ROI itself as dummy box
            matched_boxes[i] = roi_norm
    
    return matched_labels, matched_boxes