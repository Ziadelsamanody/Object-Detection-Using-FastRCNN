import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torch.optim as optim
from models import FastRCNN, FastRCNNBoxCode, FastRCNNLoss
import cv2 as cv 
from utils import plot_image, plot_image_with_bbox
from utils import generate_rois_from_gt, match_rois_to_gt
from torch.utils.data import DataLoader
from dataset import VOCFastRCNN
from torch.utils.tensorboard import SummaryWriter
import os

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized images and targets
    
    Args:
        batch: List of tuples (image, target)
    
    Returns:
        images: List of tensors with different sizes
        targets: List of dictionaries with boxes and labels
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets


transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to fixed size
    transforms.ToTensor()
])

# load data 
data = VOCFastRCNN(transform=transform)
dataloader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FastRCNN(num_classes=21).to(device)
criterion = FastRCNNLoss().to(device)  # Move criterion to device

optimizer = optim.Adam(model.parameters(), lr=0.001)

box_coder = FastRCNNBoxCode()

# Setup TensorBoard
log_dir = 'runs/fast_rcnn_training'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
print(f"📊 TensorBoard logging to: {log_dir}")

epochs = 10
global_step = 0

for epoch in range(epochs):
    model.train()
    epoch_total_loss = 0
    epoch_cls_loss = 0
    epoch_bbox_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        # stack image into a batch tensor 
        image_tensor = torch.stack(images).to(device)

        optimizer.zero_grad()
        
        batch_loss = 0
        batch_cls_loss = 0
        batch_bbox_loss = 0
        
        for i in range(len(images)):
            img = image_tensor[i : i + 1]
            gt_boxes = targets[i]['boxes'].to(device)
            gt_label = targets[i]['labels'].to(device)
            
            # generate roi
            rois = generate_rois_from_gt(gt_boxes.cpu(), image_size=(512,512), num_negative_samples=128)
            rois = rois.to(device)
            
            # matched rois from gt
            matched_labels, matched_boxes = match_rois_to_gt(
                rois, gt_boxes, gt_label,
                pos_iou_thresh=0.5, neg_iou_thresh=0.1
            )
            
            matched_labels = matched_labels.to(device)
            matched_boxes = matched_boxes.to(device)

            # encode boxes 
            proposal_boxes = rois[:, 1:]
            bbox_target = box_coder.encode_bbox(matched_boxes, proposal_boxes).to(device)

            # Forward pass
            cls_scores, bbox_pred = model(img, rois)
            
            # Compute loss
            cls_loss, bbox_loss = criterion(cls_scores, bbox_pred, matched_labels, bbox_target)

            # Accumulate losses (keep as tensors for backprop)
            batch_loss += (cls_loss + bbox_loss)
            batch_cls_loss += cls_loss.item()
            batch_bbox_loss += bbox_loss.item()
        
        # Average loss over images in batch
        batch_loss = batch_loss / len(images)
        batch_cls_loss = batch_cls_loss / len(images)
        batch_bbox_loss = batch_bbox_loss / len(images)
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()

        # Log to TensorBoard
        writer.add_scalar('Loss/Total', batch_loss.item(), global_step)
        writer.add_scalar('Loss/Classification', batch_cls_loss, global_step)
        writer.add_scalar('Loss/BBox_Regression', batch_bbox_loss, global_step)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

        # Log the loss value
        epoch_total_loss += batch_loss.item()
        epoch_cls_loss += batch_cls_loss
        epoch_bbox_loss += batch_bbox_loss
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                  f"Total Loss: {batch_loss.item():.4f}, Cls: {batch_cls_loss:.4f}, BBox: {batch_bbox_loss:.4f}")
    
    avg_epoch_loss = epoch_total_loss / num_batches
    avg_cls_loss = epoch_cls_loss / num_batches
    avg_bbox_loss = epoch_bbox_loss / num_batches
    
    # Log epoch averages
    writer.add_scalar('Loss/Epoch_Total', avg_epoch_loss, epoch)
    writer.add_scalar('Loss/Epoch_Classification', avg_cls_loss, epoch)
    writer.add_scalar('Loss/Epoch_BBox_Regression', avg_bbox_loss, epoch)
    
    print(f"Epoch [{epoch+1}/{epochs}] completed, Avg Loss: {avg_epoch_loss:.4f}, "
          f"Cls: {avg_cls_loss:.4f}, BBox: {avg_bbox_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'fast_rcnn_voc.pth')
writer.close()
print("✅ Training completed!")
print("📊 TensorBoard logs saved to:", log_dir)