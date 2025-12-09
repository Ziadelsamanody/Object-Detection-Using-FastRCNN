import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torch.optim as optim
from models import FastRCNN, FastRCNNBoxCode, FastRCNNLoss
import cv2 as cv 
from utils import plot_image, plot_image_with_bbox
from torch.utils.data import DataLoader
from dataset import VOCFastRCNN

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
dataloader = DataLoader(data, batch_size=8,shuffle=True, collate_fn=custom_collate_fn)

image , target = next(iter(dataloader))
print(image[0].shape)
