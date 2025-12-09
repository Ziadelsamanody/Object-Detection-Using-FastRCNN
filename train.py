import torch
import torch.nn as nn
import torch.optim as optim
from models import FastRCNN, FastRCNNBoxCode, FastRCNNLoss
import cv2 as cv 
from torch.utils.data import DataLoader
from dataset import VOCFastRCNN

# load data 
data = VOCFastRCNN()
dataloader = DataLoader(data, batch_size=8, shuffle=True)

image , target = next(iter(dataloader))
print(image.shape)