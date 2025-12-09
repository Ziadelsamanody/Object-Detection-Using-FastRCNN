import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models
from torchvision.ops import RoIPool


class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        resnet = models.resnet50(weights=True)
        self.backbone = nn.Sequential(*list(resnet.children())[: - 3])
        # Input image is 3 * 512*512
        # backbone output will be 1024 * 32*32 channels
        # so spatial_scale = 32 /512 = 1/16
        # we want to extract ROI and pool each 7x7
        self.roi_pool = RoIPool(output_size=(7,7), spatial_scale=1.0/16.0)
        # Each roi will be 1024 x 7 x 7
        self.fc1 = nn.Linear(1024 * 7 * 7, 8192)
        self.fc2 = nn.Linear(8192, 4096)

        # classfication and bounding box regression head
        self.classifier = nn.Linear(4096, num_classes)
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)
    
    def forward(self, images, rois):
        feature_map = self.backbone(images) # [1, 1024, 32, 32]
        # RoI pooling and reshape we have 4 RoIS
        # so will have 4 results, each  is 1024 *7 *7 
        roi_pooled = self.roi_pool(feature_map, rois) # 4, 1024, 7,7
        roi_pooled = roi_pooled.view(roi_pooled.size(0), -1) # 4, 50176

        fc1 = F.relu(self.fc1(roi_pooled)) # 4,8192
        fc2 = F.relu(self.fc2(fc1)) # 4, 4096
        class_scores = self.classifier(fc2)
        bbox_deltas = self.bbox_regressor(fc2) # 4, 40  4* 10
        return class_scores, bbox_deltas