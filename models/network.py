import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models
from torchvision.ops import RoIPool, smooth_l1_loss
from torchvision.ops.boxes import box_convert
class FastRCNN(nn.Module):
    def __int__(self, num_classes): 
        super(FastRCNN, self).__init__()
        resnet = models.resnet50(pretraied=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        self.roi_pool = RoIPool(output_size=(7,7), spatial_scale=1.0/16.0)
        
        self.fc1 = nn.Linear(1024 *7 *7 , 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.classifier = nn.Linear(4096, num_classes)
        self.bbox_reg = nn.Linear(4096, num_classes * 4)

    def forward(self, image, rois):
        features = self.backbone(image)

        roi_out = self.roi_pool(features, rois)
        roi_out = roi_out.view(roi_out.size(0), -1)

        fc1 = F.relu(self.fc1(roi_out))
        fc2 = F.relu(self.fc2(fc1))
        
        cls_scores = self.classifier(fc2)
        bbox_deltas = self.bbox_reg(fc2)

        return cls_scores, bbox_deltas
    

class FastRCNNLoss(nn.Module):
    def __init__(self, box_reg_weights = (10., 10., 5., 5.)):
        self.box_reg_weights = torch.tensor(box_reg_weights)

    
    def smooth_l1_loss(self, pred, target, beta=1.0):
        '''
        smooth l1 loss 
        '''
        diff = torch.abs(pred - target)
        less_than_one = (diff < beta).float()

        loss = less_than_one *0.5 * diff **2 / beta + (1 - less_than_one) * (diff - 0.5 * beta)
        return loss
    
    def forward(self, cls_scores, bbox_pred, labels, bbox_targets):
        # bbox targets the encoded tx,ty,tw,th 
        # classfication loss
        # apply this when  cls >= 0
        valid_mask = labels >= 0 
        cls_loss = F.cross_entropy(cls_scores[valid_mask], labels[valid_mask])

        # Regression loss 
        pos_mask = labels > 0  # labels >  0  regress the labels not background
        if pos_mask.sum() == 0 :

            return cls_loss,  torch.zeros_like(cls_loss)
        
        # select bbox prediction for the matched class 
        label_pos = labels[pos_mask]
        N, C4 = bbox_pred.shape
        C  = C4 // 4 

        # Reshape to N, C, 4
        bbox_pred = bbox_pred.view(N,C, 4)
        bbox_pred_pos = bbox_pred[pos_mask, label_pos] # [num_pos, 4]

        bbox_targets_pos = bbox_targets[pos_mask]  # same dim 

        # apply offcial weights 
        bbox_pred_pos = bbox_pred_pos * self.box_reg_weights.to(bbox_pred_pos.device)
        bbox_targets_pos = bbox_targets_pos * self.box_reg_weights.to(bbox_targets_pos.device)

        bbox_loss = smooth_l1_loss(bbox_pred_pos, bbox_targets_pos, beta=1.0)

        return cls_loss, bbox_loss