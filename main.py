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
    
# if __name__ == "__main__":
#     # create a model 
#     num_classes = 10 
#     model = FastRCNN(num_classes)

#     # dummy_input_data 
#     images = torch.randn(1, 3,  512, 512) # batch 1 image
#     rois = torch.tensor([[0, 100, 100, 200, 200],
#                          [0, 50, 100, 200, 210],
#                          [0, 10, 35, 78, 100],
#                          [0, 10, 20, 70, 70],
#                          ], dtype=torch.float) 
#       # Forward pass
#     class_scores, bbox_deltas = model(images, rois)
#     print("Class Scores:", class_scores[0])
#     print("Bounding Box Deltas:", bbox_deltas[0])

#     Class Scores: tensor([-0.0093, -0.0080, -0.0022, -0.0021,  0.0103, -0.0175,  0.0183,  0.0079,
#         -0.0081,  0.0135], grad_fn=<SelectBackward0>)
# Bounding Box Deltas: tensor([ 0.0119, -0.0026, -0.0049,  0.0092, -0.0074,  0.0194, -0.0265,  0.0215,
#         -0.0061, -0.0207,  0.0214,  0.0071, -0.0037, -0.0070,  0.0012,  0.0108,
#         -0.0085, -0.0085, -0.0071,  0.0184, -0.0242, -0.0161, -0.0070,  0.0132,
#         -0.0255,  0.0192,  0.0021,  0.0036,  0.0003, -0.0123, -0.0023, -0.0071,
#          0.0352,  0.0135, -0.0165,  0.0119,  0.0366,  0.0119, -0.0036, -0.0105],
#        grad_fn=<SelectBackward0>)


class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()
        resnet = models.resnet50(weights=True)
        self.backbone = nn.Sequential(*list(resnet.children())[: - 3])

        self.roi_pool = RoIPool(output_size=(7,7), spatial_scale=1.0 / 16.0)

        self.fc1 = nn.Linear(1024 * 7 * 7, 8192)
        self.fc2 = nn.Linear(8192, 4096)

        self.classifier = nn.Linear(4096, num_classes)
        self.bbx_reg = nn.Linear(4096, num_classes * 4)

    def forward(self, image, rois):
        features = self.backbone(image)

        roi_pool = self.roi_pool(features, rois)
        roi_pool = roi_pool.view(roi_pool.size(0), -1)

        fc1 = F.relu(self.fc1(roi_pool))
        fc2 = F.relu(self.fc2(fc1))

        class_scores = self.classifier(fc2)
        bbox_deltas = self.bbx_reg(fc2)

        return class_scores ,bbox_deltas
if __name__ == "__main__":
    # create a model 
    num_classes = 10 
    model = FastRCNN(num_classes)

    # dummy_input_data 
    images = torch.randn(1, 3,  512, 512) # batch 1 image
    rois = torch.tensor([[0, 100, 100, 200, 200],
                         [0, 50, 100, 200, 210],
                         [0, 10, 35, 78, 100],
                         [0, 10, 20, 70, 70],
                         ], dtype=torch.float) 
      # Forward pass
    class_scores, bbox_deltas = model(images, rois)
    print("Class Scores:", class_scores[0])
    print("Bounding Box Deltas:", bbox_deltas[0])