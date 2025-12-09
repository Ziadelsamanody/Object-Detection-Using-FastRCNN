import torch
from torch.utils.data import Dataset 
from utils import plot_image, plot_image_with_bbox
import numpy as np 
from torchvision.datasets import VOCDetection

def voc_to_fast_rcnn(target, label_map):
    objs = target['annotation']['object']
    if not isinstance(objs, list): 
        objs = [objs]
    
    boxes = []
    labels = []
    for obj in objs:
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[obj['name']])
    return {
        'boxes' : torch.tensor(boxes, dtype=torch.float32),
        'labels' : torch.tensor(labels, dtype=torch.int64),
    }


VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]


label_map = {clss : i + 1 for i, clss in enumerate(VOC_CLASSES)}  # starting from 1 cause 0 for background


class VOCFastRCNN(Dataset):
    def __init__(self, year='2007', split='train', transform=None):
        try:
            self.dataset = VOCDetection(
                root='./data',
                year=year,
                image_set=split,
                download=True
            )
        except RuntimeError:
            print("Auto-download failed. Attempting to use existing data...")
            self.dataset = VOCDetection(
                root='./data',
                year=year,
                image_set=split,
                download=False
            )
        self.tranform = transform
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        target = voc_to_fast_rcnn(target, label_map)
        if self.tranform :
            self.tranform(img)

        return img, target


if __name__ == "__main__":
    data = VOCFastRCNN(year='2007', split='train')
    img, target = data[1]
    print(np.array(img).shape)
    print(target)
    # plot_image(img, label=target['labels'])
    plot_image_with_bbox(img, target['labels'], box= target["boxes"])