import numpy as np
import torch 
from torchvision.utils import make_grid
import cv2 as cv 
import matplotlib.pyplot as plt 


VOC_CLASSES = [
    'background',"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]

def plot_image(image, label=None, tensor_image = False):
    if tensor_image:
        image = image.squeeze().cpu().numpy()
    if label is not None:
        plt.title(f"{VOC_CLASSES[label]}")
 
    plt.imshow(image)

    plt.show()
        

def plot_image_with_bbox(image, label, box,tensor_image=False):
    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image.copy()
    box = box[0].numpy()
    image = cv.rectangle(image, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0,0,255), thickness=1)
    plt.imshow(image)
    plt.show()

    