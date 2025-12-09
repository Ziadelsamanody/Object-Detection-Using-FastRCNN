import numpy as np
import torch 
from torchvision.utils import make_grid
from PIL import Image 
import matplotlib.pyplot as plt 


def plot_image(image, label=None, tensor_image = False):
    if tensor_image:
        image = image.squeeze().cpu().numpy()
    else : 
        plt.imshow(image)

        plt.show()
        
