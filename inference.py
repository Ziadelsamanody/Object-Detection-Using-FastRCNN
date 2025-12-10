import torch 
import torch.nn.functional as F 
from torchvision.ops import nms
from models import FastRCNNBoxCode, FastRCNN
from utils.utils import plot_image_with_bbox, VOC_CLASSES
from torchvision.ops.boxes import box_convert
from torchvision import transforms
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/fast_rcnn_voc.pth'
box_coder = FastRCNNBoxCode()

def load_model(model_path = 'models/fast_rcnn_voc.pth'):
    '''load trained model'''
    model = FastRCNN(num_classes=21).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Model loaded from {model_path}')

    return model 


def generate_proposal(h, w, num_proposals= 2000):
    '''generate simple sliding window proposal'''
    proposal = []
    scales = [0.1, 0.2, 0.3, 0.5, 0.7]

    for scale in scales :
        box_h, box_w = int(h * scale), int(w * scale)
        stride_h, stride_w = max(1, box_h // 3), max(1, box_w // 3)

        for y in range(0, h - box_h + 1, stride_h):
            for x in range(0, w - box_w + 1 , stride_w):
                proposal.append([0, x, y, x + box_w ,  y + box_h])  # Add batch_idx=0
    
    proposal = torch.tensor(proposal, dtype=torch.float32)
    proposal[:, [1, 3]] /= w # Normalize x (shifted indices)
    proposal[:, [2, 4]] /= h # Normalize y (shifted indices)

    if len(proposal) > num_proposals :
        indices = torch.randperm(len(proposal))[: num_proposals]
        proposal = proposal[indices]
    return proposal



def inference(model, images, propsal, score_thres = 0.05, nms_thresh=0.5):
    model.eval()
    with torch.no_grad():
        cls_scores, bbox_deltas = model(images, propsal)
        probablities = F.softmax(cls_scores, dim=1)
        bbox_deltas= bbox_deltas.view(-1, model.num_classes, 4)

        all_pred_boxes = []
        all_scores = []
        all_labels = []
        for cls_idx  in range(1, model.num_classes): 
            scores = probablities[:, cls_idx]
            keep = scores > score_thres

            if not keep.any():
                continue

            deltas = bbox_deltas[:, cls_idx][keep]
            boxes = propsal[keep, 1:]  # Remove batch index, get [x1, y1, x2, y2]
            boxes_cxcyhw = box_convert(boxes, "xyxy", 'cxcywh')

            pred_boxes = box_coder.decode_bbox(deltas, boxes_cxcyhw)

            # Nms
            keep_nms = nms(pred_boxes, scores[keep], nms_thresh)
            all_pred_boxes.append(pred_boxes[keep_nms])
            all_scores.append(scores[keep][keep_nms])
            all_labels.append(torch.full_like(scores[keep][keep_nms], cls_idx - 1))

        return all_pred_boxes, all_scores, all_labels


def test_image(image_path, model_path='models/fast_rcnn_voc.pth', output_path='output.jpg'):
    # load model 
    model = load_model(model_path)

    # load image 
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # generate proposal
    proposals = generate_proposal(512, 512).to(device)

    # run inference
    boxes, scores, labels = inference(model, image_tensor, proposals)
    # draw
    if boxes : 
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        for box_list, score_list, label_list in zip(boxes, scores,labels):
            for box, score, label in zip(box_list, score_list, label_list):
                x1, y1,x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1*orig_w), int(y1*orig_h), int(x2*orig_w), int(y2*orig_h)

                class_name = VOC_CLASSES[int(label.item()) + 1]  # +1 because label is 0-indexed but VOC_CLASSES has background at 0

                cv.rectangle(image, (x1,y1), (x2, y2), (0,255,0), 2)
                cv.putText(image, f'{class_name} : {score:.2f}', (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

                print(f"Detected : {class_name} : {score :.3f}")
        cv.imwrite(output_path, cv.cvtColor(image, cv.COLOR_BGR2RGB))
        print(f'saved to output : {output_path}')

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig('output_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("❌ No objects detected")


        
if __name__ == "__main__":
    test_image('testimage/000012.jpg')