import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
# import pydicom

from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 200/255, 0/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)   
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def load_img():
    # Read the image and convert to grayscale
    img = cv2.imread(f'../llm-service/trying/out-0.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

img=load_img()
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# checkpoint = "weights/segment_anything/sam_vit_h_4b8939.pth"
checkpoint = "weights/segment_anything/sam_vit_b_01ec64.pth"
# model_type = "vit_h"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(img)
masks, _, _ = predictor.predict("table")
print(masks)