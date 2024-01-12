import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
# import pydicom
from utils import show_mask, show_points, show_box, load_img,show_img_with_point

from segment_anything import sam_model_registry, SamPredictor

image, img_path=load_img()
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# checkpoint = "weights/segment_anything/sam_vit_h_4b8939.pth"
checkpoint = "weights/segment_anything/sam_vit_b_01ec64.pth"
# model_type = "vit_h"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
masks, scores, logits = predictor.predict(
    multimask_output=True,
)
print(masks.shape)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  