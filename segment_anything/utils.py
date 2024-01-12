import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import pytesseract


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 200 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def load_img(which):
    img_path = (
        "segment_anything/images/pdf_table0.jpg"
        if which == 0
        else "segment_anything/images/pdf_table1.png"
        if which == 1
        else "segment_anything/images/pdf_table_t0.jpg"
        if which ==50
        else "segment_anything/images/dog.jpg"
        if which == 100
        else "segment_anything/images/truck.jpg"
        if which ==101
        else "segment_anything/images/truck.jpg"
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_path

def show_img_with_point(image, input_point, input_label=np.array([1])):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    # show_mask(mask, plt.gca())
    if not isinstance(input_point, np.ndarray):
        input_point = np.array([input_point])
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()  

def ocr_text_from_yolo_results(results):
    texts = []
    img=results[0].orig_img
    texts = []
    for box in results[0].boxes.xyxy:
        x1,y1,x2,y2=map(lambda x: max(0,int(x)),box)
        # print(box, x1,x2,y2,y2)
        if y2-y1>0 and x2-x1>0:
            cell=img[y1:y2, x1:x2]
            text = pytesseract.image_to_string(cell)
            texts.append(text)
    return texts