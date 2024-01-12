from ultralytics import SAM
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import pytesseract
from tabulate import tabulate

from utils import ocr_text_from_yolo_results, show_mask, show_points, show_box, load_img,show_img_with_point

def make_markdown_table(text):
    # Assuming the recognized text is tabular or can be split into a list of lists
    rows = text.split("\n")
    table_data = [row.split() for row in rows]

    # Convert the table data to markdown
    markdown = tabulate(table_data, tablefmt="pipe")

    # Print the markdown
    print(markdown)

# 载入模型
# model = SAM('mobile_sam.pt')
model = SAM('sam_b.pt')
print(model.info())
img, img_path = load_img(0)
# 分析整张图片
results=model(img_path)

res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)
texts=ocr_text_from_yolo_results(results)
print("\n".join(texts))


exit(0)

# 使用一个点作为prompt，预测一个mask
# point=[900, 370] # truck
# point=[1000, 3000] # pdf table
# point=[1200, 3000] # pdf table
point=[2000, 3000] # pdf table
show_img_with_point(img, point)
# 基于点提示预测一个分段
# results=model.predict("segment_anything/images/truck.jpg", points=[900, 370], labels=[1])
# results=model.predict("segment_anything/images/pdf_table.jpg", points=point, labels=[1])
results=model.predict(img_path, points=point, labels=[1])
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)