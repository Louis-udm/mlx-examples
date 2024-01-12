from utils import (
    ocr_text_from_yolo_results,
    show_mask,
    show_points,
    show_box,
    load_img,
    show_img_with_point,
)
from ultralytics import YOLO
import cv2

# 加载预训练模型
model = YOLO("yolov8n.pt", task="detect")
# model = YOLO("yolov8n-seg.pt")
# model = YOLO("yolov8n.pt")  # task参数也可以不填写，它会根据模型去识别相应任务类别
# model=YOLO("yolov5s.pt")
print(model.info())

# from roboflow import Roboflow
# rf = Roboflow(api_key="jyL4bQg28eekCd1Z6nMs")
# project = rf.workspace("zhibinlu").project("tabulation")
# dataset = project.version(1).download("yolov8")


img, img_path = load_img(1)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

img, img_path = load_img(50)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

img, img_path = load_img(100)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

# 训练模型100个epoch
# train_res = model.train(
#     data="segment_anything/images/tabulation.v3i.yolov8/data.yaml",
#     epochs=200,
#     name="yolov8_tabulation",
#     device="mps",
#     batch=8,
#     imgsz=730,
# )

# model.export(format='onnx', dynamic=True)
# print(model.val())

# 加载last.pt, best.pt, 比较
model = YOLO(
    "/Users/zlu/Projects/mlx-examples/runs/detect/yolov8_tabulation18/weights/last.pt",
    task="detect",
)
img, img_path = load_img(0)
# 分析整张图片
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

img, img_path = load_img(1)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

texts = ocr_text_from_yolo_results(results)
print("\n".join(texts))

img, img_path = load_img(50)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

img, img_path = load_img(100)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

img, img_path = load_img(101)
results = model(img_path)
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

texts = ocr_text_from_yolo_results(results)
print("\n".join(texts))
