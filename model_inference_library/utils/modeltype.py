from enum import Enum

#   模型的选择，目前做了v8和v5 ，可以拓展到ssd  fastrcnn等算法
class ModelType(Enum):
    YOLOv8 = "yolov8"
    YOLOv5 = "yolov5"


