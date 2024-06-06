import cv2
import numpy as np

from model_inference_library.engines.base import InferenceEngine
from model_inference_library.utils.label_processing import class_file
from model_inference_library.utils.postprocess import postprocessyolov5, postprocessyolov8
from model_inference_library.utils.modeltype import ModelType

class OpenCVDNNEngine(InferenceEngine):
    def __init__(self):
        self.net = None
        self.classes =[]
        self.scale:float

    def load_model(self, model_path):
        # Load the ONNX model
        self.net: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
        #self.net.se

    def load_class_file(self, class_path):
        self.classes = class_file(class_path)
        return self.classes


    def preprocess(self, original_image):
        # Read the input image
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        self.scale = length / 640

        # Preprocess the image and prepare blob for model, opencv 并没有提供相关的读取模型输入宽度和高度的 代码
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        return blob

    def infer(self, preprocessed_image):

        self.net.setInput(preprocessed_image)
        # Perform inference
        results = self.net.forward()
        outputs = np.squeeze(results, 0)  # 删除掉第0维



        return outputs

    def postprocess(self, model: ModelType, outputs, conf, nms, score, detections):

        if model == ModelType.YOLOv8:
            print("ModelType.YOLOv8")
            return postprocessyolov8(outputs, conf, nms, score, self.scale, detections)

        if model == ModelType.YOLOv5:
            print("ModelType.YOLOv5")
            return postprocessyolov5(outputs, conf, nms, score, self.scale, detections)