import numpy as np
import cv2
from model_inference_library.engines.base import InferenceEngine
from model_inference_library.utils.label_processing import class_file
from model_inference_library.utils.modeltype import ModelType
from model_inference_library.utils.postprocess import postprocessyolov5, postprocessyolov8

import os
os.environ['Path'] += r'C:\Users\bigfly\anaconda3\envs\pyqt\Lib\site-packages\openvino\inference_engine;'\
r'C:\Users\bigfly\anaconda3\envs\pyqt\Lib\site-packages\openvino;'\
r'C:\Users\bigfly\anaconda3\envs\pyqt\Lib\site-packages\openvino\libs;'

import openvino as ov

class OpenVINOEngine(InferenceEngine):
    def __init__(self):
        self.classes = []
        self.core = ov.Core()
        self.scale: float
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None

        self.input_width:int
        self.input_height:int
        self.img_width:int
        self.img_height:int


    def load_model(self, model_path):
        self.model = self.core.read_model(model_path)
        self.ppp = ov.preprocess.PrePostProcessor(self.model)
#        self.ppp.input().tenser().set_element_type(ov.Type.u8)
        self.ppp.input().model().set_layout(ov.Layout('NCHW'))
        self.ppp.output().tensor().set_element_type(ov.Type.f64)  # 这里设置成f64反而  fps更高  ，不知道为什么
        self.model = self.ppp.build()
        self.compiled_model = self.core.compile_model(self.model, "CPU")


        # Get input and output nodes
        self.input_layer = self.model.input(0)



        self.output_layer = self.compiled_model.output(0)


    def load_class_file(self, class_path):
        self.classes = class_file(class_path)

        return self.classes


    def preprocess(self, image):
        # Resize image and keep aspect ratio
        row, col, _ = image.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = image
        img_h, img_w, img_c = result.shape

        self.scale = _max / 640  # W

        # Convert to blob
        blob = cv2.dnn.blobFromImage(result, 1 / 255.0, (640, 640), swapRB=True,crop = False)

        return blob

    def infer(self, preprocessed_image):

        results = self.compiled_model([preprocessed_image])[self.output_layer]

        outputs = np.squeeze(results, 0)  #删除掉第0维

        return outputs

    def postprocess(self, model:ModelType, outputs, conf, nms, score, detections):


        # 拉伸 还是 填充好  ，若拉伸好  将考虑此处的解决方法  得到x y的缩放因子  ，传入后处理函数  ，或者  留个接口让用户选择 填充还是拉伸
        # # Calculate the scaling factors for the bounding box coordinates
        # x_factor = self.img_width / self.input_width
        # y_factor = self.img_height / self.input_height


        print("self.input_layeropenvino",self.input_layer.shape[3])  #640
        if model == ModelType.YOLOv8:
            return postprocessyolov8(outputs, conf, nms, score, self.scale, detections)

        if model == ModelType.YOLOv5:
            return postprocessyolov5(outputs, conf, nms, score, self.scale, detections)

