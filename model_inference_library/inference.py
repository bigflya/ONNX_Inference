from model_inference_library.engines import OpenCVDNNEngine, ONNXRuntimeEngine, OpenVINOEngine
from model_inference_library.utils.enginetype import EngineType
from model_inference_library.utils.modeltype import ModelType
from model_inference_library.engines.base import InferenceEngine
from model_inference_library.utils.category_detection_settings import DetectionCategories
from model_inference_library.utils.draw_bbox import draw_bounding_box
from model_inference_library.utils.infer_settings import InferSettings
from model_inference_library.utils.framedetections import FrameDetections


#  这个类是我的推理引擎接口，用户可选择调用哪个推理方式 ，设定 每个类别的置信度  ，以及其他参数

class Inference:

    def __init__(self, model_path, class_path, engine_type, model_type:ModelType):

        self.infer_settings = InferSettings()
        self.model_path = model_path
        self.class_path = class_path

        self.scale: float  # yolo 输入要进行缩放
        self.conf_threshold:float
        self.nms_threshold:float
        self.score_threshold:float

        self.classes =[]

        # self.set_engine_type = self.EngineType.OPENCV_DNN  # 默认是opencv  dnn

        self.detection_categories_seting = DetectionCategories()  # 设定 每个类别的非极大值抑制  和置信度
        self.detections = FrameDetections()
        self.engine:InferenceEngine()
        self.engine = self._initialize_engine(engine_type)
        self._load_modelandclass()
        self.model = model_type

    def SettingParameter(self,parameter:InferSettings):

        self.conf_threshold = parameter.conf_threshold_get()
        self.nms_threshold = parameter.nms_threshold_get()
        self.score_threshold = parameter.score_threshold_get()

        #print("self.conf_threshold",  self.conf_threshold)
        #self._initialize_engine(parameter.enginetype_get())
        #print("parameter.enginetype_get()", parameter.enginetype_get())

        #.........



    def detection_result(self):
        return self.detections

    def detection_parameter(self):
        return self.infer_settings

    @property
    def set_parameter(self):
        return self.infer_settings


    def predict(self, image):
        return self.engine.predict(self.model, image,self.conf_threshold,self.nms_threshold,self.score_threshold, self.detections)

    def _initialize_engine(self, engine_type):
        if engine_type == EngineType.OPENCV_DNN:
            return OpenCVDNNEngine()
        elif engine_type == EngineType.ONNXRUNTIME:
            return ONNXRuntimeEngine()
        elif engine_type == EngineType.OPENVINO:
            return OpenVINOEngine()
        # elif engine_type == EngineType.TENSORRT:
        #     return TensorRTEngine()
        else:
            raise ValueError("Unsupported engine type")

    def _load_modelandclass(self):
        self.engine.load_model(self.model_path)
        self.classes = self.engine.load_class_file(self.class_path)  # 将文件返回

    def draw_bbox(self, img, class_id, confidence,box):

        draw_bounding_box(self.engine, img, class_id, confidence,box) # 传入正在使用的引擎 已经  设定有nmx  和 conf的 类别对象