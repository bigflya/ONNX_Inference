from abc import ABC, abstractmethod
from model_inference_library.utils.framedetections import FrameDetections
from model_inference_library.utils.modeltype import ModelType
class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def infer(self, preprocessed_image):
        pass

    @abstractmethod
    def postprocess(self,model: ModelType, results, conf: float, nms: float, score: float, detections: FrameDetections):
        pass

    def predict(self, model: ModelType, image, conf: float, nms: float, score: float, detections: FrameDetections):
        preprocessed_image = self.preprocess(image)
        outputs = self.infer(preprocessed_image)
        result_boxes = self.postprocess(model, outputs, conf, nms, score, detections)
        return result_boxes


