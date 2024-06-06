from enum import Enum

#  模型推理引擎的枚举类 目前只写了三种，后续可能会考虑 tenserrt
class EngineType(Enum):
    OPENCV_DNN = "opencv_dnn"
    OPENVINO = "openvino"
    ONNXRUNTIME = "onnxruntime"
    # TENSORRT = "tensorrt"

