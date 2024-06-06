### 这是使用一套API来进行 深度学习目标检测模型推理的开源库

你可以通过使用本库来方便的在OpenVINO、ONNXRuntime、OpenCVdnn、TenserRT这四种加速引擎之间进行方便的切换，仅仅需要改变EngineType这个枚举类来选择相应的推理引擎，而不用去单独学习和调用前面提到的四种加速引擎,目前开发了opencvdnn 、openvino 的yolov5和v8 推理代码以及 onnxruntime的yolov8版本推理代码

> tips

 TensorRT：英伟达的，用于GPU推理加速。注意需要英伟达GPU硬件的支持。

OpenVino：英特尔的，用于CPU推理加速。注意需要英特尔CPU硬件的支持。

ONNXRuntime：微软，亚马逊 ，Facebook 和 IBM 等公司共同开发的，可用于GPU、CPU

OpenCV dnn：OpenCV的调用模型的模块

无论用什么框架训练的模型，推荐转为onnx格式，方便部署。

### 加速的效果理论上如下所示


推理效率上：TensorRT>OpenVino>ONNXRuntime>OpenCV dnn>Pytorch

由于电脑只有CPU目前只开发了OpenVino、ONNXRuntime、OpenCV dnn的python API推理接口。
理论上C++版本的速度是python版本的2-3倍。后续有机会会写c++版本的 API