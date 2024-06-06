from model_inference_library.inference import Inference
from model_inference_library.utils.enginetype import EngineType
from model_inference_library.utils.modeltype import ModelType
import cv2



def mainvideo():

    model_path = "../file/v5reactor.onnx"  # 设置模型路径
    class_path = "../file/class.txt"  # 设置标签路径
    voide_file = "../file/xx.mp4"

    # 调用推理引擎 ,传入 模型   类别    引擎     和模型类型
    inference = Inference(model_path, class_path, EngineType.ONNXRUNTIME, ModelType.YOLOv5)  # 调用推理引擎，并告诉调用的是哪个

    # print(inference.model.value)


    # 设定推理的一些参数
    settings = inference.infer_settings
    settings.conf_threshold_set(0.5)
    settings.nms_threshold_set(0.4)
    settings.score_threshold_set(0.6)
    inference.SettingParameter(settings)  # 实例化参数对象

    capture = cv2.VideoCapture(voide_file)

    while True:

        e1 = cv2.getTickCount()
        ret, image = capture.read()
        if ret is not True:
            break

        result_boxes = inference.predict(image)  # 返回值是经过模型推理后的输出 并通过非极大值抑制筛选出的检测卡索引，  检测的具体信息在inference.detections.get_detections()中
        boxes, scores, class_ids = inference.detections.get_detections()
        # print("result_boxes",result_boxes)
        #  绘制结果
        for i in range(len(result_boxes)):
            index = result_boxes[i]  # 里面存放的是真正过滤后的 bbox索引
            box = boxes[index]
            class_id = class_ids[index]
            score = scores[index]
            #  绘制推理结果
            inference.draw_bbox(image, class_id, score, box)

        e2 = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (e2 - e1)
        cv2.putText(image,  "FPS: %.2f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # Display the image with bounding boxes

        cv2.imshow("image", image)
        cv2.waitKey(1)



    cv2.destroyAllWindows()



def mainframe():


    model_path = "../file/best.onnx"  # 设置模型路径
    class_path = "../file/class.txt"  # 设置标签路径


    inference = Inference(model_path, class_path, EngineType.OPENCV_DNN)  # 调用推理引擎，并告诉调用的是哪个
    settings = inference.infer_settings
    # 设定推理的一些参数
    # Can be EngineType.OPENCV_DNN, EngineType.OPENVINO, EngineType.ONNXRUNTIME, EngineType.TENSORRT
    settings.enginetype_set(EngineType.OPENCV_DNN)
    settings.conf_threshold_set(0.1)
    settings.nms_threshold_set(0.9)
    settings.score_threshold_set(0.9)
    inference.SettingParameter(settings)

    # 想着针对某个类别设定阈值，暂时未实现
    # inference.detection_categories_seting.add_category("h", nms_thresh=0.5, conf_thresh=0.7)


    image = cv2.imread("../file/2.png")

    result_boxes = inference.predict(image)  # 返回值是经过模型推理后的输出 并通过非极大值抑制筛选出的检测卡索引，  检测的具体信息在inference.detections.get_detections()中
    boxes, scores, class_ids = inference.detections.get_detections()
    #         "scale": inference.engine.scale_x,
    #         "scale": inference.engine.scale_y,

    #  绘制结果
    for i in range(len(result_boxes)):
        index = result_boxes[i]  # 里面存放的是真正过滤后的 bbox索引
        box = boxes[index]
        class_id = class_ids[index]
        score = scores[index]
        #  绘制推理结果
        inference.draw_bbox(image,class_id,score,box)


    # Display the image with bounding boxes
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #print(results)

if __name__ == "__main__":
    mainvideo()
