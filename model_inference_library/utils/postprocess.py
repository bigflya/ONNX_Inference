import cv2
import numpy as np

# 用来对推理后的模型进行后处理，目前做了v8和v5的后处理


# yolov8 的后处理函数
def postprocessyolov8(outputs, conf, nms, score,scale, detections):

    # Transpose and squeeze the output to match the expected shape
    outputs = np.array([cv2.transpose(outputs)])

    detections.clear()

    rows = outputs.shape[1]  # 8400
    #  outputs  1x8400x6
    for i in range(rows):


        row = outputs[0][i]
        # confidence = row[4]
        classes_scores = outputs[0][i][4:]  # 得到每个类别的分数

        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        # confidence = maxScore  # 由于v8和v5的输出不一样 ，v5有个整体置信度的概念 在x y w h 后面紧跟着，而v8没有所以v8 其实置信度的概念可有可无，
        if maxScore >= conf:  # 先要置信度满足要求, v8没有单独的整体置信度概念 这里用 最大的分数代替，如果某个框对应的最大分数，大于置信度阈值才会考虑此框的预测类别

            if (classes_scores[maxClassIndex] > score):  # 如果 某个预测框  ，对应所有类别中最大的置信度小于阈值那么此框作废

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                box = [
                    int((x - 0.5 * w) * scale),
                    int((y - 0.5 * h) * scale),
                    int(w * scale),
                    int(h * scale),
                ]
                detections.add_detection(box, maxScore, maxClassIndex)


    boxes, scores, class_ids = detections.get_detections()
    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf,
                                    nms, 0.5) # 返回值是筛选后的框对应的  index

    return result_boxes


# yolov5 的后处理函数
def postprocessyolov5(outputs, conf, nms, score,scale, detections):
    detections.clear()


    for i in range(outputs.shape[0]):
        row = outputs[i]
        confidence = row[4]
        if confidence >= conf:

            classes_scores = row[5:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

            if (classes_scores[maxClassIndex] > score):

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                box = [
                    int((x - 0.5 * w) * scale),
                    int((y - 0.5 * h) * scale),
                    int(w * scale),
                    int(h * scale),
                ]
                detections.add_detection(box, maxScore, maxClassIndex)


    boxes, scores, class_ids = detections.get_detections()
    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf,
                                    nms, 0.5)
    return result_boxes