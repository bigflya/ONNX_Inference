import cv2
import numpy as np


#  画框函数 ，可以根据自己的要求选择是否需要框 或者改成其他形式的图标来显示


def draw_bounding_box(curr_engine,img, class_id, confidence, box):# (curr_engine,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        box:
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """

    x = round(box[0])
    y = round(box[1])
    x_plus_w = round(box[0] + box[2])
    y_plus_h = round(box[1] + box[3])


    # 用id 来得到 类别名
    #curr_engine.classes   得到的是类别文件列表  ,将id 传入其中可以得到  类别名

    # # 获取某个类别的阈值
    #get_category = curr_engine.detection_categories_seting.categories[curr_engine.classes[class_id]]#  传进来的是id   要从id得到name

    #print(f"Person NMS Threshold: {get_category.nms_thresh}")

    CLASSES = curr_engine.classes
    # print(curr_engine.classes)
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    label = f"{CLASSES[class_id]} ({confidence:.2f})"

    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)