

# 用来保存检测结果的一个类 ，目前满足要求
class FrameDetections:
    def __init__(self):
        # 初始化时清空数据
        self.boxs = []
        self.scores = []
        self.classid = []

    def add_detection(self, box, score, class_id):
        # 添加一个检测结果
        self.boxs.append(box)
        self.scores.append(score)
        self.classid.append(class_id)

    def clear(self):
        # 清空检测结果
        self.boxs.clear()
        self.scores.clear()
        self.classid.clear()

    def get_detections(self):
        # 返回检测结果（如果需要的话）
        return self.boxs, self.scores, self.classid

    # 使用示例


# detections = FrameDetections()
#
# # 假设你有一帧的检测结果
# box1 = [10, 20, 30, 40]  # 示例边界框，通常是(x, y, width, height)
# score1 = 0.95
# class_id1 = 1
# detections.add_detection(box1, score1, class_id1)
#
# # 当你处理完这一帧并想要获取检测结果时
# boxes, scores, class_ids = detections.get_detections()
# print(boxes, scores, class_ids)
#
# # 在处理下一帧之前，清空检测结果
# detections.clear()
#
# # 现在detections对象中的boxs, scores, classid都是空的