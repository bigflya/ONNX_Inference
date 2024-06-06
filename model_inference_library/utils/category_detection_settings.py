from dataclasses import dataclass

#  想通过 主函数的配置 实现不同 目标 的不同 nms   conf  和  socre  目前暂未实现全部

@dataclass
class DetectionCategory:
    name: str
    nms_thresh: float
    conf_thresh: float


class DetectionCategories:
    def __init__(self):
        self.categories = {}

    def add_category(self, name: str, nms_thresh: float, conf_thresh: float):
        if name in self.categories:
            raise ValueError(f"Category '{name}' already exists.")
        self.categories[name] = DetectionCategory(name, nms_thresh, conf_thresh)

    def get_category(self, name: str) -> DetectionCategory:
        if name not in self.categories:
            raise ValueError(f"Category '{name}' does not exist.")
        return self.categories[name]

    def __getitem__(self, name: str) -> DetectionCategory:
        return self.get_category(name)

    # 示例用法


# detection_categories = DetectionCategories()
# detection_categories.add_category("person", nms_thresh=0.5, conf_thresh=0.7)
# detection_categories.add_category("car", nms_thresh=0.4, conf_thresh=0.6)
#
# # 获取某个类别的阈值
# person_category = detection_categories["person"]
# print(f"Person NMS Threshold: {person_category.nms_thresh}")
# print(f"Person Confidence Threshold: {person_category.conf_thresh}")