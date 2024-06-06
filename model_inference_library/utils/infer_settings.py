
# 部分超参数的设计
class InferSettings:

    def __init__(self):

        self._score_threshold = 0.25
        self._nms_threshold = 0.25
        self._conf_threshold = 0.25

    def score_threshold_get(self):
        return self._score_threshold

    def nms_threshold_get(self):
        return self._nms_threshold

    def conf_threshold_get(self):
        return self._conf_threshold

    def score_threshold_set(self, value):
        self._score_threshold = value

    def nms_threshold_set(self, value):
        self._nms_threshold = value

    def conf_threshold_set(self, value):
        self._conf_threshold = value