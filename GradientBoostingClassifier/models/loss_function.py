import numpy as np
from metrics.accuracy import accuracy_score

# Lớp cơ sở cho hàm Loss
class Loss(object):
    def loss(self, y, p):
        raise NotImplementedError()

    def gradient(self, y, p):
        raise NotImplementedError()

    def acc(self, y, p):
        return 0

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        # Tránh lỗi chia cho 0 hoặc log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        # Tính toán độ chính xác cho phân loại
        # y là one-hot, p là xác suất
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)