import numpy as np
from .confusion_matrix import confusion_matrix

def classification_report(y_true, y_pred):
    """
    Tính các metric:
    - Precision
    - Recall
    - F1-score
    - Support

    Trả về dạng dictionary giống sklearn
    """
    # lấy confusion matrix
    cm, labels = confusion_matrix(y_true, y_pred)

    report = {}

    # duyệt từng class
    for i, label in enumerate(labels):

        # True Positive
        TP = cm[i, i]

        # False Positive
        FP = np.sum(cm[:, i]) - TP

        # False Negative
        FN = np.sum(cm[i, :]) - TP

        # số mẫu thật của class
        support = np.sum(cm[i, :])

        # Precision = TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0

        # Recall = TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        # F1-score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # lưu vào report
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }

    return report