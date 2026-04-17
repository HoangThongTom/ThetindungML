import numpy as np
from .confusion_matrix import confusion_matrix

def classification_report(y_true, y_pred, target_names=None):
    """
    Tính các metric:
    - Precision
    - Recall
    - F1-score
    - Support

    Trả về dạng dictionary giống sklearn
    """
    # confusion matrix + labels
    cm, labels = confusion_matrix(y_true, y_pred)

    report = {}

    # nếu có target_names thì dùng thay cho labels
    if target_names is not None:
        if len(target_names) != len(labels):
            raise ValueError("target_names phải có cùng độ dài với số class")
        display_labels = target_names
    else:
        display_labels = labels

    # duyệt từng class
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        support = np.sum(cm[i, :])

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # dùng tên hiển thị giống sklearn
        name = display_labels[i]

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }

    return report

def print_classification_report(report):
    """
    In report theo format giống sklearn
    """

    # header
    print(f"{'':<15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print()

    # tách class và các dòng summary (nếu có)
    for label, metrics in report.items():
        precision = metrics.get("precision", "")
        recall = metrics.get("recall", "")
        f1 = metrics.get("f1-score", "")
        support = metrics.get("support", "")

        # format giống sklearn: nếu là accuracy (chỉ có 1 giá trị)
        if label == "accuracy":
            print(f"{label:<15} {'':>10} {'':>10} {metrics:>10.2f} {support:>10}")
        else:
            print(f"{str(label):<15} "
                  f"{precision:>10.2f} "
                  f"{recall:>10.2f} "
                  f"{f1:>10.2f} "
                  f"{support:>10}")