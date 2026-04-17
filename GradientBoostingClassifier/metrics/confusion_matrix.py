import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):
    """
    Tạo confusion matrix

    Parameters
    ----------
    y_true : nhãn thật
    y_pred : nhãn dự đoán

    Returns
    -------
    matrix : ma trận confusion matrix
    labels : danh sách label
    """

    # chuyển sang numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # lấy tất cả label xuất hiện
    labels = np.unique(np.concatenate((y_true, y_pred)))

    # mapping label -> index
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # tạo ma trận rỗng
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # đếm từng trường hợp
    for t, p in zip(y_true, y_pred):
        i = label_to_index[t]
        j = label_to_index[p]
        matrix[i, j] += 1

    return matrix, labels

def plot_confusion_matrix(cm, labels):
    """
    Vẽ confusion matrix bằng matplotlib
    """
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    # label trục X và Y
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # ghi số vào từng ô
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.show()