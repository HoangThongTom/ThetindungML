import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Tính độ chính xác (Accuracy)
    
    Accuracy = số dự đoán đúng / tổng số mẫu
    Parameters:
    y_true : list hoặc array
        Nhãn thật

    y_pred : list hoặc array
        Nhãn dự đoán
    Returns
    -------
    float
        giá trị accuracy từ 0 -> 1
    """
    # chuyển về numpy array để xử lý dễ hơn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # đếm số dự đoán đúng
    correct = np.sum(y_true == y_pred)
    # tổng số mẫu
    total = len(y_true)
    # tính accuracy
    return correct / total
