import numpy as np

from models.decision_tree import RegressionTree
from loss_function import CrossEntropy

# thêm categorical để chuyển đổi nhãn thành one-hot encoding, do code của cả file gần như trốn trơn:((((
def to_categorical(y):
    if 
class GradientBoostingClassifier:
    def __init__(self, n_estimators=200, learning_rate=0.1, min_samples_split=2, min_info_gain=1e-7, max_depth=2):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_info_gain
        self.max_depth = max_depth

        self.loss = CrossEntropy()# Loss function for classification

        # Initialize trees
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth
            )
            self.trees.append(tree)

    def softmax(self, z): # Chuyển đổi logits thành xác suất vs sẽ dùng nhiều nên tách riêng ra tránh lặp
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        y = to_categorical(y)

        self.init_pred = np.mean(y, axis=0)
        y_pred = np.tile(self.init_pred, (X.shape[0], 1))

        self.train_loss = []    #Lưu trữ loss trong quá trình huấn luyện để nếu cần vẽ biểu đồ sau này
        for i in self.bar(range(self.n_estimators)):
            probs = self._softmax(y_pred)

            gradient = probs - y

            self.trees[i].fit(X, gradient)

            update = self.trees[i].predict(X)

            y_pred -= self.learning_rate * update

            loss = -np.sum(y * np.log(probs + 1e-15)) / X.shape[0]
            self.train_loss.append(loss)
    #Bổ sung thêm hàm predict_proba để trả về xác suất dự đoán, và sửa lại hàm predict để sử dụng xác suất này
    def predict_proba(self, X):
        y_pred = np.tile(self.init_pred, (X.shape[0], 1))

        for tree in self.trees:
            y_pred -= self.learning_rate * tree.predict(X)

        return self._softmax(y_pred)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    # def predict(self, X): 
    #     y_pred = np.tile(self.init_pred, (X.shape[0], 1)) 
    #     for tree in self.trees: 
    #         y_pred += self.learning_rate * tree.predict(X) 
    #         exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True)) 
    #         probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
