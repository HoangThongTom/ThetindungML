import numpy as np
from GradientBoostingClassifier.models.decision_tree import RegressionTree


def to_categorical(y):
    """Chuyển nhãn 0/1 thành one-hot encoding (n_samples, n_classes)."""
    y = np.array(y, dtype=int)
    n_classes = len(np.unique(y))
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


class GradientBoostingClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1,
                 min_samples_split=2, min_info_gain=1e-7, max_depth=3):

        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity      = min_info_gain
        self.max_depth         = max_depth

        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth
            )
            self.trees.append(tree)

    def _softmax(self, z):
        """Chuyển đổi logits thành xác suất."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _loss_function(self, y, probs):
        """Tính Cross-Entropy Loss"""
        return -np.sum(y * np.log(probs + 1e-15)) / y.shape[0]

    def _loss_gradient(self, y, probs):
        """Tính Gradient của hàm Loss"""
        return probs - y

    def fit(self, X, y):
        y = to_categorical(y)

        self.init_pred = np.mean(y, axis=0)
        y_pred = np.tile(self.init_pred, (X.shape[0], 1))

        self.train_loss = []
        for i in range(self.n_estimators):
            probs    = self._softmax(y_pred)
            gradient = self._loss_gradient(y, probs)

            self.trees[i].fit(X, gradient)

            update = np.array(self.trees[i].predict(X))
            if update.ndim == 1:
                update = np.expand_dims(update, axis=1)

            y_pred -= self.learning_rate * update

            loss = self._loss_function(y, probs)
            self.train_loss.append(loss)

            if (i + 1) % 10 == 0:
                print(f"  Estimator {i+1:>3}/{self.n_estimators}  —  loss: {loss:.4f}")

    def predict_proba(self, X):
        """Trả về xác suất dự đoán cho từng class."""
        y_pred = np.tile(self.init_pred, (X.shape[0], 1))
        for tree in self.trees:
            update = np.array(tree.predict(X))
            if update.ndim == 1:
                update = np.expand_dims(update, axis=1)
            y_pred -= self.learning_rate * update
        return self._softmax(y_pred)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
