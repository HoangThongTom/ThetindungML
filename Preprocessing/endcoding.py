import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_features(X_train, X_test):
    X_train = X_train.copy()
    X_test  = X_test.copy()

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        # Fit chỉ trên train để tránh data leakage
        le.fit(X_train[col])

        X_train[col] = le.transform(X_train[col])
        # Giá trị chưa thấy trong train -> -1
        X_test[col] = X_test[col].apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    return X_train.to_numpy().astype(float), X_test.to_numpy().astype(float)
