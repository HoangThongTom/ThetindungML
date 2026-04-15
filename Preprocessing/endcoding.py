import numpy as np
import pandas as pd


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Label-encode tất cả cột categorical dựa trên mapping từ tập train,
    sau đó trả về numpy array.

    Parameters
    ----------
    X_train : DataFrame tập train
    X_test  : DataFrame tập test

    Returns
    -------
    X_train_enc, X_test_enc : numpy array đã encode
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        # Mapping chỉ học từ train để tránh data leakage
        categories = X_train[col].unique()
        mapping    = {v: i for i, v in enumerate(categories)}

        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(int)
        # Giá trị chưa thấy trong train -> -1
        X_test[col]  = X_test[col].map(mapping).fillna(-1).astype(int)

    return X_train.to_numpy().astype(float), X_test.to_numpy().astype(float)
