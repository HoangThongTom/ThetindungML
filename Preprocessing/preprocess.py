import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame, target_col: str = "class"):
    """
    Tách features và label, chuẩn hoá nhãn.

    Parameters
    ----------
    df         : DataFrame gốc
    target_col : tên cột nhãn

    Returns
    -------
    X : DataFrame features
    y : Series nhãn (0 = bad, 1 = good)
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Chuẩn hoá nhãn dạng string -> số
    unique_vals = df[target_col].unique()
    if any(isinstance(v, str) for v in unique_vals):
        df[target_col] = df[target_col].map({"good": 1, "bad": 0})

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return X, y


def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Chia dữ liệu thành train và test.

    Parameters
    ----------
    X           : DataFrame features
    y           : Series nhãn
    test_size   : tỉ lệ tập test (mặc định 0.2)
    random_state: seed ngẫu nhiên

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n       = len(X)
    indices = np.random.permutation(n)
    n_test  = int(n * test_size)

    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return (X.iloc[train_idx].reset_index(drop=True),
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True))
