import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split


def preprocess(df: pd.DataFrame, target_col: str = "class"):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Chuẩn hoá nhãn: 'good' -> 1, 'bad' -> 0
    unique_vals = df[target_col].unique()
    if any(isinstance(v, str) for v in unique_vals):
        df[target_col] = df[target_col].map({"good": 1, "bad": 0})

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return X, y


def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)
