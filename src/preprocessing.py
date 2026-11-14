import pandas as pd
from sklearn.model_selection import train_test_split


def load_aps_csv(path):
    data = pd.read_csv(path, skiprows=20, na_values="na")

    y = data["class"].map({"neg": 0, "pos": 1})
    if y.isna().any():
        raise ValueError("Expected every class label to be 'neg' or 'pos'.")

    X = data.drop(columns="class")
    return X, y.astype(int)


def split_development_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )