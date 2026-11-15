import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def make_preprocessor(*, scale=False, add_indicators=False):
    """Build preprocessing that learns from the data passed to fit."""
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=add_indicators,
                keep_empty_features=True,
            ),
        ),
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        (
            "scaler",
            StandardScaler() if scale else "passthrough",
        ),
    ])