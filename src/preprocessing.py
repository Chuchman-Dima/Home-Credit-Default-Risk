"""
preprocessing.py - Data preprocessing pipeline for credit scoring.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# Columns to drop (IDs, target, leakage)
DROP_COLS = [
    "sk_id_curr", "sk_id_bureau", "sk_id_prev",
    "index",
]


def get_feature_types(df: pd.DataFrame, target: str = "target"):
    """Split columns into numeric and categorical."""
    cols = [c for c in df.columns if c not in DROP_COLS + [target]]
    num_cols = df[cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df[cols].select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_data(df: pd.DataFrame, target: str = "target"):
    """
    Split DataFrame into X, y and build preprocessor.

    Returns:
        X_raw       - raw feature DataFrame
        y           - target Series
        num_cols    - list of numeric column names
        cat_cols    - list of categorical column names
        preprocessor- fitted-ready ColumnTransformer
    """
    # Drop useless columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # Replace XNA with NaN (Home Credit specific)
    df = df.replace("XNA", np.nan)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    num_cols, cat_cols = get_feature_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, num_cols, cat_cols, preprocessor


if __name__ == "__main__":
    from src.features import build_features
    df = build_features()
    X, y, num_cols, cat_cols, preprocessor = prepare_data(df)
    print(f"X shape: {X.shape}")