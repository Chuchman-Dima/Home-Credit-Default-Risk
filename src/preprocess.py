"""Підготовка X/y і sklearn-препроцесор для LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import ID_COL, TARGET

DROP_COLS = [
    ID_COL,
    "sk_id_bureau",
    "sk_id_prev",
    "index",
]


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string", "boolean"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(df: pd.DataFrame, target: str = TARGET):
    data = df.copy()
    drop = [c for c in DROP_COLS if c in data.columns]
    data = data.drop(columns=drop)
    data = data.replace("XNA", np.nan)

    y = data[target].astype(int)
    X = data.drop(columns=[target])

    num_cols, cat_cols = get_feature_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    print(f"Ознаки: {len(num_cols)} числових, {len(cat_cols)} категоріальних")
    print(f"TARGET: {y.value_counts().to_dict()}  (default rate {y.mean():.2%})")
    return X, y, num_cols, cat_cols, preprocessor
