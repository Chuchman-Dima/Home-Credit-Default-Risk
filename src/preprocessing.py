"""
Модуль для попередньої обробки даних (Data Preprocessing).
Створює пайплайн (Pipeline) для очищення, заповнення пропусків
та кодування ознак перед подачею їх у модель машинного навчання.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Колонки, які не несуть користі
DROP_COLS = [
    "sk_id_curr", "sk_id_bureau", "sk_id_prev",
    "index",
]


def get_feature_types(df: pd.DataFrame, target: str = "target"):
    """
    Автоматично розділяє всі колонки датафрейму на числові та категоріальні
    на основі їхнього типу даних (dtypes). Це необхідно для їхньої роздільної обробки.
    """
    cols = [c for c in df.columns if c not in DROP_COLS + [target]]
    num_cols = df[cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df[cols].select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Створює sklearn ColumnTransformer, який об'єднує два різних конвеєри (pipelines)
    для обробки числових та категоріальних ознак.
    """

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
        Головна функція підготовки даних.
        Очищає датасет, обробляє аномалії, розділяє його на X (ознаки) та y (ціль)
        і генерує готовий до тренування об'єкт препроцесора.
    """

    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

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