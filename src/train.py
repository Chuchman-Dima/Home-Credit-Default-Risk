"""
train.py - Train LightGBM credit scoring model.
Run: python -m src.train
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_features
from src.preprocessing import prepare_data

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "plots"), exist_ok=True)


def train_model(X, y, preprocessor):
    model = lgb.LGBMClassifier(
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 31,
        max_depth         = -1,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  ROC-AUC per fold: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\nFitting final model on full training data...")
    pipeline.fit(X, y)
    return pipeline, scores


def save_model(pipeline, scores):
    path = os.path.join(MODELS_DIR, "credit_scoring_lgbm.pkl")
    joblib.dump(pipeline, path)
    print(f"\n✅ Модель збережена: {path}")

    model        = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    importance_df.to_csv(imp_path, index=False)
    print(f"✅ Feature importance: {imp_path}")

    # Metadata
    meta = {
        "model_type":          "LightGBMClassifier",
        "cv_auc_mean":         float(scores.mean()),
        "cv_auc_std":          float(scores.std()),
        "n_features":          int(len(feature_names)),
        "n_train_samples":     int(len(scores) * 5),   # approx
        "feature_names":       feature_names.tolist(),
        "top_features":        importance_df.head(15)["feature"].tolist(),
        "optimal_threshold":   0.5,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nТоп-15 ознак:")
    print(importance_df.head(15).to_string(index=False))
    return path


def main():
    print("=" * 55)
    print("  Home Credit - Model Training")
    print("=" * 55)

    print("\n[1/3] Building feature matrix (з parquet кешу)...")
    df = build_features(use_local=True)   # ← parquet

    print("\n[2/3] Preparing data...")
    X, y, num_cols, cat_cols, preprocessor = prepare_data(df)
    print(f"  Training set: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"  Default rate: {y.mean():.2%}")

    print("\n[3/3] Training model...")
    pipeline, scores = train_model(X, y, preprocessor)
    save_model(pipeline, scores)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()