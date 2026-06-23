"""
train.py — Тренування LightGBM моделі.
Run: python -m src.train
"""

import os, sys, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_features
from src.preprocessing import prepare_data

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "plots"), exist_ok=True)


def build_pipeline(preprocessor) -> Pipeline:
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
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model(X, y, preprocessor):
    pipeline = build_pipeline(preprocessor)

    print("\nRunning 5-fold CV...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  per fold : {[f'{s:.4f}' for s in scores]}")
    print(f"  mean±std : {scores.mean():.4f} ± {scores.std():.4f}")

    # Тест-сет для метрик та оптимального порогу
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)
    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_te)[:, 1]

    test_auc = roc_auc_score(y_te, y_prob)
    test_ap  = average_precision_score(y_te, y_prob)
    print(f"  test AUC : {test_auc:.4f}")
    print(f"  test AP  : {test_ap:.4f}")

    # Оптимальний поріг (max F1)
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.05, 0.90, 0.01)
    f1s        = [f1_score(y_te, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    opt_thr    = float(thresholds[np.argmax(f1s)])
    print(f"  opt thr  : {opt_thr:.2f} (max F1={max(f1s):.4f})")

    # Фінальна модель на всіх даних
    print("\nFitting final model on full data...")
    pipeline_final = build_pipeline(preprocessor)
    pipeline_final.fit(X, y)

    return pipeline_final, scores, test_auc, test_ap, opt_thr


def save_model(pipeline, scores, test_auc, test_ap, opt_thr):
    # Model
    model_path = os.path.join(MODELS_DIR, "credit_scoring_lgbm.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nМодель: {model_path}")

    # Feature importance
    prep          = pipeline.named_steps["preprocessor"]
    lgbm          = pipeline.named_steps["model"]
    feature_names = list(prep.get_feature_names_out())

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": lgbm.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
    print(f"Feature importance: {len(feature_names)} ознак")

    # Metadata  ← feature_names зберігаємо для Streamlit predict_single
    meta = {
        "model_type":        "LightGBMClassifier",
        "cv_auc_mean":       float(scores.mean()),
        "cv_auc_std":        float(scores.std()),
        "test_auc":          float(test_auc),
        "test_avg_precision":float(test_ap),
        "optimal_threshold": opt_thr,
        "n_features":        len(feature_names),
        "feature_names":     feature_names,          # ← ключове для app.py
        "top_features":      imp_df.head(15)["feature"].tolist(),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata: {MODELS_DIR}/model_metadata.json")

    print(f"\nТоп-15 ознак:")
    print(imp_df.head(15).to_string(index=False))
    return model_path


def main():
    print("=" * 55)
    print("  Home Credit — Тренування моделі")
    print("=" * 55)

    print("\n[1/3] Feature engineering (з parquet)...")
    df = build_features(use_local=True)

    print("\n[2/3] Preprocessing...")
    X, y, num_cols, cat_cols, preprocessor = prepare_data(df)
    print(f"  Рядків: {X.shape[0]:,}  |  Ознак: {X.shape[1]}")
    print(f"  Дефолт rate: {y.mean():.2%}")

    print("\n[3/3] Тренування...")
    pipeline, scores, test_auc, test_ap, opt_thr = train_model(X, y, preprocessor)
    save_model(pipeline, scores, test_auc, test_ap, opt_thr)

    print("\nГотово! Запустіть: streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()