from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PARQUET = ROOT / "data" / "parquet"
DATA_PROCESSED = ROOT / "data" / "processed"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
FIGURES_DIR = ARTIFACTS / "figures"

TARGET = "target"
ID_COL = "sk_id_curr"
RANDOM_STATE = 42
DAYS_EMPLOYED_ANOMALY = 365243

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments_payments",
    "credit_card_balance",
]

CSV_ALIASES = {
    "pos_cash_balance": ["POS_CASH_balance.csv", "pos_cash_balance.csv"],
}

LGBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
