# 🏦 Home Credit Default Risk

Система кредитного скорингу на основі LightGBM з повним ML-циклом: EDA → Feature Engineering → Training → Streamlit Deploy.

## 📁 Структура проєкту

```
Home-Credit-Default-Risk/
│
├── data/
│   └── home-credit-default-risk/     # CSV файли (не в git)
│
├── models/
│   ├── credit_scoring_lgbm.pkl       # Збережена модель (після тренування)
│   ├── feature_importance.csv        # Важливість ознак
│   ├── model_metadata.json           # Метрики та метадані
│   └── plots/                        # Графіки (генеруються ноутбуком)
│
├── notebooks/
│   ├── EDA.ipynb                     # Розвідувальний аналіз
│   └── model.ipynb                   # Тренування та оцінка моделі
│
├── sql/
│   └── DBeaver-Creating/             # SQL скрипти створення таблиць
│
├── src/
│   ├── __init__.py
│   ├── db.py                         # PostgreSQL підключення
│   ├── features.py                   # Feature engineering (7 таблиць)
│   ├── preprocessing.py              # sklearn Pipeline
│   └── train.py                      # Скрипт тренування (CLI)
│
├── streamlit_app/
│   └── app.py                        # 🚀 Streamlit застосунок
│
├── .env                              # Конфігурація БД
├── check_connection.py               # Тест підключення до БД
├── init_db.py                        # Імпорт CSV → PostgreSQL
├── requirements.txt
└── README.md
```

## 🚀 Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 2. Налаштування `.env`

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=data default
DB_USER=postgres
DB_PASSWORD=1111
```

### 3. Перевірка підключення до БД

```bash
python check_connection.py
```

### 4. Імпорт даних (якщо CSV ще не в БД)

```bash
python init_db.py
```

### 5. Розвідувальний аналіз

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 6. Тренування моделі

**Через ноутбук (рекомендовано):**
```bash
jupyter notebook notebooks/model.ipynb
```

**Через скрипт:**
```bash
python -m src.train
```

### 7. Запуск Streamlit

```bash
streamlit run streamlit_app/app.py
```

Відкрийте [http://localhost:8501](http://localhost:8501)

---

## 📊 ML Pipeline

```
PostgreSQL (7 таблиць)
         │
         ▼
  Feature Engineering
  ─────────────────────
  • bureau            → 11 агрегованих ознак
  • bureau_balance    →  4 ознаки
  • previous_app      → 11 ознак
  • pos_cash          →  8 ознак
  • installments      →  8 ознак
  • credit_card       → 10 ознак
         │ ~80 ознак
         ▼
    Preprocessing
  ─────────────────────
  • SimpleImputer (median / most_frequent)
  • OrdinalEncoder (категоріальні)
  • ColumnTransformer
         │
         ▼
  LightGBM Classifier
  ─────────────────────
  • n_estimators=1000, lr=0.05
  • num_leaves=31
  • class_weight="balanced"
  • subsample=0.8, colsample=0.8
         │
         ▼
  5-fold StratifiedKFold CV
         │
         ▼
  ROC-AUC ~0.76+
```

## 🤖 Модель

| Параметр | Значення |
|---------|---------|
| Алгоритм | LightGBM Classifier |
| Метрика | ROC-AUC |
| CV | 5-fold StratifiedKFold |
| Дисбаланс | `class_weight="balanced"` |
| Ознак | ~80 (з feature engineering) |

## 🌐 Streamlit Features

| Сторінка | Функціонал |
|---------|-----------|
| 🔮 Скоринг клієнта | Ручне введення параметрів → ймовірність дефолту, кредитний скор 300-850, рекомендація |
| 📊 Аналіз портфеля | Завантаження CSV → пакетний скоринг → розподіл ризиків, графіки |
| 📈 Метрики моделі | ROC-AUC, PR-крива, confusion matrix, feature importance, SHAP |
| ℹ️ Про модель | Документація, архітектура, інструкції |

## 📦 Залежності

```
lightgbm, scikit-learn, pandas, numpy
sqlalchemy, pg8000
streamlit, plotly
shap, matplotlib, seaborn
joblib, optuna
```

## 📈 Результати

- **CV ROC-AUC:** ~0.76-0.78
- **Топ ознаки:** EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1, DAYS_BIRTH, AMT_CREDIT
- **Дефолт rate:** ~8% (сильний дисбаланс)
