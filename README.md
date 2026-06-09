# 🏦 Home Credit Default Risk
# **[Посилання на застосунок](https://home-credit-default-risk-my.streamlit.app/)**

> Система кредитного скорингу на основі **LightGBM** з повним ML-циклом:
> EDA → Feature Engineering → Training → Streamlit Deploy.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://postgresql.org)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.76%2B-green)](/)

---

## 📋 Зміст
- [Про проєкт](#про-проєкт)
- [Структура](#структура)
- [Швидкий старт](#швидкий-старт)
- [ML Pipeline](#ml-pipeline)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Результати](#результати)

---

## Про проєкт

**Задача:** передбачити чи допустить клієнт дефолт за кредитом (бінарна класифікація).

**Дані:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — 7 взаємопов'язаних таблиць у PostgreSQL:

| Таблиця | Рядків | Опис |
|---------|-------:|------|
| `application_train` | 307 511 | Основні заявки на кредит |
| `bureau` | 1 716 428 | Кредитна історія з ЦБ |
| `bureau_balance` | 27 299 925 | Місячні баланси кредитів бюро |
| `previous_application` | 1 670 214 | Попередні заявки в Home Credit |
| `pos_cash_balance` | 10 001 358 | POS кредити та готівкові займи |
| `installments_payments` | 13 605 401 | Платіжна дисципліна |
| `credit_card_balance` | 3 840 312 | Баланси кредитних карток |

---

## Структура

```
Home-Credit-Default-Risk/
│
├── data/
│   ├── home-credit-default-risk/   # CSV файли (не в git, ~3.5 GB)
│   └── parquet/                    # Локальний кеш (після dump)
│
├── models/
│   ├── credit_scoring_lgbm.pkl     # Натренована модель
│   ├── feature_importance.csv      # Важливість ознак
│   ├── model_metadata.json         # Метрики + список ознак
│   └── plots/                      # Графіки (CV, ROC, SHAP...)
│
├── notebooks/
│   ├── EDA.ipynb                   # Розвідувальний аналіз (10 розділів)
│   └── model.ipynb                 # Тренування + SHAP (8 кроків)
│
├── src/
│   ├── __init__.py
│   ├── db.py                       # PostgreSQL + parquet кеш
│   ├── features.py                 # Feature engineering (7 таблиць → ~80 ознак)
│   ├── preprocessing.py            # sklearn Pipeline
│   └── train.py                    # CLI тренування
│
├── streamlit_app/
│   └── app.py                      # 🚀 4-сторінковий дашборд
│
├── dump_to_parquet.py              # Одноразовий дамп PostgreSQL → parquet
├── init_db.py                      # Імпорт CSV → PostgreSQL
├── check_connection.py             # Тест підключення
├── requirements.txt
└── .env                            # DB credentials
```

---

## Швидкий старт

### 1. Клонування та залежності
```bash
git clone https://github.com/YOUR_USERNAME/home-credit-default-risk.git
cd home-credit-default-risk
pip install -r requirements.txt
```

### 2. Налаштування `.env`
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=data default
DB_USER=postgres
DB_PASSWORD=your_password
```

### 3. Імпорт даних у PostgreSQL (якщо ще не зроблено)
```bash
# Покласти CSV файли в data/home-credit-default-risk/
python init_db.py
```

### 4. Дамп PostgreSQL → parquet (один раз, ~5-15 хв)
```bash
python dump_to_parquet.py
```
> Після цього всі ноутбуки читають дані за **~10 секунд** замість хвилин.

### 5. EDA
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 6. Тренування моделі
```bash
# Через ноутбук (рекомендовано — є всі графіки та SHAP):
jupyter notebook notebooks/model.ipynb

# Або через CLI:
python -m src.train
```

### 7. Запуск Streamlit
```bash
streamlit run streamlit_app/app.py
```
Відкриється: **http://localhost:8501**

<img width="3549" height="1511" alt="image" src="https://github.com/user-attachments/assets/6d508038-7b6a-404f-b1e6-f16e7b616083" />
<img width="3579" height="1766" alt="image" src="https://github.com/user-attachments/assets/239f5eb0-b3bc-4b28-ae12-bde67508bac3" />


---

## ML Pipeline

```
PostgreSQL (7 таблиць, ~56M рядків)
         │
         ▼  dump_to_parquet.py (1 раз)
  data/parquet/*.parquet  (~800 MB)
         │
         ▼  features.py
  Feature Engineering
  ┌─────────────────────────────────────────┐
  │  bureau           → 11 агрег. ознак     │
  │  bureau_balance   →  4 ознаки           │
  │  previous_app     → 11 ознак            │
  │  pos_cash         →  8 ознак            │
  │  installments     →  8 ознак            │
  │  credit_card      → 10 ознак            │
  │  application_train→ ~120 вихідних ознак │
  └─────────────────────────────────────────┘
         │  ~172 ознаки разом
         ▼  preprocessing.py
  sklearn Pipeline
  ├─ SimpleImputer(strategy="median")
  └─ OrdinalEncoder(handle_unknown="use_encoded_value")
         │
         ▼  train.py
  LightGBM Classifier
  ├─ n_estimators=1000, learning_rate=0.05
  ├─ num_leaves=31, class_weight="balanced"
  └─ subsample=0.8, colsample_bytree=0.8
         │
         ▼
  5-fold StratifiedKFold CV
         │
         ▼
  ROC-AUC ~0.76+
```

**Важливі рішення:**
- `class_weight="balanced"` — обробка дисбалансу класів (~8% дефолту)
- `OrdinalEncoder(handle_unknown="use_encoded_value")` — стійкість до нових категорій
- `SimpleImputer(median)` — робастна імпутація пропусків
- Оптимальний поріг рішення обирається за max F1 на val-сеті

---

## Streamlit Dashboard

**4 сторінки:**

| Сторінка | Функціонал |
|---------|-----------|
| 🔮 **Скоринг клієнта** | Форма введення → ймовірність дефолту, кредитний скор 300–850 (gauge), рекомендації |
| 📊 **Аналіз портфеля** | Завантаження CSV → пакетний скоринг → pie chart ризиків, гістограми |
| 📈 **Метрики моделі** | ROC-AUC, PR-крива, confusion matrix, feature importance, SHAP plots |
| ℹ️ **Про проєкт** | Документація, архітектура, метадані моделі |

**Ключова логіка скорингу:**
- Введені користувачем дані → `build_input_row()` → вирівнювання під список ознак моделі
- Відсутні колонки автоматично → `NaN` → імпутуються медіаною в pipeline
- Кредитний скор: `300 + (1 − P(дефолту)) × 550`

---

## Результати

| Метрика | Значення |
|---------|---------|
| CV ROC-AUC (mean) | **~0.764** |
| CV ROC-AUC (std)  | ±0.002 |
| Test ROC-AUC | **~0.769** |
| Оптимальний поріг | ~0.15–0.20 |

**Топ ознаки (SHAP):**
1. `ext_source_2` — зовнішній скоринг 2
2. `ext_source_3` — зовнішній скоринг 3
3. `ext_source_1` — зовнішній скоринг 1
4. `days_birth` — вік клієнта
5. `amt_credit` / `amt_goods_price` — сума кредиту
6. `inst_avg_days_late` — середнє прострочення платежів
7. `bureau_total_debt` — загальний борг у бюро

---

## Стек

```
Python 3.11    pandas / numpy / pyarrow
PostgreSQL 16  SQLAlchemy / pg8000
LightGBM 4.3   scikit-learn / SHAP
Streamlit 1.35 Plotly / Matplotlib / Seaborn
```

---

## .gitignore

```
data/
models/*.pkl
models/*.csv
models/plots/
.env
__pycache__/
.venv/
*.pyc
```

-----

# Автор
**Чучман Дмитро**

Студент КНУ ім. Тараса Шевченка (Механіко-математичний факультет)

Сфера інтересів: Data Science, Machine Learning, Analytics.
