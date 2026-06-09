# 🏦 Home Credit Default Risk
# **[Посилання на застосунок](https://home-credit-default-risk-my.streamlit.app/)**
---
<img width="3539" height="1566" alt="image" src="https://github.com/user-attachments/assets/f7b6d39e-746c-4d95-849f-4b56789c0fb8" />

> ⚠️ Примітка: Застосунок працює на безкоштовному сервері. Якщо сторінка не завантажується миттєво, зачекайте 30-60 секунд (час на «прогрів» сервера) та оновіть її.

---

> Система кредитного скорингу на основі **LightGBM** з повним ML-циклом:
> EDA → Feature Engineering → Training → Streamlit Deploy.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docs.docker.com/compose/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.76%2B-green)](/)

---

## 📋 Зміст
- [Про проєкт](#про-проєкт)
- [Структура](#структура)
- [Швидкий старт через Docker](#-швидкий-старт-через-docker) ← **рекомендовано**
- [Локальний запуск без Docker](#локальний-запуск-без-docker)
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
├── docker/
│   ├── initdb/                     # SQL-скрипти для першої ініціалізації БД
│   └── streamlit_config.toml       # Конфіг Streamlit всередині контейнера
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
│   └── app.py                      # Дашборд
│
├── Dockerfile                      # Docker образ застосунку
├── docker-compose.yml              # Оркестрація app + PostgreSQL
├── .dockerignore
├── .env.example                    # Шаблон змінних середовища
├── docker_setup.sh                 # Скрипт першого запуску (Linux/macOS)
├── docker_setup.ps1                # Скрипт першого запуску (Windows)
├── dump_to_parquet.py              # Одноразовий дамп PostgreSQL → parquet
├── init_db.py                      # Імпорт CSV → PostgreSQL
├── check_connection.py             # Тест підключення
├── requirements.txt
└── .env                            # DB credentials (не в git)
```

---

## 🐳 Швидкий старт через Docker

### Вимоги
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) або Docker Engine + Compose (Linux)
- CSV файли з Kaggle у папці `data/home-credit-default-risk/` ([посилання](https://www.kaggle.com/datasets/megancrenshaw/home-credit-default-risk))
- Натренована модель у папці `models/` (або перетренувати — крок 5 нижче)

---

### Варіант А — Автоматичний скрипт (рекомендовано)

**Linux / macOS:**
```bash
chmod +x docker_setup.sh
./docker_setup.sh
```

**Windows (PowerShell):**
```powershell
.\docker_setup.ps1
```

Скрипт автоматично:
1. Створює `.env` з шаблону
2. Збирає Docker образ
3. Запускає PostgreSQL
4. Імпортує CSV → PostgreSQL (якщо файли є)
5. Робить дамп PostgreSQL → parquet
6. Запускає Streamlit

Відкрийте **http://localhost:8501** ✅

---

### Варіант Б — Покроково вручну

#### 1. Клонування та налаштування
```bash
git clone https://github.com/YOUR_USERNAME/home-credit-default-risk.git
cd home-credit-default-risk

cp .env.example .env
# За потреби відредагуйте .env (пароль, назва БД)
```

#### 2. Підготовка директорій
```bash
mkdir -p models/plots data/parquet data/home-credit-default-risk docker/initdb
```

#### 3. Збірка та запуск контейнерів
```bash
docker compose up -d --build
```
> Запускає два сервіси: `db` (PostgreSQL 16) та `app` (Streamlit).

#### 4. Перевірка підключення до БД
```bash
docker compose exec app python check_connection.py
```

#### 5. Імпорт даних у PostgreSQL (один раз)
```bash
# Помістіть CSV з Kaggle у data/home-credit-default-risk/
docker compose run --rm app python init_db.py
```

#### 6. Дамп PostgreSQL → parquet (один раз, ~5-15 хв)
```bash
docker compose run --rm app python dump_to_parquet.py
```
> Після цього Streamlit читає дані за ~10 секунд.

#### 7. Тренування моделі
```bash
# Через CLI всередині контейнера:
docker compose run --rm app python -m src.train

# Або відкрийте ноутбук локально (поза Docker):
jupyter notebook notebooks/model.ipynb
```

#### 8. Відкрийте застосунок
```
http://localhost:8501
```

---

### Корисні команди Docker

```bash
# Переглянути логи застосунку
docker compose logs -f app

# Переглянути логи БД
docker compose logs -f db

# Перезапустити лише застосунок
docker compose restart app

# Зупинити все
docker compose down

# Зупинити та видалити дані БД (⚠️ незворотно)
docker compose down -v

# Зайти всередину контейнера
docker compose exec app bash

# Перебудувати образ після зміни коду
docker compose up -d --build app
```

---

## Локальний запуск без Docker

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

### 3. Імпорт даних у PostgreSQL
```bash
python init_db.py
```

### 4. Дамп PostgreSQL → parquet
```bash
python dump_to_parquet.py
```

### 5. EDA
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 6. Тренування моделі
```bash
# Через ноутбук (є всі графіки та SHAP):
jupyter notebook notebooks/model.ipynb

# Або через CLI:
python -m src.train
```

### 7. Запуск Streamlit
```bash
streamlit run streamlit_app/app.py
```

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
| **Скоринг клієнта** | Форма введення → ймовірність дефолту, кредитний скор 300–850 (gauge), рекомендації |
| **Аналіз портфеля** | Завантаження CSV → пакетний скоринг → pie chart ризиків, гістограми |
| **Метрики моделі** | ROC-AUC, PR-крива, confusion matrix, feature importance, SHAP plots |
| **Про проєкт** | Документація, архітектура, метадані моделі |

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
Docker         Compose v2
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

---

# Автор
**Чучман Дмитро**

Студент КНУ ім. Тараса Шевченка (Механіко-математичний факультет)

Сфера інтересів: Data Science, Machine Learning, Analytics.
