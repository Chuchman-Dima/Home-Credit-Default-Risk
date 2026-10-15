# Home Credit Default Risk

> ML-система кредитного скорингу та прогнозування дефолту на основі **LightGBM**.
> Повний цикл: EDA → Feature Engineering → Training → Explainability.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-orange)](https://lightgbm.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2)](https://shap.readthedocs.io)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.76%2B-green)](#результати)

---

## Зміст

- [Про проєкт](#про-проєкт)
- [Дані](#дані)
- [Структура репозиторію](#структура-репозиторію)
- [ML Pipeline](#ml-pipeline)
- [Результати](#результати)
- [Стек](#стек)
- [Автор](#автор)

---

## Про проєкт

**Задача:** передбачити, чи допустить клієнт дефолт за кредитом (бінарна класифікація з дисбалансом класів, ~8% дефолтів).

Проєкт охоплює такі етапи:

1. **EDA** — розвідувальний аналіз 7 взаємопов'язаних таблиць.
2. **Feature Engineering** — агрегування історичних таблиць до рівня клієнта (`SK_ID_CURR`) та побудова ~170 ознак.
3. **Training** — LightGBM, 5-fold стратифікована крос-валідація, підбір порогу рішення.
4. **Explainability** — SHAP-аналіз глобальної та локальної важливості ознак.

## Дані

Датасет [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle) — 7 таблиць:

| Таблиця | Рядків | Опис |
|---------|-------:|------|
| `application_train` | 307 511 | Основні заявки на кредит |
| `bureau` | 1 716 428 | Кредитна історія з кредитного бюро |
| `bureau_balance` | 27 299 925 | Місячні баланси кредитів бюро |
| `previous_application` | 1 670 214 | Попередні заявки в Home Credit |
| `pos_cash_balance` | 10 001 358 | POS-кредити та готівкові займи |
| `installments_payments` | 13 605 401 | Платіжна дисципліна |
| `credit_card_balance` | 3 840 312 | Баланси кредитних карток |

> Сирі дані не зберігаються в репозиторії (~3.5 GB). Їх можна завантажити зі сторінки змагання на Kaggle.

## Структура репозиторію

```
Home-Credit-Default-Risk/
│
├── artifacts/
│   └── figures/        # Графіки: CV, ROC, SHAP та ін.
│
├── models/             # Натренована модель і супутні файли
│
├── notebooks/          # Jupyter-ноутбуки: EDA, тренування моделі
│
├── src/                # Код проєкту: feature engineering, препроцесинг, тренування
│
├── requirements.txt    # Залежності
├── .gitignore
└── README.md
```

## ML Pipeline

```
 7 таблиць Home Credit
         │
         ▼  Feature Engineering
  ┌──────────────────────────────────────────┐
  │  bureau            → 11 агрег. ознак     │
  │  bureau_balance    →  4 ознаки           │
  │  previous_app      → 11 ознак            │
  │  pos_cash          →  8 ознак            │
  │  installments      →  8 ознак            │
  │  credit_card       → 10 ознак            │
  │  application_train → ~120 вихідних ознак │
  └──────────────────────────────────────────┘
         │  ~172 ознаки разом
         ▼  Preprocessing (sklearn Pipeline)
  ├─ SimpleImputer(strategy="median")
  └─ OrdinalEncoder(handle_unknown="use_encoded_value")
         │
         ▼  Training
  LightGBM Classifier
  ├─ n_estimators=1000, learning_rate=0.05
  ├─ num_leaves=31, class_weight="balanced"
  └─ subsample=0.8, colsample_bytree=0.8
         │
         ▼
  5-fold StratifiedKFold CV  →  підбір порогу  →  SHAP
```

**Ключові рішення:**

- `class_weight="balanced"` — обробка дисбалансу класів.
- `OrdinalEncoder(handle_unknown="use_encoded_value")` — стійкість до нових категорій.
- `SimpleImputer(median)` — робастна імпутація пропусків.
- Оптимальний поріг рішення обирається за max F1 на валідаційній вибірці.

## Результати

| Метрика | Значення |
|---------|---------:|
| CV ROC-AUC (mean) | **~0.764** |
| CV ROC-AUC (std) | ±0.002 |
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


## Стек

| Напрям | Інструменти |
|--------|-------------|
| Дані | pandas, numpy, pyarrow, SQLAlchemy, pg8000 |
| ML | LightGBM, XGBoost, scikit-learn, imbalanced-learn, Optuna |
| Explainability | SHAP |
| Візуалізація | matplotlib, seaborn, plotly |
| Середовище | Jupyter |

## Автор

**Чучман Дмитро**
Студент КНУ ім. Тараса Шевченка (Механіко-математичний факультет)

Сфера інтересів: Data Science, Machine Learning, Analytics.

[GitHub](https://github.com/Chuchman-Dima)
