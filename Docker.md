# 🐳 Docker — Home Credit Default Risk

Повна Docker-конфігурація проєкту з трьома сервісами.

## Архітектура

```
┌─────────────────────────────────────────────────────┐
│                    docker network: hc_net            │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────┐  │
│  │   jupyter    │   │  streamlit   │   │postgres │  │
│  │  :8888       │   │   :8501      │   │  :5432  │  │
│  │              │   │              │   │         │  │
│  │ EDA.ipynb    │   │  app.py      │   │ "data   │  │
│  │ model.ipynb  │   │  scoring UI  │   │ default"│  │
│  │ train.py     │   │              │   │         │  │
│  └──────┬───────┘   └──────┬───────┘   └────┬────┘  │
│         │                  │                │        │
│         └──────────────────┘                │        │
│                   ▼                          │        │
│          ┌─────────────────┐                │        │
│          │  parquet_cache  │◄───────────────┘        │
│          │  (shared vol.)  │                         │
│          └─────────────────┘                         │
│          ┌─────────────────┐                         │
│          │   models_vol    │ (pkl + metadata)         │
│          └─────────────────┘                         │
└─────────────────────────────────────────────────────┘
```

## Сервіси

| Сервіс | Образ | Порт | Призначення |
|--------|-------|------|-------------|
| `postgres` | `postgres:16-alpine` | 5432 | База даних |
| `jupyter` | custom (python:3.11) | 8888 | EDA + тренування |
| `streamlit` | custom (python:3.11) | 8501 | Дашборд скорингу |

## Volumes

| Volume | Монтується в | Призначення |
|--------|-------------|-------------|
| `postgres_data` | postgres:/var/lib/postgresql/data | Дані PostgreSQL |
| `parquet_cache` | jupyter+streamlit:/app/data/parquet | Parquet кеш (спільний) |
| `models_vol` | jupyter+streamlit:/app/models | Модель + метадані |

---

## Швидкий старт

### Передумови
- Docker Desktop 24+ (або Docker Engine + Compose v2)
- 8 GB RAM (для тренування моделі)
- 10 GB вільного місця

### 1. Підготовка
```bash
# Клонувати репо
git clone https://github.com/YOUR/home-credit-default-risk.git
cd home-credit-default-risk

# Покласти CSV файли (з Kaggle)
mkdir -p data/home-credit-default-risk
# → скопіювати всі *.csv сюди
```

### 2. Збірка образів
```bash
make build
# або: docker compose build
```

### 3. Запуск
```bash
make up
# або: docker compose up -d
```

### 4. Імпорт даних у PostgreSQL (один раз)
```bash
make init-db
# або: docker compose exec jupyter python init_db.py
```

### 5. Дамп PostgreSQL → parquet (один раз, ~5-15 хв)
```bash
make dump
# або: docker compose exec jupyter python dump_to_parquet.py
```

### 6. Відкрити Jupyter і натренувати модель
```
http://localhost:8888
```
Відкрити `notebooks/model.ipynb` → Run All

### 7. Відкрити Streamlit дашборд
```
http://localhost:8501
```

---

## Команди Makefile

```bash
make help           # всі команди
make build          # зібрати образи
make up             # запустити (фон)
make down           # зупинити
make logs           # всі логи
make logs-streamlit # логи streamlit
make init-db        # імпорт CSV → PostgreSQL
make dump           # PostgreSQL → parquet
make train          # тренування через CLI
make shell-jupyter  # bash у jupyter контейнері
make shell-postgres # psql у postgres
make clean-all      # видалити все (включно з volumes!)
```

---

## Структура Docker файлів

```
├── docker-compose.yml
├── .env                          ← credentials
├── .dockerignore
├── Makefile
└── docker/
    ├── postgres/
    │   └── init.sql              ← створює "data default" DB
    ├── jupyter/
    │   └── Dockerfile
    └── streamlit/
        ├── Dockerfile
        └── config.toml           ← dark theme, headless
```

---

## Корисні команди

### Перевірити статус сервісів
```bash
make ps
# або: docker compose ps
```

### Перевірити підключення до БД
```bash
make check-db
```

### Переглянути parquet кеш
```bash
docker compose exec jupyter python -c "from src.db import parquet_status; parquet_status()"
```

### Підключитись до PostgreSQL напряму
```bash
make shell-postgres
# всередині: \dt  →  список таблиць
```

### Перезбудувати тільки один сервіс
```bash
docker compose build streamlit
docker compose up -d streamlit
```

---

## Змінні середовища (.env)

```env
DB_HOST=postgres       # ім'я сервісу в docker network
DB_PORT=5432
DB_NAME=data default
DB_USER=postgres
DB_PASSWORD=1111       # змінити в production!
```

> ⚠️ **Production:** змінити пароль і не комітити `.env` у git.

---

## Troubleshooting

**Streamlit не запускається — модель не знайдена:**
```bash
# Переконайтесь що модель натренована:
docker compose exec jupyter ls /app/models/
# Якщо немає pkl — відкрийте notebooks/model.ipynb і запустіть
```

**PostgreSQL не стартує:**
```bash
make logs-postgres
# Якщо volume пошкоджений:
docker compose down -v && make up
```

**Jupyter не бачить src/:**
```bash
# Перевірте монтування:
docker compose exec jupyter ls /app/src/
```

**Нова версія коду не підхоплюється:**
```bash
# src/ монтується як volume — зміни одразу видно
# Для Streamlit перезапустіть сервіс:
docker compose restart streamlit
```
