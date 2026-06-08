# Makefile — зручні команди для управління Docker
.PHONY: build up down restart logs ps dump train streamlit jupyter clean help

# ── Збірка та запуск ─────────────────────────────────────────────────
build:          ## Зібрати всі образи
	docker compose build

up:             ## Запустити всі сервіси (фон)
	docker compose up -d
	@echo ""
	@echo "✅ Сервіси запущені:"
	@echo "   Jupyter    → http://localhost:8888"
	@echo "   Streamlit  → http://localhost:8501"
	@echo "   PostgreSQL → localhost:5432"

up-build:       ## Зібрати і запустити
	docker compose up -d --build

down:           ## Зупинити всі сервіси
	docker compose down

restart:        ## Перезапустити
	docker compose restart

# ── Логи ──────────────────────────────────────────────────────────────
logs:           ## Логи всіх сервісів
	docker compose logs -f

logs-jupyter:   ## Логи Jupyter
	docker compose logs -f jupyter

logs-streamlit: ## Логи Streamlit
	docker compose logs -f streamlit

logs-postgres:  ## Логи PostgreSQL
	docker compose logs -f postgres

ps:             ## Статус контейнерів
	docker compose ps

# ── Робота з даними ───────────────────────────────────────────────────
init-db:        ## Імпорт CSV → PostgreSQL (один раз)
	docker compose exec jupyter python init_db.py

dump:           ## Дамп PostgreSQL → parquet (один раз)
	docker compose exec jupyter python dump_to_parquet.py

check-db:       ## Перевірити підключення до БД
	docker compose exec jupyter python check_connection.py

# ── ML ────────────────────────────────────────────────────────────────
train:          ## Тренування моделі через CLI
	docker compose exec jupyter python -m src.train

# ── Shell ─────────────────────────────────────────────────────────────
shell-jupyter:  ## Shell у Jupyter контейнері
	docker compose exec jupyter bash

shell-postgres: ## psql у PostgreSQL
	docker compose exec postgres psql -U postgres -d "data default"

# ── Очищення ──────────────────────────────────────────────────────────
clean:          ## Зупинити та видалити контейнери (volumes зберігаються)
	docker compose down

clean-all:      ## Видалити все включно з volumes (УВАГА: видаляє дані!)
	docker compose down -v
	@echo "⚠️  Всі volumes видалено (PostgreSQL, parquet, models)"

# ── Допомога ──────────────────────────────────────────────────────────
help:           ## Показати список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help