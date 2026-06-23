"""
db.py - PostgreSQL connection + локальний кеш через parquet.

  load_local(table)  → parquet з диску або fallback PostgreSQL
  load_table(table)  → завжди PostgreSQL
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

# ── Config ─────────────────────────────────────────────────────────────
_DB_HOST     = "localhost"
_DB_PORT     = 5432
_DB_NAME     = "data default"
_DB_USER     = "postgres"
_DB_PASSWORD = "1111"

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

_PARQUET_DIR = os.path.join(_PROJECT_ROOT, "data", "parquet")


def get_engine():
    """
        Створює об'єкт SQLAlchemy Engine.
        Пріоритет віддається змінним середовища (env variables). Якщо їх немає,
        використовуються хардкод-значення за замовчуванням.
    """

    host     = os.environ.get("DB_HOST",     _DB_HOST)
    port     = int(os.environ.get("DB_PORT", _DB_PORT))
    name     = os.environ.get("DB_NAME",     _DB_NAME)
    user     = os.environ.get("DB_USER",     _DB_USER)
    password = os.environ.get("DB_PASSWORD", _DB_PASSWORD)

    url = f"postgresql+pg8000://{user}:{password}@{host}:{port}/postgres"
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"database": name},
    )


def test_connection() -> bool:
    """
        Перевіряє з'єднання з базою даних, виконуючи найпростіший запит версії.
        Повертає True, якщо з'єднання успішне.
    """

    try:
        engine = get_engine()
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).fetchone()[0]
        print(f"PostgreSQL OK: {version[:60]}")
        return True
    except Exception as e:
        print(f"З'єднання не вдалось: {e}")
        return False


def list_tables() -> list:
    """
        Повертає список усіх таблиць у публічній схемі бази даних PostgreSQL.
    """

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name"
        ))
        return [r[0] for r in rows]


def load_table(
    table_name: str,
    columns: list = None,
    limit: int = None,
    chunksize: int = None,):
    """
        Зчитує дані безпосередньо з PostgreSQL.
        Використовується переважно для створення локальних дампів або якщо Parquet відсутній.
    """

    engine = get_engine()
    cols         = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    limit_clause = f"LIMIT {limit}" if limit else ""
    query        = f'SELECT {cols} FROM public."{table_name}" {limit_clause};'

    if chunksize:
        return pd.read_sql(query, engine, chunksize=chunksize)

    df = pd.read_sql(query, engine)
    print(f"Loaded '{table_name}' from PostgreSQL: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


# ── Local parquet cache ← ОСНОВНИЙ МЕТОД ──────────────────────────────
def load_local(
    table_name: str,
    columns: list = None,
    parquet_dir: str = None,
) -> pd.DataFrame:
    """
    Основна функція для роботи з даними.
    Намагається зчитати дані з оптимізованого Parquet файлу (значно швидше за SQL).
    Якщо файл відсутній, автоматично завантажує дані з PostgreSQL.
    """
    pdir = parquet_dir or _PARQUET_DIR
    path = os.path.join(pdir, f"{table_name}.parquet")

    if os.path.exists(path):
        df = pd.read_parquet(path, columns=columns)
        print(f"'{table_name}' з parquet: {df.shape[0]:,} рядків, {df.shape[1]} колонок")
        return df

    # fallback
    print(f"Parquet '{table_name}' не знайдено ({path})")
    print(f"   → Запустіть: python dump_to_parquet.py")
    print(f"   → Поки що читаємо з PostgreSQL...")
    df = load_table(table_name, columns=columns)
    df.columns = [c.lower() for c in df.columns]
    return df


def parquet_status() -> None:
    """
    Виводить інформацію про наявність та розмір локальних Parquet файлів.
    Допомагає зрозуміти, чи всі дані були успішно закешовані.
    """
    tables = [
        "application_train", "application_test",
        "bureau", "bureau_balance", "credit_card_balance",
        "installments_payments", "pos_cash_balance", "previous_application",
    ]
    print(f"\nParquet кеш: {_PARQUET_DIR}")
    print("-" * 55)
    total = 0
    for t in tables:
        path = os.path.join(_PARQUET_DIR, f"{t}.parquet")
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            total += size
            print(f"{t:<35} {size:>6.1f} MB")
        else:
            print(f"{t:<35}  немає  ← запустіть dump_to_parquet.py")
    print(f"  {'ВСЬОГО':<35} {total:>6.1f} MB\n")


def load_all_tables(use_local: bool = True) -> dict:
    """
        Утиліта для одночасного завантаження всіх таблиць проєкту в пам'ять.
        Повертає словник, де ключі — назви таблиць, а значення — датафрейми.
    """

    tables = [
        "application_train", "application_test",
        "bureau", "bureau_balance", "credit_card_balance",
        "installments_payments", "pos_cash_balance", "previous_application",
    ]
    loader = load_local if use_local else load_table
    data = {}
    for t in tables:
        try:
            data[t] = loader(t)
        except Exception as e:
            print(f"WARNING: '{t}' не доступна: {e}")
    return data


def get_table_info(table_name: str) -> pd.DataFrame:
    """
        Повертає метадані структури конкретної таблиці з PostgreSQL
        (назви колонок, типи даних, чи допускає NULL значення).
    """

    engine = get_engine()
    query = f"""
        SELECT column_name, data_type, is_nullable, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='{table_name}'
        ORDER BY ordinal_position;
    """
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    parquet_status()
    test_connection()