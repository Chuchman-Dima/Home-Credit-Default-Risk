"""
dump_to_parquet.py — Одноразовий дамп PostgreSQL → локальні parquet файли.

Запускається ОДИН РАЗ з кореня проєкту:
    python dump_to_parquet.py

Після цього всі ноутбуки читають з диску
"""

import os
import sys
import time

# Знайти корінь проєкту
THIS_FILE   = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)

if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)

from src.db import load_table

# Папка для збереження
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")
os.makedirs(DATA_DIR, exist_ok=True)

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "credit_card_balance",
    "installments_payments",
    "pos_cash_balance",
    "previous_application",
]


def dump_table(table_name: str) -> None:
    parquet_path = os.path.join(DATA_DIR, f"{table_name}.parquet")

    if os.path.exists(parquet_path):
        size_mb = os.path.getsize(parquet_path) / 1024 / 1024
        print(f"{table_name:<35} вже є ({size_mb:.1f} MB) — пропускаємо")
        return

    print(f"{table_name:<35} завантаження...", end=" ", flush=True)
    t0 = time.time()

    df = load_table(table_name)
    df.columns = [c.lower() for c in df.columns]

    df.to_parquet(parquet_path, index=False, compression="snappy")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(parquet_path) / 1024 / 1024
    rows = df.shape[0]
    print(f"{rows:>9,} рядків  {size_mb:>6.1f} MB  {elapsed:>4.0f}с")


def main():
    print("=" * 60)
    print("  PostgreSQL → Parquet дамп")
    print("=" * 60)
    print(f"  Корінь проєкту : {PROJECT_ROOT}")
    print(f"  Parquet папка  : {DATA_DIR}")
    print()

    try:
        import pyarrow
    except ImportError:
        print("pyarrow не встановлено!")
        print("   Виконайте: pip install pyarrow")
        sys.exit(1)

    total_start = time.time()
    success, failed = [], []

    for table in TABLES:
        try:
            dump_table(table)
            success.append(table)
        except Exception as e:
            print(f"ПОМИЛКА '{table}': {e}")
            failed.append((table, str(e)))

    elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"  {len(success)} OK, {len(failed)} помилок — {elapsed:.0f}с загалом")
    print("=" * 60)

    if success:
        total_size = 0
        print("\nЗбережені файли:")
        for table in success:
            path = os.path.join(DATA_DIR, f"{table}.parquet")
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024 / 1024
                total_size += size
                print(f"  {table:<35} {size:>7.1f} MB")
        print(f"  {'ВСЬОГО':<35} {total_size:>7.1f} MB")

    if failed:
        print("\nПомилки:")
        for t, e in failed:
            print(f"{t}: {e}")
    else:
        print("\nГотово! Тепер notebooks читають дані з диску через load_local().")

if __name__ == "__main__":
    main()