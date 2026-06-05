"""
init_db.py - Import CSV files into PostgreSQL.

Imports all CSV files from data/home-credit-default-risk/ into
the 'data default' PostgreSQL database.

Run: python init_db.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.db import get_engine

# Folder with CSV files (relative to this script)
DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "home-credit-default-risk"
)

# Tables to skip (description file, not data)
SKIP_FILES = {"homecredit_columns_description"}

CHUNKSIZE = 50_000


def get_csv_files(folder: str) -> list[tuple[str, str]]:
    """Return list of (table_name, file_path) for all CSV files."""
    result = []
    if not os.path.exists(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        table_name = fname.replace(".csv", "").replace("-", "_").lower()
        if table_name in SKIP_FILES:
            print(f"  Skipping: {fname}")
            continue
        result.append((table_name, os.path.join(folder, fname)))
    return result


def get_row_count(file_path: str) -> int:
    """Count rows in CSV without loading into memory."""
    with open(file_path, encoding="utf-8") as f:
        count = sum(1 for _ in f) - 1  # minus header
    return count


def import_csv(table_name: str, file_path: str, engine) -> None:
    """Import a single CSV into PostgreSQL using chunked reading."""
    total_rows = get_row_count(file_path)
    print(f"\n  Importing '{table_name}'...")
    print(f"  File: {os.path.basename(file_path)} ({total_rows:,} rows)")

    imported = 0
    for i, chunk in enumerate(pd.read_csv(
        file_path,
        chunksize=CHUNKSIZE,
        encoding="utf-8",
        low_memory=False,
    )):
        # Normalize column names: lowercase, no spaces
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if_exists = "replace" if i == 0 else "append"
        chunk.to_sql(
            table_name,
            engine,
            schema="public",
            if_exists=if_exists,
            index=False,
            method="multi",
        )
        imported += len(chunk)
        pct = imported / total_rows * 100
        print(f"    {imported:,} / {total_rows:,} rows ({pct:.0f}%)", end="\r")

    print(f"    Done: {imported:,} rows imported          ")


def verify_table(table_name: str, engine) -> None:
    """Print row count from DB to verify import."""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text(f'SELECT COUNT(*) FROM public."{table_name}"')
        )
        count = result.fetchone()[0]
    print(f"    Verified in DB: {count:,} rows")


def main():
    print("=" * 55)
    print("  Home Credit - CSV to PostgreSQL Import")
    print("=" * 55)
    print(f"  Data folder: {DATA_FOLDER}\n")

    engine = get_engine()

    csv_files = get_csv_files(DATA_FOLDER)
    if not csv_files:
        print("ERROR: No CSV files found!")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to import:")
    for name, path in csv_files:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  - {name:<35} ({size_mb:.1f} MB)")

    print("\nStarting import...")

    success = []
    failed = []

    for table_name, file_path in csv_files:
        try:
            import_csv(table_name, file_path, engine)
            verify_table(table_name, engine)
            success.append(table_name)
        except Exception as e:
            print(f"\n  ERROR on '{table_name}': {e}")
            failed.append((table_name, str(e)))

    print("\n" + "=" * 55)
    print(f"  Import complete: {len(success)} OK, {len(failed)} failed")
    print("=" * 55)

    if success:
        print("\nSuccessfully imported:")
        for t in success:
            print(f"  + {t}")

    if failed:
        print("\nFailed:")
        for t, err in failed:
            print(f"  x {t}: {err}")


if __name__ == "__main__":
    main()