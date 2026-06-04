"""
check_connection.py - Quick DB connection test.
Run: python check_connection.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db import test_connection, list_tables, load_table, get_table_info

if __name__ == "__main__":
    print("=" * 50)
    print("  PostgreSQL Connection Check")
    print("=" * 50)

    ok = test_connection()
    if not ok:
        sys.exit(1)

    print("\nTables in database:")
    tables = list_tables()
    for t in tables:
        print(f"  - {t}")

    if "application_train" in tables:
        print("\nFirst 3 rows of application_train:")
        df = load_table("application_train", limit=3)
        print(df.to_string())

        print("\nColumn info for application_train:")
        info = get_table_info("application_train")
        print(info.to_string(index=False))

    print("\nDone! Database connected to project.")