"""
db.py - PostgreSQL connection using pg8000 (pure Python driver).
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

_DB_HOST     = "localhost"
_DB_PORT     = 5432
_DB_NAME     = "data default"
_DB_USER     = "postgres"
_DB_PASSWORD = "1111"


def get_engine():
    """Return SQLAlchemy engine using pg8000 driver."""
    host     = os.environ.get("DB_HOST",     _DB_HOST)
    port     = int(os.environ.get("DB_PORT", _DB_PORT))
    name     = os.environ.get("DB_NAME",     _DB_NAME)
    user     = os.environ.get("DB_USER",     _DB_USER)
    password = os.environ.get("DB_PASSWORD", _DB_PASSWORD)

    # Pass database name via connect_args to avoid URL encoding issues
    url = f"postgresql+pg8000://{user}:{password}@{host}:{port}/postgres"
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"database": name},
    )


def test_connection() -> bool:
    """Test database connection."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
        print("OK - PostgreSQL connection successful!")
        print(f"    Server: {version[:70]}")
        return True
    except Exception as e:
        print(f"ERROR - Connection failed: {e}")
        return False


def list_tables() -> list:
    """Return list of all tables in public schema."""
    engine = get_engine()
    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        tables = [row[0] for row in result]
    return tables


def load_table(
    table_name: str,
    columns: list = None,
    limit: int = None,
    chunksize: int = None,
):
    """
    Load a table from PostgreSQL into a DataFrame.

    Args:
        table_name:  table name (e.g. 'application_train')
        columns:     list of columns to select (None = all)
        limit:       max number of rows (None = all)
        chunksize:   if set, returns an iterator of chunks

    Returns:
        pd.DataFrame
    """
    engine = get_engine()

    cols         = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    limit_clause = f"LIMIT {limit}" if limit else ""
    query        = f'SELECT {cols} FROM public."{table_name}" {limit_clause};'

    if chunksize:
        return pd.read_sql(query, engine, chunksize=chunksize)

    df = pd.read_sql(query, engine)
    print(f"Loaded '{table_name}': {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_all_tables() -> dict:
    """Load all main project tables. Returns dict {table_name: DataFrame}."""
    tables = [
        "application_train",
        "application_test",
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "installments_payments",
        "pos_cash_balance",
        "previous_application",
    ]
    data = {}
    for table in tables:
        try:
            data[table] = load_table(table)
        except Exception as e:
            print(f"WARNING: Table '{table}' not available: {e}")
    return data


def get_table_info(table_name: str) -> pd.DataFrame:
    """Return column info for a table: name, type, nullable."""
    engine = get_engine()
    query = f"""
        SELECT
            column_name,
            data_type,
            is_nullable,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = '{table_name}'
        ORDER BY ordinal_position;
    """
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    test_connection()
    print("\nTables in database:")
    for t in list_tables():
        print(f"  - {t}")