"""Завантаження таблиць Home Credit з parquet або CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import CSV_ALIASES, DATA_PARQUET, DATA_RAW, TABLES


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def parquet_path(table: str) -> Path:
    return DATA_PARQUET / f"{table}.parquet"


def csv_candidates(table: str) -> list[Path]:
    names = CSV_ALIASES.get(table, [f"{table}.csv"])
    extra = [f"{table}.csv", f"{table.upper()}.csv"]
    seen: set[str] = set()
    paths: list[Path] = []
    for name in names + extra:
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        paths.append(DATA_RAW / name)
    return paths


def table_overview(table: str) -> dict:
    path = parquet_path(table)
    if path.exists():
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        return {
            "table": table,
            "source": "parquet",
            "path": str(path),
            "rows": pf.metadata.num_rows,
            "cols": pf.metadata.num_columns,
            "size_mb": path.stat().st_size / 1024 / 1024,
            "columns": [name.lower() for name in schema.names],
        }
    for csv_path in csv_candidates(table):
        if csv_path.exists():
            return {
                "table": table,
                "source": "csv",
                "path": str(csv_path),
                "rows": None,
                "cols": None,
                "size_mb": csv_path.stat().st_size / 1024 / 1024,
                "columns": [],
            }
    return {"table": table, "source": None, "path": None, "rows": None, "cols": None, "size_mb": 0, "columns": []}


def parquet_status() -> pd.DataFrame:
    rows = [table_overview(t) for t in TABLES]
    df = pd.DataFrame(rows)
    print(f"parquet: {DATA_PARQUET}")
    print(f"raw CSV: {DATA_RAW}")
    return df


def load_table(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    pq_file = parquet_path(table)
    if pq_file.exists():
        cols = columns
        if columns:
            schema = pq.ParquetFile(pq_file).schema_arrow
            available = {name.lower(): name for name in schema.names}
            cols = [available[c.lower()] for c in columns if c.lower() in available]
        df = pd.read_parquet(pq_file, columns=cols)
        df = normalize_columns(df)
        print(f"{table}: {df.shape[0]:,} рядків × {df.shape[1]} колонок  [parquet]")
        return df

    for csv_path in csv_candidates(table):
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=columns)
            df = normalize_columns(df)
            print(f"{table}: {df.shape[0]:,} рядків × {df.shape[1]} колонок  [csv]")
            return df

    raise FileNotFoundError(
        f"Не знайдено {table}. Покладіть {table}.parquet у {DATA_PARQUET} "
        f"або CSV у {DATA_RAW}."
    )


def sample_table(table: str, n: int = 250_000, columns: list[str] | None = None) -> pd.DataFrame:
    """Читає перші row groups parquet (або head CSV), щоб не тягнути 10M+ рядків."""
    pq_file = parquet_path(table)
    if pq_file.exists():
        pf = pq.ParquetFile(pq_file)
        schema = pf.schema_arrow
        col_map = {name.lower(): name for name in schema.names}
        pq_cols = None
        if columns:
            pq_cols = [col_map[c.lower()] for c in columns if c.lower() in col_map]
        parts = []
        total = 0
        for i in range(pf.num_row_groups):
            chunk = pf.read_row_group(i, columns=pq_cols).to_pandas()
            parts.append(chunk)
            total += len(chunk)
            if total >= n:
                break
        df = normalize_columns(pd.concat(parts, ignore_index=True))
        return df.head(n)

    df = load_table(table, columns=columns)
    return df.head(n)


def csv_to_parquet() -> None:
    """Одноразово конвертує CSV з data/raw у data/parquet."""
    DATA_PARQUET.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        out = parquet_path(table)
        if out.exists():
            print(f"skip {table} (вже є parquet)")
            continue
        found = next((p for p in csv_candidates(table) if p.exists()), None)
        if found is None:
            print(f"немає CSV для {table}")
            continue
        print(f"конвертую {found.name} → {out.name}")
        pd.read_csv(found).pipe(normalize_columns).to_parquet(out, index=False)


if __name__ == "__main__":
    print(parquet_status().to_string(index=False))
