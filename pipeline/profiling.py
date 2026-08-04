"""
Lightweight, automatic data quality profiling for the active dataset (demo
or uploaded) - runs once a dataset is loaded, before any question is
asked, so problems are visible up front rather than discovered one warning
at a time as questions happen to touch them.

Deliberately a summary, not a full analysis: per table, just row count,
null percentage per column, a full-row duplicate count, and which columns
are "messy" (inconsistent casing/whitespace). The messiness check reuses
guardrails.checks.is_messy_column() directly rather than reimplementing
it - the same definition of "messy" the granularity/categorical-filter
guardrail uses at query time.

Streamlit-agnostic like the rest of pipeline/ - app.py is responsible for
caching (profiling touches every table/column and isn't free) and
rendering.
"""

import sqlite3
from dataclasses import dataclass, field

from guardrails.checks import is_messy_column, is_text_type
from pipeline.schema_utils import DEFAULT_DB_PATH, get_table_columns, get_table_names


@dataclass
class ColumnProfile:
    name: str
    null_pct: float
    is_messy: bool


@dataclass
class TableProfile:
    table: str
    row_count: int
    duplicate_row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    @property
    def messy_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_messy]


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def _column_stats(
    conn: sqlite3.Connection, table: str, columns: list[tuple[str, str, bool]], row_count: int
) -> dict[str, tuple[float, bool]]:
    """One query per table: null % and "is this column a de facto unique
    identifier" (every non-null value distinct, no nulls) for every column.
    SQL COUNT(col) skips NULLs, so row_count - COUNT(col) is the null
    count; COUNT(DISTINCT col) == row_count means every row has its own
    distinct, non-null value.

    The identifier flag is used to exclude such columns from duplicate-row
    detection - a declared PRIMARY KEY is unique by construction, but
    uploaded CSVs (loaded via pandas.to_sql) never get one, so relying on
    the schema's is_pk alone would miss an obvious id column and make
    every row look trivially "unique". Checking actual distinctness
    catches both cases without guessing from the column name.
    """
    if row_count == 0 or not columns:
        return {col_name: (0.0, False) for col_name, _, _ in columns}

    col_exprs = ", ".join(f'COUNT("{c}"), COUNT(DISTINCT "{c}")' for c, _, _ in columns)
    row = conn.execute(f'SELECT {col_exprs} FROM "{table}"').fetchone()

    stats = {}
    values = iter(row)
    for col_name, _, _ in columns:
        non_null_count = next(values)
        distinct_count = next(values)
        null_pct = round((row_count - non_null_count) / row_count * 100, 1)
        is_identifier = distinct_count == row_count
        stats[col_name] = (null_pct, is_identifier)
    return stats


def _count_duplicate_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: list[tuple[str, str, bool]],
    column_stats: dict[str, tuple[float, bool]],
    row_count: int,
) -> int:
    """Full-row duplicates, excluding identifier-like column(s) (declared
    PRIMARY KEY, or empirically a distinct value per row) - otherwise an id
    column alone would make every row trivially "unique" and mask genuine
    duplicates in the rest of the row (e.g. the same customer accidentally
    inserted twice under two different ids)."""
    if row_count == 0:
        return 0

    non_identifier_columns = [
        col_name for col_name, _, is_pk in columns if not is_pk and not column_stats[col_name][1]
    ]
    if not non_identifier_columns:
        return 0

    col_list = ", ".join(f'"{c}"' for c in non_identifier_columns)
    distinct_rows = conn.execute(f'SELECT COUNT(*) FROM (SELECT 1 FROM "{table}" GROUP BY {col_list})').fetchone()[0]
    return row_count - distinct_rows


def profile_dataset(db_path: str = DEFAULT_DB_PATH) -> list[TableProfile]:
    """Profile every table in the active database: row count, null % per
    column, full-row duplicate count, and which text columns are messy
    (reusing guardrails.checks.is_messy_column)."""
    profiles = []
    conn = sqlite3.connect(db_path)
    try:
        for table in get_table_names(db_path):
            columns = get_table_columns(db_path, table)
            row_count = _count_rows(conn, table)
            column_stats = _column_stats(conn, table, columns, row_count)
            duplicate_row_count = _count_duplicate_rows(conn, table, columns, column_stats, row_count)

            column_profiles = [
                ColumnProfile(
                    name=col_name,
                    null_pct=column_stats[col_name][0],
                    is_messy=row_count > 0 and is_text_type(col_type) and is_messy_column(db_path, table, col_name),
                )
                for col_name, col_type, _ in columns
            ]

            profiles.append(
                TableProfile(
                    table=table,
                    row_count=row_count,
                    duplicate_row_count=duplicate_row_count,
                    columns=column_profiles,
                )
            )
    finally:
        conn.close()

    return profiles
