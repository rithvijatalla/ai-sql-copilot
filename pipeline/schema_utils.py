"""
Shared SQLite schema-introspection helpers.

Split out from query_engine.py so guardrails/checks.py can read table/column
metadata (and sample data) for the *active* database - demo or uploaded -
without creating a circular import (query_engine already imports
guardrails.checks).
"""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(BASE_DIR / "db" / "analytics.db")


def get_table_names(db_path: str = DEFAULT_DB_PATH) -> list[str]:
    """Return every user table name in the database, alphabetically."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_table_columns(db_path: str, table_name: str) -> list[tuple[str, str, bool]]:
    """Return (column_name, declared_type, is_primary_key) for a table, in
    declaration order."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
        return [(row[1], row[2], bool(row[5])) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_schema_description(db_path: str = DEFAULT_DB_PATH) -> str:
    """Return a text description of every table and column in the database.

    Works against any SQLite file - the hardcoded demo DB or a dynamically
    loaded one - since it's pure introspection over sqlite_master/PRAGMA.
    """
    table_names = get_table_names(db_path)

    lines = []
    for table_name in table_names:
        lines.append(f"Table: {table_name}")
        for col_name, col_type, is_pk in get_table_columns(db_path, table_name):
            marker = " (PRIMARY KEY)" if is_pk else ""
            lines.append(f"  - {col_name}: {col_type}{marker}")
        lines.append("")

    return "\n".join(lines).strip()
