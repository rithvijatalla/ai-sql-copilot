"""
Loads user-uploaded CSV/Excel files into a fresh SQLite database so the rest
of the pipeline (schema_utils, query_engine, guardrails) can treat uploaded
data exactly like the bundled demo dataset - just a different db_path.

Each upload gets its own temp SQLite file (not the shared in-process
:memory: special-case, since execute_sql_readonly() needs a real file path
to reopen the connection in genuine read-only mode). The caller is
responsible for deleting the file (see delete_database) once the session no
longer needs it.
"""

import os
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

_INVALID_CHARS = re.compile(r"[^0-9a-zA-Z_]")
_SUPPORTED_EXCEL_SUFFIXES = (".xlsx", ".xls")


def sanitize_identifier(name: str, existing: set[str] = frozenset()) -> str:
    """Turn an arbitrary string (e.g. a filename stem) into a valid, unique
    SQL identifier: lowercase, alnum/underscore only, doesn't start with a
    digit, doesn't collide with a name already in `existing`."""
    cleaned = _INVALID_CHARS.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"

    candidate = cleaned
    suffix = 2
    while candidate in existing:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    return candidate


def load_uploaded_files(files) -> str:
    """Load a list of Streamlit UploadedFile objects (CSV or Excel) into a
    new temp SQLite database, one table per CSV file / Excel sheet. Returns
    the path to that database.

    Table names are derived from the filename (and sheet name, for
    multi-sheet Excel files), sanitized to a valid SQL identifier, and
    de-duplicated if two files/sheets would otherwise collide.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="upload_")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    existing_tables: set[str] = set()
    try:
        for uploaded_file in files:
            suffix = Path(uploaded_file.name).suffix.lower()
            stem = Path(uploaded_file.name).stem

            if suffix == ".csv":
                df = pd.read_csv(uploaded_file)
                table_name = sanitize_identifier(stem, existing_tables)
                existing_tables.add(table_name)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            elif suffix in _SUPPORTED_EXCEL_SUFFIXES:
                sheets = pd.read_excel(uploaded_file, sheet_name=None)
                multi_sheet = len(sheets) > 1
                for sheet_name, df in sheets.items():
                    label = f"{stem}_{sheet_name}" if multi_sheet else stem
                    table_name = sanitize_identifier(label, existing_tables)
                    existing_tables.add(table_name)
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
            else:
                raise ValueError(
                    f"Unsupported file type for '{uploaded_file.name}' - "
                    "only .csv, .xlsx, and .xls are supported."
                )
    except Exception:
        # Partial failure (e.g. a malformed file later in the list) would
        # otherwise leak this temp file on disk forever - the caller never
        # gets db_path back to clean it up themselves, since we're about to
        # raise instead of return. Delete it ourselves, then let the
        # original exception propagate unchanged.
        conn.close()
        delete_database(db_path)
        raise
    else:
        conn.close()

    return db_path


def delete_database(db_path: str) -> None:
    """Remove a temp database created by load_uploaded_files, if it exists."""
    path = Path(db_path)
    if path.exists():
        path.unlink()
