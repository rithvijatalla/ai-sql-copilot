"""
Core natural-language-to-SQL pipeline for the analytics copilot.

Flow: get_schema_description() reads the live schema from db/analytics.db ->
generate_sql() sends the question + schema to Claude and gets back a SQL
SELECT statement -> execute_sql_readonly() runs it against the database
opened in genuine read-only mode -> ask() wires all three together.

Safety is layered: the system prompt instructs SELECT-only generation, and
execute_sql_readonly() independently refuses to run anything that isn't a
single SELECT/WITH statement, on a connection SQLite physically cannot write
through (URI "?mode=ro"). Neither layer trusts the other.
"""

import re
import sqlite3
import sys
from pathlib import Path

import anthropic
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from guardrails.checks import (
    check_granularity_mismatch,
    check_messy_categorical_filter,
    check_out_of_scope,
    check_undefined_metric,
)
from pipeline.schema_utils import DEFAULT_DB_PATH, get_schema_description

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a SQL generator for a SQLite analytics database.

You will be given a description of the database schema and a natural language
question. Respond with ONLY a single valid SQLite SELECT query that answers
the question - nothing else.

Rules:
- Output only the raw SQL. No explanation, no markdown formatting, no code
  fences, no backticks, no trailing commentary.
- Only generate SELECT statements (WITH ... SELECT CTEs are fine). Never
  generate INSERT, UPDATE, DELETE, DROP, ALTER, or any other statement that
  modifies data or schema.
- Use only the tables and columns given in the schema.
- Output a single statement, not multiple statements separated by semicolons.
"""


def generate_sql(question: str, schema_description: str) -> str:
    """Ask Claude to translate a natural language question into a SQL SELECT query."""
    client = anthropic.Anthropic()

    user_message = (
        f"Schema:\n{schema_description}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": user_message}],
    )

    sql = next(block.text for block in response.content if block.type == "text")
    return sql.strip().strip(";").strip()


_ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH")


def execute_sql_readonly(sql: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Execute a SQL query against the database opened in genuine read-only mode.

    Refuses (raises ValueError) if the statement isn't a single SELECT/WITH
    query, as a second layer of defense independent of the prompt-level
    instruction to the model.
    """
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        raise ValueError(f"Expected a single SQL statement, got {len(statements)}")

    stripped = statements[0].strip()
    first_word = re.match(r"^([A-Za-z]+)", stripped)
    keyword = first_word.group(1).upper() if first_word else ""
    if keyword not in _ALLOWED_LEADING_KEYWORDS:
        raise ValueError(
            f"Refusing to execute non-SELECT statement (starts with {keyword!r}): {sql!r}"
        )

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return pd.read_sql_query(stripped, conn)
    finally:
        conn.close()


def ask(question: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Answer a natural language question by generating and running SQL against the database."""
    result = {
        "question": question,
        "sql": None,
        "result": None,
        "error": None,
        "clarification_needed": None,
        "warning": None,
    }

    try:
        schema_description = get_schema_description(db_path)
    except sqlite3.Error as e:
        result["error"] = f"Failed to read schema: {e}"
        return result

    try:
        clarification = check_undefined_metric(
            question, schema_description
        ) or check_out_of_scope(question, schema_description)
    except (anthropic.AnthropicError, TypeError) as e:
        result["error"] = f"Guardrail check failed: {e}"
        return result

    if clarification:
        result["clarification_needed"] = clarification
        return result

    try:
        sql = generate_sql(question, schema_description)
        result["sql"] = sql
    except (anthropic.AnthropicError, TypeError) as e:
        result["error"] = f"Failed to generate SQL: {e}"
        return result

    warnings = [
        w
        for w in (
            check_granularity_mismatch(sql, db_path),
            check_messy_categorical_filter(sql, db_path),
        )
        if w
    ]
    result["warning"] = "\n\n".join(warnings) if warnings else None

    try:
        result["result"] = execute_sql_readonly(sql, db_path)
    except (sqlite3.Error, ValueError) as e:
        result["error"] = str(e)

    return result
