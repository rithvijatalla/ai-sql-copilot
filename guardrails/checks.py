"""
Guardrail checks for the NL-to-SQL analyst copilot.

Each check catches a specific way a question or generated query can produce
a technically-valid but misleading or unsafe answer. All four are
schema-agnostic: they introspect whichever database is active (the bundled
demo dataset or a dynamically uploaded one) rather than hardcoding table or
column names.

1. check_undefined_metric (pre-generation, LLM judgment)
   Ranking language ("top", "best", "leading", ...) is meaningless without a
   metric to rank by. Left unspecified, the SQL generator will silently pick
   one (usually revenue, or whatever column looks numeric) and the user gets
   a confident-looking answer to a question they didn't actually ask. This
   needs an LLM rather than keyword matching because "top 5 customers by
   revenue" and "top 5 customers" differ only in whether a metric is present
   further in the sentence - a real semantic read, not a regex. The active
   schema is passed in so clarifying-question suggestions name real columns
   from the active dataset instead of a fixed example schema.

2. check_granularity_mismatch (post-generation, static analysis + data sample)
   When the generated SQL joins two tables, find a date/timestamp-like
   column in each (by declared type, column name, or by sampling values and
   checking how many parse as dates), sample distinct values from that
   column, and estimate the table's grain from the minimum gap between
   consecutive sorted distinct dates (~1 day = daily, ~7 = weekly, else
   monthly). If the two joined tables' estimated grains differ and the SQL
   doesn't group by a date-truncation of the finer-grained side, flag it -
   the query is likely to double-count or misalign rows across the mismatch.
   This generalizes what was originally a hardcoded fact about `orders`
   (daily) vs `marketing_spend` (monthly) - see generate_synthetic_data.py.

3. check_out_of_scope (pre-generation, LLM judgment)
   Requests for unbounded dumps ("show me everything") or full per-row
   detail without aggregation are a scoping problem, not a SQL problem: they
   don't have a wrong answer so much as an answer nobody actually wants to
   receive as a flat table, and they're the shape of request most likely to
   leak more row-level detail than intended. Like check_undefined_metric,
   this needs semantic judgment rather than a keyword blocklist, and takes
   the active schema so its reasoning and examples aren't tied to one fixed
   set of table names.

4. check_messy_categorical_filter (post-generation, static analysis + data
   sample)
   Scans every exact-match filter (`column = 'value'`, not wrapped in a
   normalizing function) in the generated SQL. For each one, samples the
   distinct values of that column from the active database and checks
   whether any two raw values collapse to the same case/whitespace-
   normalized form (e.g. "West"/"west"/"WEST"). If so, the column is
   "messy" and the exact-match filter will silently miss the other variants
   and undercount. This generalizes what was originally hardcoded to
   customers.region (see generate_synthetic_data.py) to any text column in
   any uploaded dataset.
"""

import itertools
import re
import sqlite3

import anthropic
import pandas as pd

from pipeline.schema_utils import DEFAULT_DB_PATH, get_table_columns, get_table_names

MODEL = "claude-sonnet-4-6"


def _run_judgment(
    system_prompt: str,
    question: str,
    tool_name: str,
    tool_description: str,
    flag_field: str,
    message_field: str,
) -> str | None:
    """Force a structured yes/no + message judgment out of the model via a
    single required tool call, so the result is reliably parseable rather
    than free text that needs its own parsing heuristics."""
    client = anthropic.Anthropic(timeout=90.0)

    tool = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": {
            "type": "object",
            "properties": {
                flag_field: {"type": "boolean"},
                message_field: {"type": "string"},
            },
            "required": [flag_field, message_field],
        },
    }

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system_prompt,
        output_config={"effort": "low"},
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": question}],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        # Model declined or didn't call the tool (e.g. refusal) - fail open
        # rather than block the pipeline on an unrelated safety classifier.
        return None

    if tool_use.input.get(flag_field):
        return tool_use.input.get(message_field) or None
    return None


def analyze_undefined_metric(question: str, schema_description: str | None = None) -> dict | None:
    """Core detection shared by check_undefined_metric() (prose message,
    for ask()/existing tests) and the interactive resolution UI (structured
    candidate list) - one LLM call backing both, rather than two.

    Returns {"message": str, "candidates": list[str]} if the question needs
    metric clarification, else None. "candidates" is 2-4 short,
    human-readable metric labels (e.g. "Total revenue") suitable for a
    dropdown, grounded in the schema when one is provided.
    """
    schema_hint = (
        f"\n\nThe active database schema, for grounding metric suggestions in "
        f"real columns:\n{schema_description}"
        if schema_description
        else ""
    )

    system_prompt = f"""You review questions asked to an analytics SQL copilot.

Determine whether the question uses ranking or superlative language (e.g.
"top", "best", "worst", "highest", "lowest", "leading", "greatest") WITHOUT
specifying what metric to rank by (e.g. revenue, order count, spend, ticket
count, signups).

If a metric is explicitly named or unambiguously implied ("top customers by
revenue", "worst month for support tickets"), there is no problem.

If ranking language is used but no metric is specified or implied ("top
customers", "best region", "leading channel"), flag it, draft a short
clarifying question to ask the user, AND list 2-4 short, human-readable
candidate metric labels (e.g. "Total revenue", "Order count") that would
fit the question. Ground both the clarifying question and the candidate
labels in real, numeric-looking columns from the schema below if one is
provided; otherwise suggest generically-plausible metrics.

If the question doesn't use ranking/superlative language at all, there is no
problem.{schema_hint}"""

    client = anthropic.Anthropic(timeout=90.0)
    tool = {
        "name": "report_metric_check",
        "description": "Report whether the question needs a clarifying question about which metric to rank by.",
        "input_schema": {
            "type": "object",
            "properties": {
                "needs_clarification": {"type": "boolean"},
                "clarifying_question": {"type": "string"},
                "candidate_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 short candidate metric labels, e.g. 'Total revenue'.",
                },
            },
            "required": ["needs_clarification", "clarifying_question", "candidate_metrics"],
        },
    }

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system_prompt,
        output_config={"effort": "low"},
        tools=[tool],
        tool_choice={"type": "tool", "name": "report_metric_check"},
        messages=[{"role": "user", "content": question}],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        # Model declined or didn't call the tool (e.g. refusal) - fail open
        # rather than block the pipeline on an unrelated safety classifier.
        return None

    if not tool_use.input.get("needs_clarification"):
        return None

    return {
        "message": tool_use.input.get("clarifying_question") or None,
        "candidates": tool_use.input.get("candidate_metrics") or [],
    }


def check_undefined_metric(question: str, schema_description: str | None = None) -> str | None:
    """Flag ranking/superlative questions that don't specify what to rank by."""
    result = analyze_undefined_metric(question, schema_description)
    return result["message"] if result else None


def check_out_of_scope(question: str, schema_description: str | None = None) -> str | None:
    """Flag requests for unbounded data dumps or ungrouped per-row detail."""
    schema_hint = (
        f"\n\nThe active database schema:\n{schema_description}" if schema_description else ""
    )

    system_prompt = f"""You review questions asked to an analytics SQL copilot
over a tabular analytics database.

Determine whether the question asks for an unbounded, comprehensive data
dump (e.g. "show me everything", "list all the data", "dump the whole
database") or for individual-level, non-aggregated detail across an entire
table with no filtering or summarization (e.g. "list every row with full
details", "give me every record for every entity").

A filtered or aggregated request is fine, even if it returns many rows
("list customers in the West region", "show all orders from December",
"total revenue by customer"). The problem is specifically requests with no
bound, filter, or aggregation at all.

If the question is out of scope this way, flag it and draft a short message
explaining that it needs to be scoped down (e.g. by adding a filter, a time
range, a limit, or an aggregation), with a concrete suggestion grounded in
the schema below if one is provided.

Otherwise there is no problem.{schema_hint}"""

    return _run_judgment(
        system_prompt=system_prompt,
        question=question,
        tool_name="report_scope_check",
        tool_description="Report whether the question needs to be scoped down before it can be answered.",
        flag_field="needs_scoping",
        message_field="message",
    )


# ---------------------------------------------------------------------------
# check_granularity_mismatch - static SQL analysis + data-driven grain
# estimation
# ---------------------------------------------------------------------------

_DATE_NAME_HINTS = (
    "date",
    "time",
    "month",
    "week",
    "day",
    "year",
    "created",
    "updated",
)


def _looks_like_date_type(declared_type: str) -> bool:
    return any(hint in (declared_type or "").upper() for hint in ("DATE", "TIME"))


def _looks_like_date_name(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(hint in lowered for hint in _DATE_NAME_HINTS)


def _sample_column_values(db_path: str, table: str, column: str, limit: int = 500) -> list:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}'
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def _parses_as_date(values: list, threshold: float = 0.9) -> bool:
    if not values:
        return False
    parsed = pd.to_datetime(pd.Series(values, dtype="object"), errors="coerce", format="mixed")
    return parsed.notna().mean() >= threshold


def _find_date_column(db_path: str, table: str) -> str | None:
    """Return the best-guess date/timestamp column for a table, preferring a
    DATE/TIME-typed or date-hinted column name, falling back to sampling
    values and checking how many parse as dates."""
    columns = get_table_columns(db_path, table)

    by_type_or_name = [
        col_name
        for col_name, col_type, _ in columns
        if _looks_like_date_type(col_type) or _looks_like_date_name(col_name)
    ]
    for col_name in by_type_or_name:
        values = _sample_column_values(db_path, table, col_name)
        if _parses_as_date(values):
            return col_name

    for col_name, col_type, is_pk in columns:
        if is_pk or col_name in by_type_or_name or (col_type or "").upper() == "INTEGER":
            continue
        values = _sample_column_values(db_path, table, col_name)
        if _parses_as_date(values):
            return col_name

    return None


def _estimate_grain(db_path: str, table: str, column: str) -> str | None:
    """Estimate a table's date grain from the minimum gap between
    consecutive sorted distinct values of its date column."""
    raw_values = _sample_column_values(db_path, table, column)
    dates = pd.to_datetime(pd.Series(raw_values, dtype="object"), errors="coerce", format="mixed")
    distinct_sorted = dates.dropna().drop_duplicates().sort_values()
    if len(distinct_sorted) < 2:
        return None

    min_gap_days = distinct_sorted.diff().dropna().dt.days.min()
    if min_gap_days <= 1:
        return "daily"
    if min_gap_days <= 7:
        return "weekly"
    return "monthly"


def _referenced_tables(sql: str, db_path: str) -> list[str]:
    return [
        table
        for table in get_table_names(db_path)
        if re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE)
    ]


def _has_reconciling_aggregation(sql: str) -> bool:
    """True if the SQL groups by something and also truncates a date to a
    coarser bucket (strftime/substr/date_trunc) - a signal the finer-grained
    side was rolled up before comparison, rather than joined raw."""
    has_group_by = bool(re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE))
    has_date_truncation = bool(
        re.search(r"\bstrftime\s*\(", sql, re.IGNORECASE)
        or re.search(r"\bsubstr\s*\(\s*[\w.]+\s*,\s*1\s*,\s*\d+\s*\)", sql, re.IGNORECASE)
        or re.search(r"\bdate_trunc\s*\(", sql, re.IGNORECASE)
    )
    return has_group_by and has_date_truncation


def _find_granularity_mismatch(sql: str, db_path: str) -> dict | None:
    """Core detection shared by check_granularity_mismatch() (prose
    message) and get_granularity_mismatch_details() (structured, for the
    interactive resolution UI). Returns the first mismatched pair found as
    {"table_a", "col_a", "grain_a", "table_b", "col_b", "grain_b"}, or
    None."""
    tables = _referenced_tables(sql, db_path)
    has_join = bool(re.search(r"\bjoin\b", sql, re.IGNORECASE))
    if len(tables) < 2 or not has_join:
        return None

    table_grains: dict[str, tuple[str, str]] = {}
    for table in tables:
        date_column = _find_date_column(db_path, table)
        if not date_column:
            continue
        grain = _estimate_grain(db_path, table, date_column)
        if grain:
            table_grains[table] = (date_column, grain)

    if _has_reconciling_aggregation(sql):
        return None

    for (table_a, (col_a, grain_a)), (table_b, (col_b, grain_b)) in itertools.combinations(
        table_grains.items(), 2
    ):
        if grain_a != grain_b:
            return {
                "table_a": table_a,
                "col_a": col_a,
                "grain_a": grain_a,
                "table_b": table_b,
                "col_b": col_b,
                "grain_b": grain_b,
            }

    return None


def check_granularity_mismatch(sql: str, db_path: str = DEFAULT_DB_PATH) -> str | None:
    """Flag SQL that joins two tables with different estimated date grains
    (e.g. daily vs. monthly) without an aggregation that reconciles them."""
    mismatch = _find_granularity_mismatch(sql, db_path)
    if not mismatch:
        return None

    return (
        f"This query joins '{mismatch['table_a']}' ({mismatch['grain_a']} grain, via "
        f"{mismatch['table_a']}.{mismatch['col_a']}) with '{mismatch['table_b']}' "
        f"({mismatch['grain_b']} grain, via {mismatch['table_b']}.{mismatch['col_b']}) "
        f"without an aggregation that reconciles the two - e.g. grouping the "
        f"finer-grained side by a truncated date (strftime/substr) before "
        f"comparing. Results may double-count or misalign values across the "
        f"mismatched grains."
    )


def get_granularity_mismatch_details(sql: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Structured version of check_granularity_mismatch(), for the
    interactive resolution UI: which two tables/columns/grains are
    mismatched, without formatting it into prose."""
    return _find_granularity_mismatch(sql, db_path)


# ---------------------------------------------------------------------------
# check_messy_categorical_filter - static SQL analysis + data-driven
# messiness detection
# ---------------------------------------------------------------------------

_EXACT_MATCH_FILTER = re.compile(r"\b(?:(\w+)\.)?(\w+)\s*=\s*'([^']*)'")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def is_text_type(declared_type: str) -> bool:
    upper = (declared_type or "").upper()
    return upper == "" or "CHAR" in upper or "TEXT" in upper or "CLOB" in upper


def is_messy_column(db_path: str, table: str, column: str) -> bool:
    """A column is 'messy' if two or more of its distinct raw values
    collapse to the same case/whitespace-normalized form.

    Public (not just used internally by _find_messy_filter_match): also
    the detection pipeline/profiling.py reuses to flag messy columns
    proactively, independent of whether any SQL filters on them yet."""
    values = _sample_column_values(db_path, table, column)
    normalized_groups: dict[str, set[str]] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        normalized_groups.setdefault(_normalize_text(value), set()).add(value)
    return any(len(variants) > 1 for variants in normalized_groups.values())


def _find_messy_filter_match(sql: str, db_path: str) -> dict | None:
    """Core detection shared by check_messy_categorical_filter() (prose
    message) and get_messy_filter_details() (structured, for the
    interactive resolution UI). Returns the first messy exact-match filter
    found as {"table", "column", "value", "span", "matched_text"] - "span"
    is the (start, end) character offset of the matched `column = 'value'`
    text in `sql`, so a resolution can be applied with a precise substring
    replace rather than re-matching against possibly-different SQL later.
    Returns None if no messy filter is found."""
    tables = _referenced_tables(sql, db_path)
    if not tables:
        return None

    columns_by_table = {table: get_table_columns(db_path, table) for table in tables}
    # column name (lowercased) -> tables that have a text column with that name
    column_to_tables: dict[str, list[str]] = {}
    for table, columns in columns_by_table.items():
        for col_name, col_type, _ in columns:
            if is_text_type(col_type):
                column_to_tables.setdefault(col_name.lower(), []).append(table)

    for match in _EXACT_MATCH_FILTER.finditer(sql):
        qualifier, column, value = match.groups()
        candidate_tables = column_to_tables.get(column.lower())
        if not candidate_tables:
            continue

        # If the qualifier is itself a real table name (not an alias),
        # narrow to it; aliases are otherwise left ambiguous and every
        # candidate table is checked.
        if qualifier and qualifier.lower() in {t.lower() for t in tables}:
            candidate_tables = [t for t in candidate_tables if t.lower() == qualifier.lower()] or candidate_tables

        for table in candidate_tables:
            # Recover the real declared column name (case) for the message.
            real_column = next(
                col_name for col_name, col_type, _ in columns_by_table[table] if col_name.lower() == column.lower()
            )
            if is_messy_column(db_path, table, real_column):
                return {
                    "table": table,
                    "column": real_column,
                    "value": value,
                    "span": match.span(),
                    "matched_text": match.group(0),
                }

    return None


def check_messy_categorical_filter(sql: str, db_path: str = DEFAULT_DB_PATH) -> str | None:
    """Flag exact-match filters on columns that turn out to have
    inconsistent casing/whitespace in the active dataset, where the
    comparison isn't normalized on both sides."""
    match = _find_messy_filter_match(sql, db_path)
    if not match:
        return None

    return (
        f"This query filters on '{match['table']}.{match['column']}' with an "
        f"exact match ({match['column']} = '{match['value']}'), but that column "
        f"contains inconsistent casing/whitespace in the data "
        f"(different raw values that normalize to the same thing, "
        f"e.g. 'West'/'west'/'WEST'). An exact match like this will "
        f"silently miss the other variants and undercount. "
        f"Normalize both sides of the comparison, e.g. "
        f"LOWER(TRIM({match['column']})) = LOWER('{match['value']}'), or "
        f"explicitly account for all known variants."
    )


def get_messy_filter_details(sql: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Structured version of check_messy_categorical_filter(), for the
    interactive resolution UI: which table/column/value is affected,
    without formatting it into prose."""
    return _find_messy_filter_match(sql, db_path)
