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

5. check_bad_join (post-generation, static analysis + data sample)
   For every simple `table.column = table.column` condition inside a JOIN's
   ON clause, checks whether the two columns look like a real relationship
   rather than an arbitrary/coincidental match: do the names suggest a link
   (matching, or one following the `<table>_id` convention), and do the
   actual values meaningfully overlap. Two columns that just happen to
   share a generic name (e.g. both called "id") but reference unrelated
   entities are exactly the case name-matching alone would miss and
   overlap alone would sometimes miss too (small integer PK ranges overlap
   coincidentally) - this check combines both signals, plus a basic
   declared-type compatibility check, and offers the best-overlapping
   related-name column pair as a suggested fix when one exists.
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


# ---------------------------------------------------------------------------
# check_bad_join - static SQL analysis + data-driven relationship validation
# ---------------------------------------------------------------------------

# Column names too generic to trust as evidence of a real relationship on
# their own (e.g. two unrelated tables both happening to have an "id"
# column) - a match on one of these needs value-overlap confirmation.
_GENERIC_JOIN_COLUMN_NAMES = {"id", "key", "code", "name", "type", "value", "status"}

_TABLE_OR_ALIAS_REF = re.compile(r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?", re.IGNORECASE)
_JOIN_ON_CLAUSE = re.compile(
    r"\bJOIN\s+\w+(?:\s+(?:AS\s+)?\w+)?\s+ON\s+(.+?)"
    r"(?=\bJOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\)|$)",
    re.IGNORECASE | re.DOTALL,
)
_EQUI_JOIN_CONDITION = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)")


def _is_numeric_type(declared_type: str) -> bool:
    upper = (declared_type or "").upper()
    return any(hint in upper for hint in ("INT", "REAL", "FLOA", "DOUB", "NUM"))


def _table_alias_map(sql: str, known_tables: list[str]) -> dict[str, str]:
    """Map every table-name/alias reference in FROM/JOIN clauses
    (lowercased) to its real table name, for resolving qualified column
    references like `o.channel` back to `orders`."""
    known_lower = {t.lower(): t for t in known_tables}
    alias_map: dict[str, str] = {}
    for match in _TABLE_OR_ALIAS_REF.finditer(sql):
        ref, alias = match.groups()
        real_table = known_lower.get(ref.lower())
        if not real_table:
            continue
        alias_map[ref.lower()] = real_table
        if alias:
            alias_map[alias.lower()] = real_table
    return alias_map


def _extract_equi_join_pairs(sql: str, db_path: str) -> list[dict]:
    """Find simple `qualifier.column = qualifier.column` conditions inside
    JOIN ... ON clauses, resolved to real (table, column) pairs on each
    side. Conditions involving a function call (e.g.
    `strftime(...) = ms.month`, the pattern check_granularity_mismatch
    treats as a deliberately reconciled join), OR, or an
    unqualified/unresolvable side are skipped - not something this check
    can evaluate reliably, and not a plain "does this column match that
    column" claim to begin with."""
    known_tables = get_table_names(db_path)
    alias_map = _table_alias_map(sql, known_tables)

    pairs = []
    for on_clause_match in _JOIN_ON_CLAUSE.finditer(sql):
        on_clause = on_clause_match.group(1)
        for cond_match in _EQUI_JOIN_CONDITION.finditer(on_clause):
            qual_a, col_a, qual_b, col_b = cond_match.groups()
            table_a = alias_map.get(qual_a.lower())
            table_b = alias_map.get(qual_b.lower())
            if not table_a or not table_b or table_a == table_b:
                continue
            pairs.append({"table_a": table_a, "col_a": col_a, "table_b": table_b, "col_b": col_b})
    return pairs


def _column_relates_to_table(column_name: str, table_name: str) -> bool:
    """True if the table's name (singular or as-is) appears in the column
    name - the standard `<table>_id`/`<table>_ref` FK-naming convention,
    e.g. "customer_id" relating to table "customers"."""
    col = column_name.lower()
    table = table_name.lower()
    table_singular = table[:-1] if table.endswith("s") else table
    return table_singular in col or table in col


def _names_relate(col_a: str, table_a: str, col_b: str, table_b: str) -> bool | None:
    """True: the column names clearly suggest an intentional link. False:
    they clearly don't. None: ambiguous - matching names, but too generic
    (e.g. both just called "id") to be confident without checking whether
    the actual values overlap too."""
    same_name = col_a.lower() == col_b.lower()
    generic = col_a.lower() in _GENERIC_JOIN_COLUMN_NAMES

    if same_name and not generic:
        return True
    if _column_relates_to_table(col_a, table_b) or _column_relates_to_table(col_b, table_a):
        return True
    if same_name and generic:
        return None
    return False


def _sample_value_set(db_path: str, table: str, column: str) -> set[str]:
    values = _sample_column_values(db_path, table, column)
    return {str(v).strip().lower() for v in values if v is not None}


def _overlap_ratio(values_a: set, values_b: set) -> float:
    if not values_a or not values_b:
        return 0.0
    return len(values_a & values_b) / min(len(values_a), len(values_b))


def _suggest_join_columns(db_path: str, table_a: str, table_b: str) -> dict | None:
    """Search every column pair between the two tables for one that looks
    like the real relationship (related names, confirmed by meaningful
    value overlap), to offer as a fix. Returns the best-overlapping
    candidate, or None if nothing clearly better is found."""
    columns_a = get_table_columns(db_path, table_a)
    columns_b = get_table_columns(db_path, table_b)

    best = None
    for col_a, _, _ in columns_a:
        for col_b, _, _ in columns_b:
            if _names_relate(col_a, table_a, col_b, table_b) is not True:
                continue
            overlap = _overlap_ratio(
                _sample_value_set(db_path, table_a, col_a),
                _sample_value_set(db_path, table_b, col_b),
            )
            if overlap < 0.3:
                continue
            if best is None or overlap > best["overlap_pct"] / 100:
                best = {"col_a": col_a, "col_b": col_b, "overlap_pct": round(overlap * 100, 1)}
    return best


def _find_bad_join(sql: str, db_path: str) -> dict | None:
    """Core detection shared by check_bad_join() (prose message) and
    get_bad_join_details() (structured, for the interactive resolution
    UI). Returns the first suspicious join found, or None if every simple
    equi-join in the SQL looks structurally sound."""
    for pair in _extract_equi_join_pairs(sql, db_path):
        table_a, col_a, table_b, col_b = pair["table_a"], pair["col_a"], pair["table_b"], pair["col_b"]

        columns_a = get_table_columns(db_path, table_a)
        columns_b = get_table_columns(db_path, table_b)
        type_a = next((t for c, t, _ in columns_a if c.lower() == col_a.lower()), None)
        type_b = next((t for c, t, _ in columns_b if c.lower() == col_b.lower()), None)
        if type_a is None or type_b is None:
            continue  # couldn't resolve to a real column - nothing to evaluate

        type_mismatch = _is_numeric_type(type_a) != _is_numeric_type(type_b)
        related = _names_relate(col_a, table_a, col_b, table_b)
        values_a = _sample_value_set(db_path, table_a, col_a)
        values_b = _sample_value_set(db_path, table_b, col_b)
        overlap = _overlap_ratio(values_a, values_b)
        low_cardinality = min(len(values_a), len(values_b)) < 5

        # Built in the branch that actually triggered, not generically from
        # every signal computed above - e.g. citing "only 100% overlap" as
        # evidence of a *bad* join reads as self-contradictory, and it's
        # exactly what happens when two small sequential integer id columns
        # (like order_id and customer_id) coincidentally overlap despite
        # naming that doesn't relate them at all. The name mismatch is the
        # real reason in that case, not overlap.
        reasons = []
        if related is False:
            reasons.append(f"the names don't suggest a link between '{table_a}' and '{table_b}'")
        elif related is None and low_cardinality:
            reasons.append(
                f"both are just called '{col_a}', a name too generic to confirm a relationship on its "
                f"own, and there isn't enough distinct data on either side to confirm a match from values alone"
            )
        elif related is None and overlap < 0.2:
            reasons.append(
                f"both are just called '{col_a}', a name too generic to confirm a relationship on its "
                f"own, and only {round(overlap * 100, 1)}% of their values actually match"
            )
        if type_mismatch:
            reasons.append(f"their declared types don't match ({type_a} vs {type_b})")

        if not reasons:
            continue

        return {
            "table_a": table_a,
            "col_a": col_a,
            "table_b": table_b,
            "col_b": col_b,
            "overlap_pct": round(overlap * 100, 1),
            "type_mismatch": type_mismatch,
            "reasons": reasons,
            "suggestion": _suggest_join_columns(db_path, table_a, table_b),
        }

    return None


def check_bad_join(sql: str, db_path: str = DEFAULT_DB_PATH) -> str | None:
    """Flag SQL that joins two tables on columns that don't look like a
    real relationship - unrelated names, barely-overlapping values, and/or
    mismatched declared types - rather than a genuine primary/foreign-key
    link."""
    bad_join = _find_bad_join(sql, db_path)
    if not bad_join:
        return None

    message = (
        f"This query joins '{bad_join['table_a']}.{bad_join['col_a']}' to "
        f"'{bad_join['table_b']}.{bad_join['col_b']}', but these columns don't look like a real "
        f"relationship - {'; '.join(bad_join['reasons'])}. Joining on unrelated columns can silently "
        f"return wrong, missing, or duplicated rows."
    )

    suggestion = bad_join["suggestion"]
    if suggestion:
        message += (
            f" '{bad_join['table_a']}.{suggestion['col_a']}' and '{bad_join['table_b']}.{suggestion['col_b']}' "
            f"look like a better match ({suggestion['overlap_pct']}% overlap)."
        )

    return message


def get_bad_join_details(sql: str, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    """Structured version of check_bad_join(), for the interactive
    resolution UI: which tables/columns are involved and the suggested
    replacement (if any), without formatting it into prose."""
    return _find_bad_join(sql, db_path)
