"""
Interactive, structured guardrail resolution.

detect_issues() runs all four guardrails against a question up front -
rather than stopping at the first one that fires, the way ask() does - and
returns every issue found, each with a resolution option where one makes
sense. resolve_and_ask() takes the user's choices and folds them into a
final question/SQL/execution pass, so the user never has to retype
anything.

This sits alongside pipeline.query_engine.ask() rather than replacing it:
ask() (and ask_unguarded()) remain exactly as they were, used by
tests/run_benchmark.py and guardrails/test_checks.py. This module backs the
interactive resolution UI in app/app.py specifically.

granularity_mismatch/messy_categorical_filter only make sense to check
against generated SQL, not the raw question - so detect_issues() generates
a "draft" query unconditionally, purely to run those two checks against.
That draft is thrown away and regenerated (possibly augmented by the
user's resolution choices) once everything is resolved; it costs one extra
SQL-generation call even on a perfectly clean question, which is the
tradeoff for surfacing every issue in one pass instead of round-tripping
per issue.

Resolution values, keyed by issue type (matching DetectedIssues.issues):
  - undefined_metric: one of the candidate metric label strings.
  - messy_categorical_filter: "normalized" or "exact".
  - granularity_mismatch: "aggregate_finer" or "keep_finer".
  - out_of_scope: no resolution value - see note on resolve_and_ask().
"""

from dataclasses import dataclass, field

import anthropic

from guardrails.checks import (
    analyze_undefined_metric,
    check_out_of_scope,
    get_granularity_mismatch_details,
    get_messy_filter_details,
)
from pipeline.query_engine import DEFAULT_DB_PATH, execute_sql_readonly, generate_sql
from pipeline.schema_utils import get_schema_description

_GRAIN_ORDER = {"daily": 1, "weekly": 2, "monthly": 3}


@dataclass
class DetectedIssues:
    question: str
    db_path: str
    schema_description: str
    draft_sql: str | None
    draft_sql_error: str | None
    issues: dict = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


def detect_issues(question: str, db_path: str = DEFAULT_DB_PATH) -> DetectedIssues:
    """Run all four guardrails against `question` and collect every issue
    found (unlike ask(), which stops at the first pre-generation issue)."""
    schema_description = get_schema_description(db_path)

    draft_sql = None
    draft_sql_error = None
    try:
        draft_sql = generate_sql(question, schema_description)
    except (anthropic.AnthropicError, TypeError) as e:
        draft_sql_error = str(e)

    issues: dict = {}

    metric_result = analyze_undefined_metric(question, schema_description)
    if metric_result:
        issues["undefined_metric"] = {
            "message": metric_result["message"],
            "candidates": metric_result["candidates"],
        }

    scope_message = check_out_of_scope(question, schema_description)
    if scope_message:
        issues["out_of_scope"] = {"message": scope_message}

    if draft_sql:
        granularity = get_granularity_mismatch_details(draft_sql, db_path)
        if granularity:
            issues["granularity_mismatch"] = {
                **granularity,
                "message": (
                    f"This question joins '{granularity['table_a']}' ({granularity['grain_a']} grain) "
                    f"with '{granularity['table_b']}' ({granularity['grain_b']} grain) - these don't line "
                    f"up without reconciling, or the numbers may double-count or misalign."
                ),
            }

        messy = get_messy_filter_details(draft_sql, db_path)
        if messy:
            issues["messy_categorical_filter"] = {
                **messy,
                "message": (
                    f"Filtering on '{messy['table']}.{messy['column']}' with an exact match on "
                    f"'{messy['value']}' - this column has inconsistent casing/whitespace in the data, "
                    f"so an exact match may silently miss some rows."
                ),
            }

    return DetectedIssues(
        question=question,
        db_path=db_path,
        schema_description=schema_description,
        draft_sql=draft_sql,
        draft_sql_error=draft_sql_error,
        issues=issues,
    )


def granularity_resolution_options(details: dict) -> list[tuple[str, str]]:
    """Returns [(value, label), ...] for the granularity_mismatch radio,
    ordered so the recommended option (aggregate the finer-grain table up)
    comes first. Values are what resolve_and_ask() expects back.

    Disaggregating the coarser table down isn't offered as an option -
    there's no data to invent (you can't turn a monthly total into daily
    figures), so the real choice is between reconciling the grain properly
    or explicitly accepting the mismatch."""
    a_is_finer = _GRAIN_ORDER[details["grain_a"]] < _GRAIN_ORDER[details["grain_b"]]
    finer_table, finer_col, finer_grain = (
        (details["table_a"], details["col_a"], details["grain_a"])
        if a_is_finer
        else (details["table_b"], details["col_b"], details["grain_b"])
    )
    coarser_table, coarser_grain = (
        (details["table_b"], details["grain_b"]) if a_is_finer else (details["table_a"], details["grain_a"])
    )
    return [
        (
            "aggregate_finer",
            f"Aggregate '{finer_table}' up to {coarser_grain} grain to match '{coarser_table}' (recommended)",
        ),
        (
            "keep_finer",
            f"Keep '{finer_table}' at {finer_grain} grain - '{coarser_table}'s value will repeat across "
            f"every {finer_grain} row within the same {coarser_grain} period, so don't sum it",
        ),
    ]


def messy_filter_resolution_options(details: dict) -> list[tuple[str, str]]:
    """Returns [(value, label), ...] for the messy_categorical_filter
    radio. Values are what resolve_and_ask() expects back."""
    return [
        (
            "normalized",
            f"Include all case/whitespace variants of '{details['value']}' (recommended)",
        ),
        (
            "exact",
            f"Exact match on '{details['value']}' only",
        ),
    ]


def _augment_question_for_metric(question: str, chosen_metric: str) -> str:
    return f"{question.rstrip('?.! ')}, ranked by {chosen_metric}?"


def _augment_question_for_granularity(question: str, details: dict, resolution: str) -> str:
    a_is_finer = _GRAIN_ORDER[details["grain_a"]] < _GRAIN_ORDER[details["grain_b"]]
    finer_table, finer_col = (
        (details["table_a"], details["col_a"]) if a_is_finer else (details["table_b"], details["col_b"])
    )
    coarser_table, coarser_grain = (
        (details["table_b"], details["grain_b"]) if a_is_finer else (details["table_a"], details["grain_a"])
    )

    if resolution == "aggregate_finer":
        instruction = (
            f"Reconcile the grain mismatch between '{details['table_a']}' and '{details['table_b']}' by "
            f"aggregating '{finer_table}' up to {coarser_grain} grain (group by a truncated "
            f"'{finer_table}.{finer_col}', e.g. via strftime) before comparing to '{coarser_table}'."
        )
    else:
        instruction = (
            f"Keep '{finer_table}' at its native grain when joining to '{coarser_table}' - do not aggregate "
            f"either side. Note that '{coarser_table}'s value will repeat across every matching "
            f"'{finer_table}' row within the same {coarser_grain} period; do not SUM it in that shape."
        )

    return f"{question.rstrip('?.! ')}. {instruction}"


def _apply_messy_filter_resolution(sql: str, details: dict, resolution: str) -> str:
    if resolution != "normalized":
        return sql

    column = details["column"]
    value = details["value"]
    replacement = f"LOWER(TRIM({column})) = LOWER('{value}')"

    start, end = details["span"]
    matched_text = details.get("matched_text")
    # Prefer the recorded span if the SQL text is unchanged there; the
    # regenerated SQL (after a metric/granularity augmentation) is usually
    # structurally the same for an unrelated filter, but fall back to a
    # plain substring search if it shifted, rather than silently no-op-ing.
    if matched_text and sql[start:end] == matched_text:
        return sql[:start] + replacement + sql[end:]
    if matched_text and matched_text in sql:
        return sql.replace(matched_text, replacement, 1)
    return sql


def resolve_and_ask(detected: DetectedIssues, resolutions: dict) -> dict:
    """Fold the user's resolution choices into a final question/SQL pass
    and execute it. `resolutions` is keyed by issue type (matching
    detected.issues), with values as documented in the module docstring.

    out_of_scope has no resolution value and isn't handled here - the
    caller (app.py) doesn't offer a Run Query action while that issue is
    present, since the intended fix is rephrasing the question, not
    picking a parameter.

    If `resolutions` needs neither a metric nor a granularity augmentation
    (the common no-issues-detected case, or a messy-filter-only fix), the
    draft SQL from detect_issues() is reused instead of paying for a second,
    identical generate_sql() call.
    """
    result = {"question": detected.question, "sql": None, "result": None, "error": None}

    needs_regeneration = "undefined_metric" in resolutions or "granularity_mismatch" in resolutions
    if not needs_regeneration and detected.draft_sql:
        sql = detected.draft_sql
    else:
        question = detected.question
        if "undefined_metric" in resolutions:
            question = _augment_question_for_metric(question, resolutions["undefined_metric"])
        if "granularity_mismatch" in resolutions:
            question = _augment_question_for_granularity(
                question, detected.issues["granularity_mismatch"], resolutions["granularity_mismatch"]
            )

        try:
            sql = generate_sql(question, detected.schema_description)
        except (anthropic.AnthropicError, TypeError) as e:
            result["error"] = f"Failed to generate SQL: {e}"
            return result

    if "messy_categorical_filter" in resolutions:
        sql = _apply_messy_filter_resolution(
            sql, detected.issues["messy_categorical_filter"], resolutions["messy_categorical_filter"]
        )

    result["sql"] = sql

    try:
        result["result"] = execute_sql_readonly(sql, detected.db_path)
    except Exception as e:
        result["error"] = str(e)

    return result
