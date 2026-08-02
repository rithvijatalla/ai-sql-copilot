"""
Benchmarks the guarded pipeline (ask()) against the unguarded one
(ask_unguarded()) over the full eval set, to produce a real measured
statistic: how often does skipping the guardrails lead to a wrong or
misleading answer, and how often do the guardrails catch it?

Two grading paths, matching which categories carry a `ground_truth` field
in eval_questions.py:

  - messy_categorical_filter / granularity_mismatch: graded automatically.
    ground_truth["sql"] is executed directly against the database (bypassing
    the LLM entirely) to get the objectively correct value, which is
    compared to what ask_unguarded() actually returned. Because these two
    guardrails only *warn* (they don't block or rewrite the query - see
    guardrails/checks.py), ask() runs the identical SQL and gets the
    identical number; the only difference guarded adds is the warning
    itself. So "guarded caught it" here means the warning fired for a
    question whose unguarded answer was in fact wrong - not that the number
    came out different.

  - undefined_metric / out_of_scope: no reference query can settle whether
    an LLM's silently-chosen metric or scope was reasonable or misleading -
    that's a human judgment call. This script prints the guarded and
    unguarded outcomes side by side and leaves a manual_grade field (one of
    "silently wrong" / "acceptable" / "correctly caught") for a human to
    fill in, in both the printed table and the saved JSON/CSV.

  - clean / edge_case: not part of the accuracy measurement (no
    ground_truth, not sent for manual grading). Still run and recorded for
    two informational checks: whether ask_unguarded() executed without
    error, and whether ask() false-positived a guardrail on a question that
    was designed to be unambiguous.

Usage:
    python3 tests/run_benchmark.py

Results are saved to tests/benchmark_results.json and
tests/benchmark_results.csv. Run this, get the manual-grading rows graded by
a human, then run again with --grades to fold those grades into the final
summary (see finalize_benchmark.py... actually see the bottom of this file's
module docstring in the README for the two-pass workflow).
"""

import csv
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval_questions import EVAL_QUESTIONS
from pipeline.query_engine import DEFAULT_DB_PATH, ask, ask_unguarded

RESULTS_JSON = BASE_DIR / "tests" / "benchmark_results.json"
RESULTS_CSV = BASE_DIR / "tests" / "benchmark_results.csv"

GROUND_TRUTH_CATEGORIES = {"messy_categorical_filter", "granularity_mismatch"}
MANUAL_GRADING_CATEGORIES = {"undefined_metric", "out_of_scope"}


def classify_outcome(result: dict) -> str:
    if result["error"]:
        return "ERROR"
    if result["clarification_needed"]:
        return "CLARIFIED"
    if result["warning"]:
        return "WARNED"
    return "ANSWERED"


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _extract_scalar(df: pd.DataFrame | None):
    if df is not None and df.shape == (1, 1):
        return _json_safe(df.iloc[0, 0])
    return None


def _summarize_df(df: pd.DataFrame | None, max_rows: int = 5) -> dict | None:
    if df is None:
        return None
    if df.shape == (1, 1):
        return {"scalar": _extract_scalar(df)}
    records = [{k: _json_safe(v) for k, v in row.items()} for row in df.head(max_rows).to_dict(orient="records")]
    return {"shape": list(df.shape), "columns": list(df.columns), "sample_rows": records}


def compute_ground_truth_value(sql: str, db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return _extract_scalar(df)


def grade_against_ground_truth(unguarded_df: pd.DataFrame | None, truth_value, category: str):
    """Returns (is_correct: bool | None, actual_value). is_correct is None
    if there's nothing sensible to compare (e.g. unguarded errored, or a
    granularity_mismatch answer that isn't a scalar at all)."""
    actual = _extract_scalar(unguarded_df)
    if actual is None and category == "messy_categorical_filter" and unguarded_df is not None:
        # Listing-style answer (e.g. "List customers in the East region") -
        # row count is a legitimate proxy for "how many matched".
        actual = len(unguarded_df)

    if actual is None or truth_value is None:
        return None, actual

    try:
        return math.isclose(float(actual), float(truth_value), rel_tol=1e-6, abs_tol=1e-6), actual
    except (TypeError, ValueError):
        return actual == truth_value, actual


def run() -> list[dict]:
    rows = []
    total = len(EVAL_QUESTIONS)

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        question = item["question"]
        category = item["category"]
        print(f"[{i}/{total}] ({category}) {question}", file=sys.stderr)

        guarded = ask(question, db_path=DEFAULT_DB_PATH)
        unguarded = ask_unguarded(question, db_path=DEFAULT_DB_PATH)

        row = {
            "num": i,
            "category": category,
            "question": question,
            "expected_behavior": item["expected_behavior"],
            "note": item["note"],
            "guarded_outcome": classify_outcome(guarded),
            "guarded_sql": guarded["sql"],
            "guarded_clarification": guarded["clarification_needed"],
            "guarded_warning": guarded["warning"],
            "guarded_result_summary": _summarize_df(guarded["result"]),
            "guarded_error": guarded["error"],
            "unguarded_sql": unguarded["sql"],
            "unguarded_result_summary": _summarize_df(unguarded["result"]),
            "unguarded_error": unguarded["error"],
            "ground_truth_sql": None,
            "ground_truth_value": None,
            "unguarded_actual_value": None,
            "unguarded_correct": None,
            "guarded_flagged": None,
            "needs_manual_grading": False,
            "manual_grade": "",
        }

        ground_truth = item.get("ground_truth")
        if ground_truth:
            truth_value = compute_ground_truth_value(ground_truth["sql"], DEFAULT_DB_PATH)
            is_correct, actual_value = grade_against_ground_truth(unguarded["result"], truth_value, category)
            row.update(
                {
                    "ground_truth_sql": ground_truth["sql"],
                    "ground_truth_value": truth_value,
                    "unguarded_actual_value": actual_value,
                    "unguarded_correct": is_correct,
                    "guarded_flagged": bool(guarded["warning"]),
                }
            )
        elif category in MANUAL_GRADING_CATEGORIES:
            row["needs_manual_grading"] = True

        rows.append(row)

    return rows


def print_ground_truth_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("AUTO-GRADED CATEGORIES (messy_categorical_filter, granularity_mismatch)")
    print("=" * 100)

    for category in sorted(GROUND_TRUTH_CATEGORIES):
        cat_rows = [r for r in rows if r["category"] == category]
        graded = [r for r in cat_rows if r["unguarded_correct"] is not None]
        ungradable = [r for r in cat_rows if r["unguarded_correct"] is None]
        wrong = [r for r in graded if not r["unguarded_correct"]]
        caught = [r for r in wrong if r["guarded_flagged"]]
        missed = [r for r in wrong if not r["guarded_flagged"]]

        print(f"\n{category}  ({len(cat_rows)} questions, {len(graded)} auto-gradable)")
        if ungradable:
            print(f"  ungradable (unguarded answer wasn't a comparable scalar): "
                  + ", ".join(f"#{r['num']}" for r in ungradable))
        print(f"  unguarded wrong:            {len(wrong)}/{len(graded)}")
        print(f"  guarded flagged the wrong ones: {len(caught)}/{len(wrong) if wrong else 0}")
        if missed:
            print(f"  ** guardrail MISSED {len(missed)} wrong answer(s): "
                  + ", ".join(f"#{r['num']}" for r in missed))

        for r in cat_rows:
            status = "?" if r["unguarded_correct"] is None else ("CORRECT" if r["unguarded_correct"] else "WRONG")
            flag = "flagged" if r["guarded_flagged"] else "not flagged"
            print(
                f"  [{r['num']:>2}] unguarded={status:7s}  guarded={flag:11s}  "
                f"truth={r['ground_truth_value']}  actual={r['unguarded_actual_value']}  "
                f"-- {r['question']}"
            )


def print_manual_grading_rows(rows: list[dict]) -> None:
    manual_rows = [r for r in rows if r["needs_manual_grading"]]
    print("\n" + "=" * 100)
    print(f"NEEDS MANUAL GRADING ({len(manual_rows)} questions - undefined_metric, out_of_scope)")
    print("For each: mark manual_grade as 'silently wrong', 'acceptable', or 'correctly caught'.")
    print("=" * 100)

    for r in manual_rows:
        print(f"\n[{r['num']}] category={r['category']}  question: {r['question']}")
        print(f"    note: {r['note']}")
        print(f"    GUARDED   -> outcome={r['guarded_outcome']}")
        if r["guarded_clarification"]:
            print(f"                 clarification: {r['guarded_clarification']}")
        if r["guarded_warning"]:
            print(f"                 warning: {r['guarded_warning']}")
        if r["guarded_result_summary"]:
            print(f"                 result: {r['guarded_result_summary']}")
        print(f"    UNGUARDED -> sql: {r['unguarded_sql']}")
        if r["unguarded_error"]:
            print(f"                 error: {r['unguarded_error']}")
        if r["unguarded_result_summary"]:
            print(f"                 result: {r['unguarded_result_summary']}")
        print("    manual_grade: <-- fill in: silently wrong / acceptable / correctly caught")


def save_results(rows: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(rows, indent=2, default=str))

    flat_fields = [
        "num", "category", "question", "expected_behavior", "note",
        "guarded_outcome", "guarded_warning", "guarded_clarification", "guarded_error",
        "unguarded_sql", "unguarded_error",
        "ground_truth_sql", "ground_truth_value", "unguarded_actual_value",
        "unguarded_correct", "guarded_flagged", "needs_manual_grading", "manual_grade",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {RESULTS_JSON} and {RESULTS_CSV}", file=sys.stderr)


def main() -> None:
    rows = run()
    save_results(rows)
    print_ground_truth_summary(rows)
    print_manual_grading_rows(rows)
    print(
        "\n\nManual grading rows are marked above and in benchmark_results.csv "
        "(manual_grade column). Overall summary stats are withheld until "
        "those are graded - see the README/benchmark writeup for the "
        "finalization step.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
