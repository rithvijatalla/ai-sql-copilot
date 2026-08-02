"""
Folds human grades into tests/benchmark_results.json/.csv (produced by
run_benchmark.py) and computes the final guarded-vs-unguarded summary
statistics.

Two grading paths feed into a single "verdict" per question:

  - messy_categorical_filter / granularity_mismatch: verdict comes from
    run_benchmark.py's automated ground-truth comparison
    (unguarded_correct). Rows where that comparison couldn't be made
    (unguarded's answer wasn't a comparable scalar) stay ungradable unless
    a human grade is supplied for them too - see MANUAL_GRADES below,
    which covers a couple of those alongside the undefined_metric/
    out_of_scope questions.
  - undefined_metric / out_of_scope: verdict comes entirely from
    MANUAL_GRADES, keyed by question number, one of "silently wrong" or
    "acceptable" (matches the values used when grading - see the
    benchmark write-up for what "acceptable" means: guarded still fired a
    clarification in every one of these questions, so "acceptable" marks
    cases where that clarification looks like it may not have been
    necessary, not cases where anything was missed).

Run again with an updated MANUAL_GRADES dict to re-finalize after
re-grading.
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "tests" / "benchmark_results.json"
RESULTS_CSV = BASE_DIR / "tests" / "benchmark_results.csv"
SUMMARY_MD = BASE_DIR / "tests" / "benchmark_summary.md"

# Human grades collected 2026-08-01. "silently wrong" = the unguarded
# answer was a misleading default a user could easily mistake for correct.
# "acceptable" = the unguarded answer, while technically ungrounded in an
# explicit metric/scope, was a reasonable-enough default that a user
# wouldn't likely be misled. #25/#27 are graded per the benchmark write-up
# discussion: #25's per-row-repeated non-total is misleading; #27 is
# "acceptable" in the narrow sense that the one correct scalar it computed
# is right, but marked separately below since guarded avoided the question
# entirely rather than warning on a bad answer.
MANUAL_GRADES = {
    8: "silently wrong",
    9: "silently wrong",
    10: "silently wrong",
    11: "acceptable",
    12: "silently wrong",
    13: "acceptable",
    14: "silently wrong",
    15: "silently wrong",
    16: "acceptable",
    17: "silently wrong",
    18: "acceptable",
    19: "acceptable",
    20: "acceptable",
    25: "silently wrong",
    27: "acceptable",
}

# Questions called out as possible guardrail over-triggers: guarded fired
# (asked for clarification) but the human grade says the unguarded answer
# was fine - a clear, bounded, single-table request.
OVER_TRIGGER_FLAGGED = {16, 18, 19, 20}

# Distinct pipeline finding, not a guardrail miss: #14's unguarded SQL used
# SQLite's real system clock ('now'), which is past the demo dataset's 2025
# cutoff, so "this year" silently matched zero rows. The pipeline never
# tells the model what "today" is.
DATE_CLOCK_BUG_QUESTIONS = {14}

ISSUE_CATEGORIES = ["messy_categorical_filter", "granularity_mismatch", "undefined_metric", "out_of_scope"]
INFORMATIONAL_CATEGORIES = ["clean", "edge_case"]


def classify_outcome(row: dict) -> str:
    if row["guarded_error"]:
        return "ERROR"
    if row["guarded_clarification"]:
        return "CLARIFIED"
    if row["guarded_warning"]:
        return "WARNED"
    return "ANSWERED"


def compute_verdict(row: dict) -> str | None:
    """Returns 'wrong', 'correct', or None (ungradable / not applicable)."""
    num = row["num"]
    category = row["category"]

    if category in ("messy_categorical_filter", "granularity_mismatch"):
        if num in MANUAL_GRADES:
            return "wrong" if MANUAL_GRADES[num] == "silently wrong" else "correct"
        if row["unguarded_correct"] is None:
            return None
        return "correct" if row["unguarded_correct"] else "wrong"

    if category in ("undefined_metric", "out_of_scope"):
        grade = MANUAL_GRADES.get(num)
        if grade is None:
            return None
        return "wrong" if grade == "silently wrong" else "correct"

    return None


def main() -> None:
    rows = json.loads(RESULTS_JSON.read_text())

    for row in rows:
        num = row["num"]
        if num in MANUAL_GRADES:
            row["manual_grade"] = MANUAL_GRADES[num]
        row["guarded_outcome"] = classify_outcome(row)
        row["final_verdict"] = compute_verdict(row)
        row["possible_over_trigger"] = num in OVER_TRIGGER_FLAGGED
        row["date_clock_bug"] = num in DATE_CLOCK_BUG_QUESTIONS

    RESULTS_JSON.write_text(json.dumps(rows, indent=2, default=str))

    flat_fields = [
        "num", "category", "question", "expected_behavior", "note",
        "guarded_outcome", "guarded_warning", "guarded_clarification", "guarded_error",
        "unguarded_sql", "unguarded_error",
        "ground_truth_sql", "ground_truth_value", "unguarded_actual_value",
        "unguarded_correct", "guarded_flagged", "needs_manual_grading", "manual_grade",
        "final_verdict", "possible_over_trigger", "date_clock_bug",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_lines = build_summary(rows)
    SUMMARY_MD.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nSaved {RESULTS_JSON}, {RESULTS_CSV}, {SUMMARY_MD}", file=sys.stderr)


def build_summary(rows: list[dict]) -> list[str]:
    lines = []
    lines.append("# Guarded vs. unguarded benchmark - final results")
    lines.append("")
    lines.append(f"40 questions total, run against the demo dataset (`db/analytics.db`).")
    lines.append("")

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | N | Gradable | Unguarded wrong | Guarded caught | Guarded missed |")
    lines.append("|---|---|---|---|---|---|")

    total_gradable = 0
    total_wrong = 0
    total_caught = 0
    total_missed = 0

    for category in ISSUE_CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == category]
        graded = [r for r in cat_rows if r["final_verdict"] is not None]
        wrong = [r for r in graded if r["final_verdict"] == "wrong"]
        caught = [r for r in wrong if r["guarded_flagged"] or r["guarded_outcome"] == "CLARIFIED"]
        missed = [r for r in wrong if r not in caught]

        total_gradable += len(graded)
        total_wrong += len(wrong)
        total_caught += len(caught)
        total_missed += len(missed)

        lines.append(
            f"| {category} | {len(cat_rows)} | {len(graded)} | "
            f"{len(wrong)} ({_pct(len(wrong), len(graded))}) | "
            f"{len(caught)}/{len(wrong) or 0} | {len(missed)} |"
        )

    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(
        f"- **Without guardrails, {_pct(total_wrong, total_gradable)} of gradable "
        f"questions ({total_wrong}/{total_gradable}) got a wrong or misleading answer, "
        f"delivered silently with no indication anything was off.**"
    )
    lines.append(
        f"- **With guardrails, {_pct(total_caught, total_wrong)} of those wrong answers "
        f"({total_caught}/{total_wrong}) were caught** - either flagged with a warning "
        f"(messy_categorical_filter, granularity_mismatch) or intercepted with a "
        f"clarifying question before any SQL ran (undefined_metric, out_of_scope)."
    )
    if total_missed:
        missed_nums = [r["num"] for r in rows if r["final_verdict"] == "wrong"
                        and not (r["guarded_flagged"] or r["guarded_outcome"] == "CLARIFIED")]
        lines.append(f"- Guardrails missed {total_missed} wrong answer(s): {missed_nums}")
    else:
        lines.append(
            "- Guardrails missed **0** of the wrong answers in this eval set - every "
            "case a human or ground-truth query identified as wrong was either "
            "flagged or avoided."
        )

    lines.append("")
    lines.append("## Caveats and other findings (read before citing the headline number)")
    lines.append("")
    lines.append(
        "- **2 granularity_mismatch questions (#25, #27) were graded manually**, not "
        "by the automated ground-truth comparison, because the unguarded answer wasn't "
        "a single comparable number (a full per-order table instead of a total). "
        "#25's unguarded/guarded SQL is identical - the guardrail only adds a warning, "
        "it doesn't fix the query - and repeats one channel's *all-time* total on every "
        "matching row rather than computing the grand total asked for; graded "
        "'silently wrong'. #27 is graded 'acceptable' because the one correct number it "
        "computed is actually right, though buried in a 4,000-row table - but note "
        "guarded didn't warn-and-answer here, it declined to generate SQL at all."
    )
    lines.append(
        "- **Possible guardrail over-triggering on out_of_scope**: questions "
        f"{sorted(OVER_TRIGGER_FLAGGED)} were graded 'acceptable' - clear, bounded, "
        "single-table requests (e.g. 'dump the entire orders table', 'show every "
        "support ticket in full detail') that still triggered a clarifying question. "
        "4 of the 6 out_of_scope questions in this eval set fall into this bucket, "
        "which suggests check_out_of_scope's current framing may skew toward flagging "
        "any full-table SELECT * as unbounded, even when the table itself is a "
        "reasonable, bounded unit of data. Worth tightening if reducing "
        "false-positive friction matters more than being conservative."
    )
    lines.append(
        "- **#14 ('best-performing channel this year') is a separate pipeline finding, "
        "not a guardrail failure**: the unguarded SQL used SQLite's real system clock "
        "(`strftime('%Y','now')`) to resolve \"this year\", which falls after the demo "
        "dataset's 2025 cutoff, so it silently returned zero rows. The pipeline never "
        "tells the model what \"today\" is - a relative-date resolution gap, unrelated "
        "to any of the four guardrails, that would affect the guarded path too if the "
        "question had been unambiguous enough to reach SQL generation."
    )

    lines.append("")
    lines.append("## Informational only (not in the headline number)")
    lines.append("")
    lines.append(
        "- **clean** (7 questions, all designed to be unambiguous): 0 guardrail "
        "false-positives. No regression here."
    )
    lines.append(
        "- **edge_case** (6 questions, a mix of regression checks and deliberately "
        "borderline phrasing - not a clean-question set): 1 of 6 triggered a "
        "clarification (#37, \"Which customers are most valuable?\"), which is the "
        "question this eval set flagged in advance as genuinely ambiguous and "
        "expected to reasonably go either way - not a false positive. The other 5 "
        "edge cases (NULL handling, a 'top' phrasing with an explicit metric, "
        "zero-order customers, an average with a zero-order denominator, a "
        "'top N by count' phrasing) all answered without any guardrail firing, as "
        "expected."
    )

    return lines


def _pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.0f}%" if d else "n/a"


if __name__ == "__main__":
    main()
