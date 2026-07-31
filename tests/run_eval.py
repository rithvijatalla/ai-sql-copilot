"""
Runs every question in eval_questions.py through ask() and prints a report
for manual grading.

This script does NOT auto-grade anything - "answered" doesn't mean correct,
and "clarified" doesn't mean it should have clarified. It only classifies
what ask() actually did (answered / clarified / warned / errored) and lays
that next to what you expected, so you can make the actual judgment call
yourself.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "pipeline"))

from eval_questions import EVAL_QUESTIONS
from query_engine import ask

COL_WIDTHS = {"num": 3, "category": 20, "expected": 10, "actual": 10, "question": 55}


def classify_outcome(result: dict) -> str:
    if result["error"]:
        return "ERROR"
    if result["clarification_needed"]:
        return "CLARIFIED"
    if result["warning"]:
        return "WARNED"
    return "ANSWERED"


def truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_summary_table(rows: list[tuple]) -> None:
    header = (
        f"{'#':<{COL_WIDTHS['num']}} "
        f"{'Category':<{COL_WIDTHS['category']}} "
        f"{'Expected':<{COL_WIDTHS['expected']}} "
        f"{'Actual':<{COL_WIDTHS['actual']}} "
        f"Question"
    )
    print(header)
    print("-" * len(header))
    for num, category, expected, actual, question in rows:
        print(
            f"{num:<{COL_WIDTHS['num']}} "
            f"{category:<{COL_WIDTHS['category']}} "
            f"{expected:<{COL_WIDTHS['expected']}} "
            f"{actual:<{COL_WIDTHS['actual']}} "
            f"{truncate(question, COL_WIDTHS['question'])}"
        )


def print_details(details: list[tuple]) -> None:
    for num, item, result, actual in details:
        print(f"\n[{num}] category={item['category']}  expected={item['expected_behavior']}  actual={actual}")
        print(f"Question: {item['question']}")
        print(f"Note: {item['note']}")

        if result["error"]:
            print(f"Error: {result['error']}")
        if result["clarification_needed"]:
            print(f"Clarification requested:\n{result['clarification_needed']}")
        if result["sql"]:
            print(f"SQL:\n{result['sql']}")
        if result["warning"]:
            print(f"Warning: {result['warning']}")
        if result["result"] is not None:
            print(f"Result:\n{result['result']}")

        print("-" * 100)


def main() -> None:
    rows = []
    details = []

    for num, item in enumerate(EVAL_QUESTIONS, start=1):
        result = ask(item["question"])
        actual = classify_outcome(result)
        rows.append((num, item["category"], item["expected_behavior"], actual, item["question"]))
        details.append((num, item, result, actual))

    print_summary_table(rows)
    print()
    print("=" * 100)
    print("DETAILS (for manual grading)")
    print("=" * 100)
    print_details(details)


if __name__ == "__main__":
    main()
