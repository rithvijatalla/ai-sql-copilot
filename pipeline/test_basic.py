"""Manual smoke test for the NL-to-SQL pipeline: runs a few easy questions end to end."""

from query_engine import ask

QUESTIONS = [
    "What was total revenue in 2024?",
    "How many customers are on the Pro plan?",
    "What is the average order revenue?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"Question: {question}")
        outcome = ask(question)
        print(f"SQL: {outcome['sql']}")
        if outcome["error"]:
            print(f"Error: {outcome['error']}")
        else:
            print(f"Result:\n{outcome['result']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
