"""
Hand-written evaluation set for the analyst copilot.

Each entry is a question paired with the behavior we expect the pipeline to
land on:

  - "answer"  - should generate SQL and execute cleanly, no clarification
                or warning.
  - "clarify" - should be caught by check_undefined_metric or
                check_out_of_scope and short-circuit before SQL generation
                (ask()["clarification_needed"] is populated).
  - "warn"    - should still generate and execute SQL, but either
                check_granularity_mismatch or check_messy_categorical_filter
                should flag it (ask()["warning"] is populated alongside a
                real result).

This file does not judge correctness - it records intent so a human (you)
can compare it against what tests/run_eval.py actually produced. Several
entries are deliberately borderline; the "note" field says so.
"""

EVAL_QUESTIONS = [
    # ------------------------------------------------------------------
    # Clean: clear metric, clear scope - should just answer.
    # ------------------------------------------------------------------
    {
        "question": "What was total revenue in 2024?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Clear metric (revenue), clear time bound. Baseline sanity check.",
    },
    {
        "question": "How many customers signed up in 2025?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Simple filtered count on customers.signup_date.",
    },
    {
        "question": "What is the average order revenue?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Single aggregate over orders, no ranking or scope ambiguity.",
    },
    {
        "question": "How many support tickets are currently open?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Simple filtered count on support_tickets.status.",
    },
    {
        "question": "What was the total marketing spend in 2024?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Single-table aggregate on marketing_spend, no orders join involved.",
    },
    {
        "question": "How many orders came through the Email channel?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Filtered count on orders.channel; channel value is explicit.",
    },
    # ------------------------------------------------------------------
    # Undefined metric: ranking/superlative language, no metric named.
    # ------------------------------------------------------------------
    {
        "question": "Who are the top customers?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Top\" with no metric - by revenue? order count? tickets?",
    },
    {
        "question": "Which region performs best?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Best\" is undefined - revenue, customer count, retention?",
    },
    {
        "question": "What's the leading marketing channel?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Leading\" by spend, by orders attributed, by revenue?",
    },
    {
        "question": "Show me the worst month.",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Worst\" with no metric and no dimension (revenue? tickets? spend?).",
    },
    # ------------------------------------------------------------------
    # Out of scope: unbounded dumps / full per-row detail, no aggregation.
    # ------------------------------------------------------------------
    {
        "question": "Show me everything in the database.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Textbook unbounded dump across all four tables.",
    },
    {
        "question": "List all customers with full details.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Every row, every column, no filter or aggregation.",
    },
    {
        "question": "Give me every order for every customer, with all their support tickets.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Unbounded multi-table row-level dump, no scoping.",
    },
    # ------------------------------------------------------------------
    # Granularity mismatch: orders (daily) vs marketing_spend (monthly)
    # without reconciling aggregation.
    # ------------------------------------------------------------------
    {
        "question": "Show each order alongside the marketing spend for its channel.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Per-order join against monthly spend repeats the same monthly total across every day in the month.",
    },
    {
        "question": "Compare daily order revenue to marketing spend by channel.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "\"Daily\" revenue explicitly invites a grain mismatch against monthly spend.",
    },
    {
        "question": "List every order next to that month's total marketing spend for its channel.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Row-level orders joined to a monthly figure - same duplication problem phrased differently.",
    },
    # ------------------------------------------------------------------
    # Messy categorical filter: region has inconsistent casing on purpose
    # ('West'/'west'/'WEST'/'W' all mean the same thing), so an unnormalized
    # exact-match filter on it silently undercounts. This is the specific
    # bug check_messy_categorical_filter was added to catch (confirmed in
    # testing: this exact question generated `region = 'West'` and returned
    # 134, missing every other-case row).
    # ------------------------------------------------------------------
    {
        "question": "How many customers are in the West region?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": (
            "The motivating bug for check_messy_categorical_filter: generated "
            "`WHERE region = 'West'` and returned 134, silently missing "
            "'west'/'WEST'/'W' rows. Should now still answer, but with a warning."
        ),
    },
    {
        "question": "List customers in the East region.",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Same undernormalized-filter risk as the West-region question, different region and phrasing (list vs. count).",
    },
    {
        "question": "How many customers signed up in the South region last year?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Combines a region filter with a date filter - checks the guardrail still fires when region isn't the only WHERE clause.",
    },
    # ------------------------------------------------------------------
    # Edge cases: NULL channels, borderline ranking/scope phrasing.
    # ------------------------------------------------------------------
    {
        "question": "How many orders don't have a channel recorded?",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": "Tests whether the generated SQL correctly uses 'channel IS NULL' rather than a broken equality check.",
    },
    {
        "question": "What was the top month for revenue?",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": (
            "Borderline for check_undefined_metric: uses \"top\" but the metric "
            "(revenue) IS specified, so this should NOT be flagged for "
            "clarification. Included to check for over-triggering."
        ),
    },
    {
        "question": "Which customers are most valuable?",
        "category": "edge_case",
        "expected_behavior": "clarify",
        "note": (
            "Genuinely borderline: \"most valuable\" strongly implies revenue but "
            "never says so. Could reasonably go either way - judge this one "
            "yourself rather than trusting the guessed expected_behavior."
        ),
    },
]
