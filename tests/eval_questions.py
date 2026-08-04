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

This file does not judge correctness on its own - it records intent so a
human (you) can compare it against what tests/run_eval.py or
tests/run_benchmark.py actually produced.

ground_truth
------------
Every question in the "messy_categorical_filter" and "granularity_mismatch"
categories carries a `ground_truth` field: a hand-written reference SQL
query, independent of anything the LLM might generate, that computes the
objectively correct answer. tests/run_benchmark.py executes this query
directly against the database and compares it to what the *unguarded*
pipeline (ask_unguarded(), which skips all five guardrails) actually
returns, so these two categories can be graded automatically rather than by
hand.

  - messy_categorical_filter ground truth: customers.region has four
    variants per region on purpose ("West"/"west"/"WEST"/"W" all mean the
    same thing - see generate_synthetic_data.py). The reference query
    normalizes with UPPER(TRIM(region)) and explicitly includes the
    single-letter abbreviation, so it counts every variant.
  - granularity_mismatch ground truth: marketing_spend is monthly-grain and
    is NOT decomposable to the order level - "total marketing spend
    associated with X channel orders" is therefore correctly answered by a
    single-table aggregate directly on marketing_spend (channel/month
    filtered), no join required. An ungrounded model that joins orders to
    marketing_spend to "attribute" spend per order will fan out the monthly
    spend row across every matching order and overcount.

Other categories (clean, undefined_metric, out_of_scope, edge_case) don't
carry ground_truth: clean/edge_case are informational only in the
benchmark (used to sanity-check guardrails don't misfire, not to measure
guarded-vs-unguarded accuracy); undefined_metric/out_of_scope require a
human judgment call ("was the LLM's silent default reasonable or
misleading?") that no reference query can settle, so run_benchmark.py
surfaces those side by side for manual grading instead.

check_bad_join (guardrails/checks.py) deliberately has no eval_questions
entry and isn't wired into run_benchmark.py's categories. Every other
category here was included because a specific natural-language question
reliably reproduces the issue via live SQL generation - verified
empirically before adding it. check_bad_join doesn't have one: a
competently-schemed database gives the model enough signal (clear column
names, a coherent schema description) that it doesn't generate
structurally nonsensical joins from ordinary questions, even against a
deliberately ambiguous demo schema built specifically to try to trip it up
(two tables that both happen to have a generic "id" column - see
guardrails/test_checks.py's join_db_path fixture). Forcing an eval
question in here would just be flaky. check_bad_join is covered instead by
direct unit tests against hand-crafted SQL (both the demo dataset and
join_db_path), the same way check_granularity_mismatch/
check_messy_categorical_filter's static-analysis logic is tested.
"""

EVAL_QUESTIONS = [
    # ------------------------------------------------------------------
    # Clean: clear metric, clear scope - should just answer. Informational
    # in the benchmark (no ground_truth) - used to sanity-check guardrails
    # don't false-positive on unambiguous questions.
    # ------------------------------------------------------------------
    {
        "question": "What was total revenue in 2024?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Clear metric (revenue), clear time bound. Baseline sanity check.",
        "ground_truth": None,
    },
    {
        "question": "How many customers signed up in 2025?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Simple filtered count on customers.signup_date.",
        "ground_truth": None,
    },
    {
        "question": "What is the average order revenue?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Single aggregate over orders, no ranking or scope ambiguity.",
        "ground_truth": None,
    },
    {
        "question": "How many support tickets are currently open?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Simple filtered count on support_tickets.status.",
        "ground_truth": None,
    },
    {
        "question": "What was the total marketing spend in 2024?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Single-table aggregate on marketing_spend, no orders join involved.",
        "ground_truth": None,
    },
    {
        "question": "How many orders came through the Email channel?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Filtered count on orders.channel; channel value is explicit.",
        "ground_truth": None,
    },
    {
        "question": "How many customers are on the Enterprise plan?",
        "category": "clean",
        "expected_behavior": "answer",
        "note": "Simple filtered count on customers.plan_tier, unambiguous.",
        "ground_truth": None,
    },
    # ------------------------------------------------------------------
    # Undefined metric: ranking/superlative language, no metric named.
    # No ground_truth - grading whether the LLM's silently-picked metric was
    # a reasonable default or misleading requires human judgment.
    # ------------------------------------------------------------------
    {
        "question": "Who are the top customers?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Top\" with no metric - by revenue? order count? tickets?",
        "ground_truth": None,
    },
    {
        "question": "Which region performs best?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Best\" is undefined - revenue, customer count, retention?",
        "ground_truth": None,
    },
    {
        "question": "What's the leading marketing channel?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Leading\" by spend, by orders attributed, by revenue?",
        "ground_truth": None,
    },
    {
        "question": "Show me the worst month.",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Worst\" with no metric and no dimension (revenue? tickets? spend?).",
        "ground_truth": None,
    },
    {
        "question": "What's the top plan tier?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Top\" by customer count? by revenue per tier? unspecified.",
        "ground_truth": None,
    },
    {
        "question": "Which support ticket status is highest?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Highest\" implies count, but that's assumed, not stated.",
        "ground_truth": None,
    },
    {
        "question": "What's the best-performing channel this year?",
        "category": "undefined_metric",
        "expected_behavior": "clarify",
        "note": "\"Best-performing\" by revenue, order count, or ROI is all plausible and unstated.",
        "ground_truth": None,
    },
    # ------------------------------------------------------------------
    # Out of scope: unbounded dumps / full per-row detail, no aggregation.
    # No ground_truth - "was this scoped enough" is a judgment call.
    # ------------------------------------------------------------------
    {
        "question": "Show me everything in the database.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Textbook unbounded dump across all four tables.",
        "ground_truth": None,
    },
    {
        "question": "List all customers with full details.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Every row, every column, no filter or aggregation.",
        "ground_truth": None,
    },
    {
        "question": "Give me every order for every customer, with all their support tickets.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Unbounded multi-table row-level dump, no scoping.",
        "ground_truth": None,
    },
    {
        "question": "Dump the entire orders table.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Explicit unbounded single-table dump, no filter or aggregation.",
        "ground_truth": None,
    },
    {
        "question": "Show me every support ticket in full detail.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "Every row, every column, no filter or aggregation - same pattern, different table.",
        "ground_truth": None,
    },
    {
        "question": "Give me a complete export of all marketing spend records.",
        "category": "out_of_scope",
        "expected_behavior": "clarify",
        "note": "\"Complete export\" is an unbounded dump phrased as a business ask.",
        "ground_truth": None,
    },
    # ------------------------------------------------------------------
    # Granularity mismatch: orders (daily) vs marketing_spend (monthly).
    # marketing_spend can't actually be decomposed to the order level, so
    # the objectively correct answer to "spend associated with X channel's
    # orders" is just the single-table aggregate on marketing_spend - no
    # join needed. An unguarded model that joins orders to marketing_spend
    # to "attribute" spend per order will fan the monthly figure out across
    # every matching order and overcount, sometimes by 10-30x.
    # ------------------------------------------------------------------
    {
        "question": "What is the total marketing spend associated with Email channel orders in 2024?",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Naive join fans the monthly Email spend out across every 2024 Email order and overcounts.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Email' AND month LIKE '2024-%'",
            "description": "Total 2024 Email-channel spend, single-table aggregate - no join needed.",
        },
    },
    {
        "question": "What is the total marketing spend behind Paid Search channel orders in 2025?",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Same trap, different channel/year.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Paid Search' AND month LIKE '2025-%'",
            "description": "Total 2025 Paid Search-channel spend, single-table aggregate - no join needed.",
        },
    },
    {
        "question": "What is the total marketing spend tied to Organic channel orders overall?",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "No year filter - all-time version of the same trap.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Organic'",
            "description": "Total all-time Organic-channel spend, single-table aggregate - no join needed.",
        },
    },
    {
        "question": "What is the total marketing spend that funded Referral channel orders in 2023?",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Same trap, Referral channel, first year of data.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Referral' AND month LIKE '2023-%'",
            "description": "Total 2023 Referral-channel spend, single-table aggregate - no join needed.",
        },
    },
    {
        "question": "Show each order alongside the marketing spend for its channel, and report the total spend involved.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Original per-order framing (invites a join) plus an explicit request for a total, so the answer is a single comparable number.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend",
            "description": "Grand total marketing spend across all channels/time, single-table aggregate.",
        },
    },
    {
        "question": "Compare daily order revenue to marketing spend by channel, then report the total marketing spend for the Email channel in 2025.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Keeps the original daily-comparison framing that reliably invited a join in testing, narrowed to a scalar answer.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Email' AND month LIKE '2025-%'",
            "description": "Total 2025 Email-channel spend, single-table aggregate - no join needed.",
        },
    },
    {
        "question": "List every order next to that month's marketing spend for its channel, then report the total spend shown for the Paid Search channel in 2024.",
        "category": "granularity_mismatch",
        "expected_behavior": "warn",
        "note": "Same per-order listing framing as the original eval question, narrowed to a scalar answer.",
        "ground_truth": {
            "sql": "SELECT SUM(amount_spent) FROM marketing_spend WHERE channel = 'Paid Search' AND month LIKE '2024-%'",
            "description": "Total 2024 Paid Search-channel spend, single-table aggregate - no join needed.",
        },
    },
    # ------------------------------------------------------------------
    # Messy categorical filter: region has inconsistent casing on purpose
    # ('West'/'west'/'WEST'/'W' all mean the same thing), so an unnormalized
    # exact-match filter on it silently undercounts. Ground truth
    # normalizes with UPPER(TRIM(region)) and explicitly includes the
    # single-letter abbreviation.
    # ------------------------------------------------------------------
    {
        "question": "How many customers are in the West region?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": (
            "The motivating bug for check_messy_categorical_filter: generated "
            "`WHERE region = 'West'` and returned 134, silently missing "
            "'west'/'WEST'/'W' rows (184 is correct)."
        ),
        "ground_truth": {
            "sql": "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('WEST', 'W')",
            "description": "Correct West-region count including all case and abbreviation variants.",
        },
    },
    {
        "question": "List customers in the East region.",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Same undernormalized-filter risk as the West-region question, different region and phrasing (list vs. count).",
        "ground_truth": {
            "sql": "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('EAST', 'E')",
            "description": "Correct East-region count; graded by row count if the unguarded answer is a listing rather than a COUNT(*).",
        },
    },
    {
        "question": "How many customers signed up in the South region last year?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Combines a region filter with a date filter - checks the guardrail still fires when region isn't the only WHERE clause.",
        "ground_truth": {
            "sql": (
                "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('SOUTH', 'S') "
                "AND signup_date BETWEEN '2025-01-01' AND '2025-12-31'"
            ),
            "description": "\"Last year\" relative to 2026-08-01 is 2025; correct South-region signups in 2025.",
        },
    },
    {
        "question": "How many customers are in the North region?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Same bug, North region.",
        "ground_truth": {
            "sql": "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('NORTH', 'N')",
            "description": "Correct North-region count including all case and abbreviation variants.",
        },
    },
    {
        "question": "How many Pro-plan customers are in the West region?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Region filter combined with a second, unrelated (clean) filter on plan_tier.",
        "ground_truth": {
            "sql": "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('WEST', 'W') AND plan_tier = 'Pro'",
            "description": "Correct West-region Pro-plan count including all case and abbreviation variants.",
        },
    },
    {
        "question": "How many customers signed up in the North region in 2024?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Region filter combined with an explicit (not relative) year filter.",
        "ground_truth": {
            "sql": (
                "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('NORTH', 'N') "
                "AND signup_date BETWEEN '2024-01-01' AND '2024-12-31'"
            ),
            "description": "Correct North-region 2024 signups including all case and abbreviation variants.",
        },
    },
    {
        "question": "How many Enterprise-plan customers are in the South region?",
        "category": "messy_categorical_filter",
        "expected_behavior": "warn",
        "note": "Same pattern as the West/Pro question, South region and a different plan tier.",
        "ground_truth": {
            "sql": "SELECT COUNT(*) FROM customers WHERE UPPER(TRIM(region)) IN ('SOUTH', 'S') AND plan_tier = 'Enterprise'",
            "description": "Correct South-region Enterprise-plan count including all case and abbreviation variants.",
        },
    },
    # ------------------------------------------------------------------
    # Edge cases: NULL channels, borderline ranking/scope phrasing.
    # Informational in the benchmark (no ground_truth) - regression checks
    # for over/under-triggering, not part of the guarded-vs-unguarded
    # accuracy measurement.
    # ------------------------------------------------------------------
    {
        "question": "How many orders don't have a channel recorded?",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": "Tests whether the generated SQL correctly uses 'channel IS NULL' rather than a broken equality check.",
        "ground_truth": None,
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
        "ground_truth": None,
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
        "ground_truth": None,
    },
    {
        "question": "How many customers have never placed an order?",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": "Clear question, but tests whether the generated SQL correctly handles the NOT IN / LEFT JOIN NULL pattern.",
        "ground_truth": None,
    },
    {
        "question": "What's the average number of orders per customer?",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": "Clear metric, but tests whether customers with zero orders are handled correctly in the denominator.",
        "ground_truth": None,
    },
    {
        "question": "Show me the top 3 support ticket statuses by count.",
        "category": "edge_case",
        "expected_behavior": "answer",
        "note": "Uses \"top\" but explicitly names the metric (\"by count\") - should NOT trigger check_undefined_metric.",
        "ground_truth": None,
    },
]
