"""
Guardrail checks for the NL-to-SQL analyst copilot.

Each check catches a specific way a question or generated query can produce
a technically-valid but misleading or unsafe answer:

1. check_undefined_metric (pre-generation, LLM judgment)
   Ranking language ("top", "best", "leading", ...) is meaningless without a
   metric to rank by. Left unspecified, the SQL generator will silently pick
   one (usually revenue, or whatever column looks numeric) and the user gets
   a confident-looking answer to a question they didn't actually ask. This
   needs an LLM rather than keyword matching because "top 5 customers by
   revenue" and "top 5 customers" differ only in whether a metric is present
   further in the sentence - a real semantic read, not a regex.

2. check_granularity_mismatch (post-generation, static analysis)
   orders is daily grain; marketing_spend is monthly grain (see
   generate_synthetic_data.py's docstring - this mismatch is intentional in
   the data). A query that touches both tables without reconciling the grain
   (e.g. grouping orders by month before comparing to spend) will join or
   aggregate across mismatched row granularities, which silently
   double-counts or misaligns values. This is caught by inspecting the SQL
   text - a schema-level fact about these two specific tables, not something
   that needs a model call.

3. check_out_of_scope (pre-generation, LLM judgment)
   Requests for unbounded dumps ("show me everything") or full per-row
   customer detail without aggregation are a scoping problem, not a SQL
   problem: they don't have a wrong answer so much as an answer nobody
   actually wants to receive as a flat table, and they're the shape of
   request most likely to leak more row-level detail than intended. Like
   check_undefined_metric, this needs semantic judgment ("list all customers
   in the West region" is a normal filtered query; "list all customers with
   full details" is not) rather than a keyword blocklist.

4. check_messy_categorical_filter (post-generation, static analysis)
   customers.region contains inconsistent values on purpose ("West",
   "west", "WEST", "W" all mean the same thing - see
   generate_synthetic_data.py's docstring). This was caught in testing:
   "How many customers are in the West region?" generated
   `WHERE region = 'West'` and returned 134 - a real number that silently
   excludes every "west"/"WEST"/"W" row. Like check_granularity_mismatch,
   this is a known fact about one specific column's data quality, not
   something that needs semantic judgment - a regex for an unnormalized
   exact-match comparison on that column is sufficient and faster/cheaper
   than a model call.
"""

import re

import anthropic

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
    client = anthropic.Anthropic()

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


def check_undefined_metric(question: str) -> str | None:
    """Flag ranking/superlative questions that don't specify what to rank by."""
    system_prompt = """You review questions asked to an analytics SQL copilot.

Determine whether the question uses ranking or superlative language (e.g.
"top", "best", "worst", "highest", "lowest", "leading", "greatest") WITHOUT
specifying what metric to rank by (e.g. revenue, order count, spend, ticket
count, signups).

If a metric is explicitly named or unambiguously implied ("top customers by
revenue", "worst month for support tickets"), there is no problem.

If ranking language is used but no metric is specified or implied ("top
customers", "best region", "leading channel"), flag it and draft a short
clarifying question to ask the user, naming 2-3 plausible metric options
from the schema (customers, orders/revenue, marketing_spend, support_tickets)
that would fit the question.

If the question doesn't use ranking/superlative language at all, there is no
problem."""

    return _run_judgment(
        system_prompt=system_prompt,
        question=question,
        tool_name="report_metric_check",
        tool_description="Report whether the question needs a clarifying question about which metric to rank by.",
        flag_field="needs_clarification",
        message_field="clarifying_question",
    )


def check_granularity_mismatch(sql: str) -> str | None:
    """Flag SQL that mixes orders (daily grain) and marketing_spend (monthly
    grain) without an aggregation that reconciles the two."""
    has_orders = bool(re.search(r"\borders\b", sql, re.IGNORECASE))
    has_marketing_spend = bool(re.search(r"\bmarketing_spend\b", sql, re.IGNORECASE))

    if not (has_orders and has_marketing_spend):
        return None

    has_group_by = bool(re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE))
    has_month_alignment = bool(
        re.search(r"strftime\s*\(\s*['\"]%Y-%m['\"]", sql, re.IGNORECASE)
        or re.search(r"substr\s*\(\s*[\w.]*order_date\s*,\s*1\s*,\s*7\s*\)", sql, re.IGNORECASE)
    )

    if has_group_by and has_month_alignment:
        return None

    return (
        "This query combines 'orders' (daily grain) with 'marketing_spend' "
        "(monthly grain) without an aggregation that reconciles the two - e.g. "
        "grouping orders by month (strftime('%Y-%m', order_date)) before "
        "comparing to marketing_spend. Results may double-count or misalign "
        "values across the mismatched grains."
    )


def check_out_of_scope(question: str) -> str | None:
    """Flag requests for unbounded data dumps or ungrouped per-row detail."""
    system_prompt = """You review questions asked to an analytics SQL copilot
over a database of customers, orders, marketing_spend, and support_tickets.

Determine whether the question asks for an unbounded, comprehensive data
dump (e.g. "show me everything", "list all the data", "dump the whole
database") or for individual-level, non-aggregated detail across an entire
table with no filtering or summarization (e.g. "list all customers with
full details", "give me every order for every customer").

A filtered or aggregated request is fine, even if it returns many rows
("list customers in the West region", "show all orders from December",
"total revenue by customer"). The problem is specifically requests with no
bound, filter, or aggregation at all.

If the question is out of scope this way, flag it and draft a short message
explaining that it needs to be scoped down (e.g. by adding a filter, a time
range, a limit, or an aggregation), with a concrete suggestion.

Otherwise there is no problem."""

    return _run_judgment(
        system_prompt=system_prompt,
        question=question,
        tool_name="report_scope_check",
        tool_description="Report whether the question needs to be scoped down before it can be answered.",
        flag_field="needs_scoping",
        message_field="message",
    )


# Columns known to contain inconsistent casing/formatting on purpose.
# Currently just customers.region - see generate_synthetic_data.py.
_MESSY_CATEGORICAL_COLUMNS = ("region",)


def check_messy_categorical_filter(sql: str) -> str | None:
    """Flag exact-match filters on columns known to have inconsistent
    casing/formatting, where the comparison isn't normalized on both sides."""
    for column in _MESSY_CATEGORICAL_COLUMNS:
        # A normalization function (LOWER/UPPER/TRIM) directly wrapping the
        # column means the query already accounts for casing/whitespace
        # variance, so this pattern won't match - the column is followed by
        # ")" rather than "=" at the point where we're looking for it.
        match = re.search(
            rf"\b(?:\w+\.)?{column}\b\s*=\s*'([^']*)'", sql, re.IGNORECASE
        )
        if not match:
            continue

        value = match.group(1)
        return (
            f"This query filters on '{column}' with an exact match "
            f"({column} = '{value}'), but {column} contains inconsistent "
            f"casing/formatting on purpose (e.g. 'West', 'west', 'WEST', and "
            f"'W' all represent the same value - see "
            f"generate_synthetic_data.py). An exact match like this will "
            f"silently miss the other variants and undercount. Normalize both "
            f"sides of the comparison, e.g. LOWER(TRIM({column})) = "
            f"LOWER('{value}'), or explicitly account for all known variants."
        )

    return None
