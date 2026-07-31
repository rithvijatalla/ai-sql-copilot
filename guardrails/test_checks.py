"""
Tests for guardrails/checks.py.

check_granularity_mismatch and check_messy_categorical_filter are pure text
inspection, so they're tested directly against real SQL strings - no
mocking needed.

check_undefined_metric and check_out_of_scope call an LLM. Their parsing and
tool-forcing wiring is tested against a mocked Anthropic client (fast,
deterministic, no API key or network required, runs in CI). A second set of
tests exercises the real model to catch prompt-quality regressions the mocks
can't - those are skipped automatically when ANTHROPIC_API_KEY isn't set.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.checks import (
    check_granularity_mismatch,
    check_messy_categorical_filter,
    check_out_of_scope,
    check_undefined_metric,
)

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _tool_use_block(input_dict):
    return types.SimpleNamespace(type="tool_use", input=input_dict)


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _mock_response(anthropic_mock, content_blocks):
    anthropic_mock.return_value.messages.create.return_value = types.SimpleNamespace(
        content=content_blocks
    )


# ---------------------------------------------------------------------------
# check_granularity_mismatch - pure text inspection
# ---------------------------------------------------------------------------


def test_granularity_mismatch_flags_unreconciled_join():
    sql = (
        "SELECT o.revenue, m.amount_spent FROM orders o "
        "JOIN marketing_spend m ON o.channel = m.channel"
    )
    warning = check_granularity_mismatch(sql)
    assert warning is not None
    assert "orders" in warning
    assert "marketing_spend" in warning


def test_granularity_mismatch_allows_strftime_reconciled_join():
    sql = (
        "SELECT strftime('%Y-%m', o.order_date) AS month, SUM(o.revenue), m.amount_spent "
        "FROM orders o JOIN marketing_spend m "
        "ON strftime('%Y-%m', o.order_date) = m.month AND o.channel = m.channel "
        "GROUP BY month, o.channel"
    )
    assert check_granularity_mismatch(sql) is None


def test_granularity_mismatch_allows_substr_reconciled_join():
    sql = (
        "SELECT substr(o.order_date, 1, 7) AS month, SUM(o.revenue), m.amount_spent "
        "FROM orders o JOIN marketing_spend m ON substr(o.order_date, 1, 7) = m.month "
        "GROUP BY month"
    )
    assert check_granularity_mismatch(sql) is None


def test_granularity_mismatch_flags_group_by_without_month_alignment():
    # GROUP BY is present, but not on a month-truncated date - still a mismatch.
    sql = (
        "SELECT o.channel, SUM(o.revenue), m.amount_spent FROM orders o "
        "JOIN marketing_spend m ON o.channel = m.channel GROUP BY o.channel"
    )
    assert check_granularity_mismatch(sql) is not None


def test_granularity_mismatch_ignores_orders_only():
    assert check_granularity_mismatch("SELECT SUM(revenue) FROM orders") is None


def test_granularity_mismatch_ignores_marketing_spend_only():
    assert check_granularity_mismatch("SELECT SUM(amount_spent) FROM marketing_spend") is None


def test_granularity_mismatch_ignores_unrelated_tables():
    assert check_granularity_mismatch("SELECT * FROM customers") is None


def test_granularity_mismatch_is_case_insensitive():
    sql = "SELECT * FROM ORDERS o JOIN Marketing_Spend m ON o.channel = m.channel"
    assert check_granularity_mismatch(sql) is not None


def test_granularity_mismatch_does_not_match_substring_of_table_name():
    # 'orders' must not match inside an unrelated identifier like 'reorders_view'.
    sql = "SELECT * FROM reorders_view JOIN marketing_spend ON 1=1"
    assert check_granularity_mismatch(sql) is None


# ---------------------------------------------------------------------------
# check_messy_categorical_filter - pure text inspection
# ---------------------------------------------------------------------------


def test_messy_categorical_filter_flags_bare_exact_match():
    warning = check_messy_categorical_filter("SELECT COUNT(*) FROM customers WHERE region = 'West'")
    assert warning is not None
    assert "region" in warning
    assert "'West'" in warning


def test_messy_categorical_filter_flags_qualified_column():
    sql = "SELECT COUNT(*) FROM customers c WHERE c.region = 'East'"
    assert check_messy_categorical_filter(sql) is not None


def test_messy_categorical_filter_is_case_insensitive_on_column_name():
    sql = "SELECT * FROM customers WHERE REGION = 'North'"
    assert check_messy_categorical_filter(sql) is not None


def test_messy_categorical_filter_allows_lower_normalized_comparison():
    sql = "SELECT COUNT(*) FROM customers WHERE LOWER(region) = 'west'"
    assert check_messy_categorical_filter(sql) is None


def test_messy_categorical_filter_allows_upper_normalized_comparison():
    sql = "SELECT COUNT(*) FROM customers WHERE UPPER(region) = 'WEST'"
    assert check_messy_categorical_filter(sql) is None


def test_messy_categorical_filter_allows_trim_normalized_comparison():
    sql = "SELECT COUNT(*) FROM customers WHERE TRIM(region) = 'West'"
    assert check_messy_categorical_filter(sql) is None


def test_messy_categorical_filter_ignores_other_columns():
    sql = "SELECT * FROM orders WHERE channel = 'Email'"
    assert check_messy_categorical_filter(sql) is None


def test_messy_categorical_filter_ignores_queries_without_region_filter():
    assert check_messy_categorical_filter("SELECT * FROM customers") is None


# ---------------------------------------------------------------------------
# check_undefined_metric - mocked LLM
# ---------------------------------------------------------------------------


def test_undefined_metric_flags_when_model_says_needs_clarification():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_clarification": True, "clarifying_question": "By what metric?"})],
        )
        result = check_undefined_metric("Who are the top customers?")
    assert result == "By what metric?"


def test_undefined_metric_passes_when_model_says_no_issue():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_clarification": False, "clarifying_question": ""})],
        )
        result = check_undefined_metric("Top 5 customers by revenue")
    assert result is None


def test_undefined_metric_fails_open_when_no_tool_use_returned():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(mock_anthropic, [_text_block("I can't help with that.")])
        result = check_undefined_metric("Who are the top customers?")
    assert result is None


def test_undefined_metric_forces_the_correct_tool():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_clarification": False, "clarifying_question": ""})],
        )
        check_undefined_metric("Top 5 customers by revenue")
        _, kwargs = mock_anthropic.return_value.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_metric_check"}
    assert kwargs["tools"][0]["name"] == "report_metric_check"


# ---------------------------------------------------------------------------
# check_out_of_scope - mocked LLM
# ---------------------------------------------------------------------------


def test_out_of_scope_flags_when_model_says_needs_scoping():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_scoping": True, "message": "Please narrow this down."})],
        )
        result = check_out_of_scope("Show me everything")
    assert result == "Please narrow this down."


def test_out_of_scope_passes_when_model_says_no_issue():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_scoping": False, "message": ""})],
        )
        result = check_out_of_scope("List customers in the West region")
    assert result is None


def test_out_of_scope_fails_open_when_no_tool_use_returned():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(mock_anthropic, [_text_block("I can't help with that.")])
        result = check_out_of_scope("Show me everything")
    assert result is None


def test_out_of_scope_forces_the_correct_tool():
    with patch("guardrails.checks.anthropic.Anthropic") as mock_anthropic:
        _mock_response(
            mock_anthropic,
            [_tool_use_block({"needs_scoping": False, "message": ""})],
        )
        check_out_of_scope("List customers in the West region")
        _, kwargs = mock_anthropic.return_value.messages.create.call_args
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_scope_check"}
    assert kwargs["tools"][0]["name"] == "report_scope_check"


# ---------------------------------------------------------------------------
# Live-model tests - real semantic judgment, skipped without an API key
# ---------------------------------------------------------------------------


@requires_api_key
def test_live_undefined_metric_flags_bare_ranking_question():
    assert check_undefined_metric("Who are the top customers?") is not None


@requires_api_key
def test_live_undefined_metric_passes_when_metric_is_specified():
    assert check_undefined_metric("Who are the top 5 customers by revenue?") is None


@requires_api_key
def test_live_undefined_metric_passes_non_ranking_question():
    assert check_undefined_metric("What was total revenue in 2024?") is None


@requires_api_key
def test_live_out_of_scope_flags_unbounded_dump():
    assert check_out_of_scope("Show me everything") is not None


@requires_api_key
def test_live_out_of_scope_flags_full_detail_request():
    assert check_out_of_scope("List all customers with full details") is not None


@requires_api_key
def test_live_out_of_scope_passes_filtered_request():
    assert check_out_of_scope("List customers in the West region") is None
