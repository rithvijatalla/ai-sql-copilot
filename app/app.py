"""
Streamlit front-end for the analyst copilot.

Thin UI over pipeline.query_engine.ask() - all the actual logic (schema
introspection, SQL generation, guardrails, read-only execution) lives
there. This file is just responsible for collecting a question, calling
ask(), and rendering whichever of its four outcomes applies:
clarification needed, error, warning-plus-result, or plain result.
"""

import os
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.query_engine import DEFAULT_DB_PATH, ask

EXAMPLE_QUESTIONS = [
    ("Answers cleanly", "What was total revenue in 2024?"),
    ("Triggers check_undefined_metric", "Who are the top customers?"),
    ("Triggers check_out_of_scope", "Show me everything in the database."),
    (
        "Triggers check_granularity_mismatch",
        "Show each order alongside the marketing spend for its channel.",
    ),
    (
        "Triggers check_messy_categorical_filter",
        "How many customers are in the West region?",
    ),
]

st.set_page_config(page_title="AI SQL Analyst Copilot", layout="centered")


def render_setup_check() -> bool:
    """Check for the database and API key, showing a clear setup message
    instead of letting a missing prerequisite surface as a stack trace or
    confusing empty result. Returns True if it's safe to proceed."""
    if not Path(DEFAULT_DB_PATH).exists():
        st.error(
            "The analytics database wasn't found at "
            f"`{DEFAULT_DB_PATH}`.\n\n"
            "Generate it first by running:\n\n"
            "```\npython3 generate_synthetic_data.py\n```"
        )
        return False

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning(
            "`ANTHROPIC_API_KEY` isn't set in this environment. Asking a "
            "question will likely fail until it's configured, e.g.:\n\n"
            "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n```"
        )

    return True


st.title("AI SQL Analyst Copilot")
st.markdown(
    "Ask a question in plain English about the analytics database "
    "(customers, orders, marketing spend, and support tickets). It "
    "generates SQL with Claude, runs it read-only, and flags a few "
    "specific ways an answer can be misleading before showing it to you."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool answers questions over a small synthetic analytics "
        "database. Before running any generated SQL, it checks the "
        "question and query against four guardrails - two catch problems "
        "in the question itself (ambiguous metrics, unbounded requests) "
        "and two catch problems in the generated SQL (a daily/monthly "
        "grain mismatch, and an unnormalized filter on a column with "
        "inconsistent casing)."
    )
    st.header("Example questions")
    st.markdown("Try one of these to see a specific behavior:")
    for label, question in EXAMPLE_QUESTIONS:
        st.markdown(f"**{label}**")
        st.code(question, language=None)

ready = render_setup_check()

with st.form(key="ask_form"):
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What was total revenue in 2024?",
    )
    submitted = st.form_submit_button("Ask", disabled=not ready)

if submitted and ready:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking..."):
            result = ask(question)

        if result["clarification_needed"]:
            st.info(result["clarification_needed"])
        else:
            if result["error"]:
                st.error(result["error"])

            if result["warning"]:
                st.warning(result["warning"])

            if result["result"] is not None:
                st.dataframe(result["result"])

            if result["sql"]:
                with st.expander("Generated SQL"):
                    st.code(result["sql"], language="sql")
