"""
Streamlit front-end for the analyst copilot.

Thin UI over pipeline.query_engine.ask() - all the actual logic (schema
introspection, SQL generation, guardrails, read-only execution) lives
there. This file is responsible for: letting the user pick between the
bundled demo dataset and their own uploaded CSV/Excel files, collecting a
question, calling ask() against whichever database is active, and rendering
whichever of its four outcomes applies: clarification needed, error,
warning-plus-result, or plain result.
"""

import os
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.data_loader import delete_database, load_uploaded_files
from pipeline.query_engine import DEFAULT_DB_PATH, ask
from pipeline.schema_utils import get_schema_description

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

st.set_page_config(page_title="AI SQL Copilot", page_icon="🧠", layout="centered")


def render_setup_check(db_path: str) -> bool:
    """Check for the database and API key, showing a clear setup message
    instead of letting a missing prerequisite surface as a stack trace or
    confusing empty result. Returns True if it's safe to proceed."""
    if not Path(db_path).exists():
        st.error(
            "The analytics database wasn't found at "
            f"`{db_path}`.\n\n"
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


def render_data_source() -> tuple[str, bool]:
    """Render the demo/upload toggle and, in upload mode, the file uploader.

    Returns (active_db_path, is_demo). Uploaded files are loaded into a
    fresh temp SQLite database each time the uploaded file set changes; the
    previous temp database (if any) is deleted so temp files don't pile up
    across reruns.
    """
    mode = st.radio(
        "Data source",
        ["Use demo dataset", "Upload your own data"],
        horizontal=True,
    )

    if mode == "Use demo dataset":
        if st.session_state.get("uploaded_db_path"):
            delete_database(st.session_state["uploaded_db_path"])
            st.session_state["uploaded_db_path"] = None
            st.session_state["uploaded_file_key"] = None
        return DEFAULT_DB_PATH, True

    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one CSV or Excel file to ask questions about it.")
        return "", False

    file_key = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("uploaded_file_key") != file_key:
        if st.session_state.get("uploaded_db_path"):
            delete_database(st.session_state["uploaded_db_path"])
        with st.spinner("Loading uploaded files..."):
            try:
                db_path = load_uploaded_files(uploaded_files)
            except Exception as e:
                st.error(f"Failed to load uploaded files: {e}")
                st.session_state["uploaded_db_path"] = None
                st.session_state["uploaded_file_key"] = None
                return "", False
        st.session_state["uploaded_db_path"] = db_path
        st.session_state["uploaded_file_key"] = file_key

    db_path = st.session_state["uploaded_db_path"]
    with st.expander("Detected schema"):
        st.code(get_schema_description(db_path), language="text")

    return db_path, False


st.title("🧠 AI SQL Analyst Copilot")
st.markdown(
    "Ask a question in plain English about your data. It generates SQL "
    "with Claude, runs it read-only, and flags a few specific ways an "
    "answer can be misleading before showing it to you."
)

db_path, is_demo = render_data_source()

with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool answers questions over either a small synthetic demo "
        "database or data you upload yourself. Before running any "
        "generated SQL, it checks the question and query against four "
        "guardrails - two catch problems in the question itself (ambiguous "
        "metrics, unbounded requests) and two catch problems in the "
        "generated SQL, auto-detected from whichever dataset is active: a "
        "date-grain mismatch between joined tables, and an unnormalized "
        "filter on a column with inconsistent casing."
    )
    if is_demo:
        st.header("Example questions")
        st.markdown("Try one of these to see a specific behavior:")
        for i, (label, question) in enumerate(EXAMPLE_QUESTIONS):
            st.markdown(f"**{label}**")
            if st.button(question, key=f"example_{i}"):
                st.session_state["question_input"] = question

ready = bool(db_path) and render_setup_check(db_path)

with st.form(key="ask_form"):
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What was total revenue in 2024?",
        key="question_input",
    )
    submitted = st.form_submit_button("Ask", disabled=not ready)

if submitted and ready:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking..."):
            result = ask(question, db_path=db_path)

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
