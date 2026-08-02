"""
Streamlit front-end for the analyst copilot.

Thin UI over pipeline.query_engine.ask() - all the actual logic (schema
introspection, SQL generation, guardrails, read-only execution) lives
there. This file is responsible for: letting the user pick between the
bundled demo dataset and their own uploaded CSV/Excel files, collecting a
question, calling ask() against whichever database is active, and rendering
whichever of its four outcomes applies: clarification needed, error,
warning-plus-result, or plain result.

Visual design lives entirely in this file (CSS injected via st.markdown +
.streamlit/config.toml for the base theme) - it has no bearing on the
pipeline/guardrail logic above.
"""

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.data_loader import delete_database, load_uploaded_files
from pipeline.query_engine import DEFAULT_DB_PATH, ask
from pipeline.schema_utils import get_schema_description


def resolve_api_key() -> str | None:
    """Resolve the Anthropic API key from either deployment context:
    st.secrets (Streamlit Cloud) takes priority when available, falling
    back to os.environ (local development via `source .env`).

    Resolving it here rather than in the pipeline keeps
    pipeline/guardrails Streamlit-agnostic - they just read
    ANTHROPIC_API_KEY from the environment as before, via
    anthropic.Anthropic(), so it's injected into os.environ once below.

    Uses load_if_toml_exists() rather than a bare st.secrets.get() /
    try-except: when no secrets.toml exists, st.secrets internally calls
    st.error() and renders it on the page as a side effect *before*
    raising FileNotFoundError, so catching the exception in Python doesn't
    stop the error banner from having already been queued for render.
    load_if_toml_exists() is Streamlit's own silent-probe variant, built
    for exactly this "secrets may not be configured, and that's fine"
    case.
    """
    if st.secrets.load_if_toml_exists():
        secret_key = st.secrets.get("ANTHROPIC_API_KEY")
    else:
        secret_key = None
    return secret_key or os.environ.get("ANTHROPIC_API_KEY")


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

# ---------------------------------------------------------------------------
# Icons - hand-drawn inline SVGs (line-icon style), no emoji anywhere.
# ---------------------------------------------------------------------------

_ICON_DATABASE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/></svg>'
_ICON_SEARCH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
_ICON_RESULTS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="12" width="3.5" height="8"/><rect x="10.25" y="7" width="3.5" height="13"/><rect x="16.5" y="3" width="3.5" height="17"/></svg>'
_ICON_INFO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><circle cx="12" cy="8" r="0.6" fill="currentColor" stroke="none"/></svg>'

_ICON_METRIC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polyline points="3,17 9,11 13,15 21,7"/><polyline points="14,7 21,7 21,14"/></svg>'
_ICON_SCOPE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v4M2 6h4"/><path d="M18 2v4M22 6h-4"/><path d="M6 22v-4M2 18h4"/><path d="M18 22v-4M22 18h-4"/></svg>'
_ICON_GRANULARITY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="10" x2="21" y2="10"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="16" y1="2" x2="16" y2="6"/></svg>'
_ICON_MESSY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="4,4 20,4 14,12 14,19 10,17 10,12"/></svg>'

_ICON_ALERT_INFO = _ICON_INFO
_ICON_ALERT_WARNING = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 20h20L12 3z"/><line x1="12" y1="9" x2="12" y2="14"/><circle cx="12" cy="17" r="0.6" fill="currentColor" stroke="none"/></svg>'
_ICON_ALERT_ERROR = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="8" y1="8" x2="16" y2="16"/><line x1="16" y1="8" x2="8" y2="16"/></svg>'

_CALLOUT_ICONS = {
    "info": _ICON_ALERT_INFO,
    "warning": _ICON_ALERT_WARNING,
    "error": _ICON_ALERT_ERROR,
}

GUARDRAIL_INFO = [
    (
        _ICON_METRIC,
        "Undefined metric",
        'Ranking language ("top", "best") with no metric specified is flagged before SQL is generated.',
    ),
    (
        _ICON_SCOPE,
        "Out of scope",
        "Unbounded dumps or full per-row detail requested with no filter or aggregation.",
    ),
    (
        _ICON_GRANULARITY,
        "Granularity mismatch",
        "Joins between tables whose date columns have different estimated grains (e.g. daily vs. monthly).",
    ),
    (
        _ICON_MESSY,
        "Messy categorical filter",
        "Exact-match filters on columns whose values have inconsistent casing or whitespace in the data.",
    ),
]

st.set_page_config(page_title="AI SQL Copilot", page_icon="◆", layout="centered")

# st.secrets access must happen after set_page_config (it must be the first
# Streamlit command in the script), so the key is resolved and injected
# into os.environ here rather than at module top.
_resolved_api_key = resolve_api_key()
if _resolved_api_key:
    os.environ["ANTHROPIC_API_KEY"] = _resolved_api_key

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 760px; }

/* ---- brand header ----
   Extra top margin on top of block-container's own padding: defensive
   clearance so the accent bar/title/tagline never sit flush against
   Streamlit's fixed top toolbar (in-app, or any hosting-provided chrome
   on top of that in a deployed context) - a few extra px of whitespace
   is cheap, a clipped header is not.
   Title is sized as a landing-page hero heading (4.5rem / 72px, well
   past the 64px floor) - unmistakably the largest, boldest thing on the
   page - with the tagline, accent bar, and surrounding spacing all
   scaled up to match rather than looking cramped next to it. */
.brand { display: flex; align-items: center; gap: 24px; padding-bottom: 2.5rem;
    margin-top: 1.5rem; margin-bottom: 2.75rem; border-bottom: 1px solid rgba(232,163,61,0.16); }
.brand-mark { width: 7px; height: 112px; border-radius: 4px; flex-shrink: 0;
    background: linear-gradient(180deg, #E8A33D, #C97F1E); }
.brand-title { font-size: 4.5rem; font-weight: 800; letter-spacing: -0.03em;
    color: #E9ECF3; margin: 0; line-height: 1.1; }
.brand-title .accent { color: #E8A33D; }
.brand-subtitle { font-size: 1.4rem; font-weight: 500; color: #8A93AC;
    letter-spacing: 0.01em; margin-top: 14px; }

/* ---- section headers inside cards ---- */
.section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 0.9rem; }
.section-header svg { width: 18px; height: 18px; color: #E8A33D; flex-shrink: 0; }
.section-header span { font-size: 0.98rem; font-weight: 700; color: #E9ECF3; letter-spacing: -0.01em; }

/* ---- status badge ---- */
.status-badge { display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px;
    border-radius: 999px; background: rgba(232,163,61,0.10); border: 1px solid rgba(232,163,61,0.28);
    font-size: 0.78rem; font-weight: 600; color: #E8A33D; white-space: nowrap; margin-top: 6px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #E8A33D;
    box-shadow: 0 0 6px rgba(232,163,61,0.85); flex-shrink: 0; }

/* ---- callouts (replace default st.error/warning/info chrome) ---- */
.callout { display: flex; gap: 10px; padding: 0.85rem 1rem; border-radius: 10px;
    margin: 0.7rem 0; border: 1px solid; font-size: 0.88rem; line-height: 1.55; }
.callout-icon svg { width: 17px; height: 17px; margin-top: 2px; }
.callout-info { background: rgba(94,158,214,0.08); border-color: rgba(94,158,214,0.28); color: #C7DFF1; }
.callout-info .callout-icon { color: #5E9ED6; }
.callout-warning { background: rgba(232,163,61,0.09); border-color: rgba(232,163,61,0.32); color: #F3D9AE; }
.callout-warning .callout-icon { color: #E8A33D; }
.callout-error { background: rgba(224,90,90,0.09); border-color: rgba(224,90,90,0.34); color: #F3C4C4; }
.callout-error .callout-icon { color: #E05A5A; }

/* ---- guardrail list (About card) ---- */
.guardrail-item { display: flex; gap: 12px; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
.guardrail-item:last-child { border-bottom: none; padding-bottom: 0; }
.guardrail-icon { width: 28px; height: 28px; border-radius: 8px; background: rgba(232,163,61,0.12);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.guardrail-icon svg { width: 15px; height: 15px; color: #E8A33D; }
.guardrail-title { font-weight: 700; font-size: 0.82rem; color: #E9ECF3; margin-bottom: 1px; }
.guardrail-desc { font-size: 0.76rem; color: #8F98B0; line-height: 1.45; }

/* ---- misc polish ---- */
div[data-testid="stMetric"] { background: rgba(255,255,255,0.025); border-radius: 10px;
    padding: 0.8rem 1.1rem; border: 1px solid rgba(255,255,255,0.06); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px !important; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


def render_brand_header() -> None:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-mark"></div>'
        '<div>'
        '<p class="brand-title">AI SQL <span class="accent">Copilot</span></p>'
        '<p class="brand-subtitle">Natural language &rarr; SQL, with guardrails that catch misleading answers</p>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">{icon}<span>{html.escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def render_callout(kind: str, message: str) -> None:
    escaped = html.escape(message).replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(
        f'<div class="callout callout-{kind}">'
        f'<span class="callout-icon">{_CALLOUT_ICONS[kind]}</span>'
        f'<div>{escaped}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _format_metric_value(value) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


def render_result_table(df: pd.DataFrame) -> None:
    """Render a single numeric value as a metric tile; anything else as a table."""
    is_single_metric = df.shape == (1, 1) and pd.api.types.is_numeric_dtype(df.dtypes.iloc[0])
    if is_single_metric:
        label = df.columns[0].replace("_", " ")
        st.metric(label=label, value=_format_metric_value(df.iloc[0, 0]))
    else:
        st.dataframe(df, use_container_width=True)


def render_setup_check(db_path: str) -> bool:
    """Check for the database and API key, showing a clear setup message
    instead of letting a missing prerequisite surface as a stack trace or
    confusing empty result. Returns True if it's safe to proceed."""
    if not Path(db_path).exists():
        render_callout(
            "error",
            "The analytics database wasn't found at "
            f"`{db_path}`.\n\nGenerate it first by running:\n\npython3 generate_synthetic_data.py",
        )
        return False

    if not resolve_api_key():
        render_callout(
            "warning",
            "`ANTHROPIC_API_KEY` isn't configured. Asking a question will "
            "likely fail until it's set - locally, via a `.env` file "
            "(`echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env`, then "
            "`source .env`); on Streamlit Cloud, via the app's "
            "Settings -> Secrets (`ANTHROPIC_API_KEY = \"sk-ant-...\"`).",
        )

    return True


def render_data_source() -> tuple[str, bool]:
    """Render the demo/upload toggle and, in upload mode, the file uploader.

    Returns (active_db_path, is_demo). Uploaded files are loaded into a
    fresh temp SQLite database each time the uploaded file set changes; the
    previous temp database (if any) is deleted so temp files don't pile up
    across reruns.
    """
    with st.container(border=True):
        render_section_header(_ICON_DATABASE, "Data source")

        col_toggle, col_status = st.columns([2.4, 1])
        with col_toggle:
            mode = st.radio(
                "Data source",
                ["Use demo dataset", "Upload your own data"],
                horizontal=True,
                label_visibility="collapsed",
            )

        if mode == "Use demo dataset":
            if st.session_state.get("uploaded_db_path"):
                delete_database(st.session_state["uploaded_db_path"])
                st.session_state["uploaded_db_path"] = None
                st.session_state["uploaded_file_key"] = None
            with col_status:
                st.markdown(
                    '<span class="status-badge"><span class="status-dot"></span>Demo dataset</span>',
                    unsafe_allow_html=True,
                )
            return DEFAULT_DB_PATH, True

        uploaded_files = st.file_uploader(
            "Upload one or more CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if not uploaded_files:
            with col_status:
                st.markdown(
                    '<span class="status-badge"><span class="status-dot"></span>No files yet</span>',
                    unsafe_allow_html=True,
                )
            render_callout("info", "Upload at least one CSV or Excel file to ask questions about it.")
            return "", False

        file_key = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("uploaded_file_key") != file_key:
            if st.session_state.get("uploaded_db_path"):
                delete_database(st.session_state["uploaded_db_path"])
            with st.spinner("Loading uploaded files..."):
                try:
                    db_path = load_uploaded_files(uploaded_files)
                except Exception as e:
                    render_callout("error", f"Failed to load uploaded files: {e}")
                    st.session_state["uploaded_db_path"] = None
                    st.session_state["uploaded_file_key"] = None
                    return "", False
            st.session_state["uploaded_db_path"] = db_path
            st.session_state["uploaded_file_key"] = file_key

        db_path = st.session_state["uploaded_db_path"]
        table_count = get_schema_description(db_path).count("Table:")
        with col_status:
            st.markdown(
                f'<span class="status-badge"><span class="status-dot"></span>'
                f'{len(uploaded_files)} file(s) &middot; {table_count} table(s)</span>',
                unsafe_allow_html=True,
            )
        with st.expander("Detected schema"):
            st.code(get_schema_description(db_path), language="text")

        return db_path, False


def render_about_sidebar(is_demo: bool) -> None:
    with st.sidebar:
        with st.container(border=True):
            render_section_header(_ICON_INFO, "About")
            st.markdown(
                "Answers questions over either the demo dataset or data you "
                "upload. Every generated query is checked against four "
                "guardrails, auto-detected from whichever dataset is active:"
            )
            items_html = "".join(
                f'<div class="guardrail-item">'
                f'<div class="guardrail-icon">{icon}</div>'
                f'<div><div class="guardrail-title">{html.escape(title)}</div>'
                f'<div class="guardrail-desc">{html.escape(desc)}</div></div>'
                f"</div>"
                for icon, title, desc in GUARDRAIL_INFO
            )
            st.markdown(items_html, unsafe_allow_html=True)

        if is_demo:
            with st.container(border=True):
                render_section_header(_ICON_SEARCH, "Example questions")
                st.caption("Try one of these to see a specific behavior:")
                for i, (label, question) in enumerate(EXAMPLE_QUESTIONS):
                    st.caption(f"**{label}**")
                    if st.button(question, key=f"example_{i}", use_container_width=True):
                        st.session_state["question_input"] = question


render_brand_header()

db_path, is_demo = render_data_source()
render_about_sidebar(is_demo)

ready = bool(db_path) and render_setup_check(db_path)

with st.container(border=True):
    render_section_header(_ICON_SEARCH, "Ask a question")
    with st.form(key="ask_form"):
        col_input, col_button = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. What was total revenue in 2024?",
                key="question_input",
                label_visibility="collapsed",
            )
        with col_button:
            submitted = st.form_submit_button(
                "Ask", disabled=not ready, use_container_width=True, type="primary"
            )

if submitted and ready:
    if not question.strip():
        render_callout("warning", "Enter a question first.")
    else:
        with st.spinner("Generating SQL and running guardrail checks..."):
            result = ask(question, db_path=db_path)

        with st.container(border=True):
            render_section_header(_ICON_RESULTS, "Result")

            if result["clarification_needed"]:
                render_callout("info", result["clarification_needed"])
            else:
                if result["error"]:
                    render_callout("error", result["error"])

                if result["warning"]:
                    render_callout("warning", result["warning"])

                if result["result"] is not None:
                    render_result_table(result["result"])

                if result["sql"]:
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
