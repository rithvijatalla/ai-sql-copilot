# AI SQL Copilot

A natural-language-to-SQL analyst copilot. You ask a question in plain
English; it generates a SQL query with Claude, runs it against a read-only
SQLite database, and returns the result — while a set of guardrails catches
the specific ways a technically-valid query can still be a misleading
answer.

It ships with a small, deliberately messy synthetic "demo dataset"
(inconsistent region casing, null marketing channels, a daily-vs-monthly
grain mismatch between two tables) so the guardrails have something to catch
out of the box. You can also upload your own CSV/Excel file(s) instead — the
schema, the SQL generation, and all five guardrails are fully data-driven and
don't hardcode anything about the demo dataset's table or column names. The
point of this project isn't just wiring an LLM up to a database — it's
demonstrating what has to be true around that wiring for the answers to be
trustworthy, on whatever data you point it at.

## How it works

```
question
   │
   ▼
get_schema_description()                      (reads live schema from the active SQLite db - demo or uploaded)
   │
   ▼
check_undefined_metric / check_out_of_scope   (pre-generation guardrails, grounded in the active schema)
   │  clarification needed? → stop here, ask the user instead
   ▼
generate_sql()                                 (Claude → SQL SELECT statement)
   │
   ▼
check_granularity_mismatch / check_messy_categorical_filter / check_bad_join   (post-generation guardrails, sample the active data)
   │  issue found? → still execute, but attach a warning
   ▼
execute_sql_readonly()                         (runs against a read-only connection)
   │
   ▼
result
```

All of this is wired together in `pipeline/query_engine.py`'s `ask()`
function, which returns a dict with `question`, `sql`, `result`, `error`,
`clarification_needed`, and `warning`.

### Safety, layered

- The SQL-generation system prompt instructs SELECT-only output.
- `execute_sql_readonly()` independently refuses to run anything that isn't a
  single `SELECT`/`WITH` statement, regardless of what the model produced.
- The database connection itself is opened with `file:...?mode=ro` — SQLite
  will refuse writes at the OS/driver level even if the first two checks
  somehow let something through.

Neither layer trusts the layer before it.

## Project structure

```
generate_synthetic_data.py   Generates the 4 demo tables and loads them into SQLite
data/                        Generated CSVs (customers, orders, marketing_spend, support_tickets)
db/analytics.db              Generated demo SQLite database (read-only at query time)

app/
  app.py                       Streamlit front-end — demo/upload toggle over pipeline.query_engine.ask()

pipeline/
  query_engine.py            SQL generation, read-only execution, ask() (wires schema + guardrails together)
  schema_utils.py             Schema introspection (get_schema_description, etc.) — works against any SQLite file
  data_loader.py               Loads uploaded CSV/Excel files into a fresh temp SQLite database
  test_basic.py               Manual smoke test — a few easy questions, prints question/SQL/result

guardrails/
  checks.py                   The 5 guardrail checks (see below) — all data-driven, no hardcoded schema
  test_checks.py               Automated tests: deterministic for the static checks (against both the
                                demo dataset and second, differently-shaped synthetic datasets),
                                mocked-LLM + live-LLM for the two model-backed checks

tests/
  eval_questions.py            40 hand-written questions across four of the five guardrails via live
                                SQL generation, plus clean-query and edge-case questions checking for
                                guardrail over/under-triggering; each with an expected behavior and
                                rationale
  run_eval.py                  Runs every question through ask() and prints a report for
                                manual grading — does NOT auto-judge correctness
```

## The data

You can either use the bundled demo dataset or upload your own CSV/Excel
file(s) in the app — see [Bring your own data](#bring-your-own-data) below.
The rest of this section describes the demo dataset specifically.

`generate_synthetic_data.py` builds four related tables with a fixed random
seed (42), reproducible on every run:

| Table | Rows | Grain |
|---|---|---|
| `customers` | 800 | one row per customer |
| `orders` | 4,000 | one row per order (**daily**) |
| `marketing_spend` | 144 | one row per channel per month (**monthly**) |
| `support_tickets` | 600 | one row per ticket |

Three data-quality issues are baked in on purpose, and the guardrails exist
specifically because of them:

1. **Inconsistent region casing** — `customers.region` is `"West"` for ~70%
   of West-region rows and a randomly-cased/abbreviated variant
   (`"west"`, `"WEST"`, `"W"`) for the rest, independently per row.
2. **Null marketing channel** — exactly 15% of `orders.channel` values are
   `NULL`, simulating untracked/unattributed orders.
3. **Grain mismatch** — `orders` is daily grain, `marketing_spend` is
   monthly grain, and they are *not* pre-aligned. Joining them for
   channel-level ROI analysis requires explicitly rolling orders up to
   month first.

Regenerate the data (and the SQLite DB) at any time with:

```bash
python3 generate_synthetic_data.py
```

## The guardrails

All five guardrails are **data-driven**: none of them hardcode a table or
column name. They introspect and sample whichever database is active — the
demo dataset or an uploaded one — at check time.

| Check | Runs | Method | Catches |
|---|---|---|---|
| `check_undefined_metric` | before SQL generation | LLM judgment, grounded in the active schema | Ranking language ("top", "best", "leading") with no metric specified — the generator would otherwise silently pick one. |
| `check_out_of_scope` | before SQL generation | LLM judgment, grounded in the active schema | Unbounded dumps ("show me everything") or full per-row detail requested with no filter or aggregation. |
| `check_granularity_mismatch` | after SQL generation | static SQL analysis + data sample | Queries that join two tables whose date columns have different *estimated* grains (daily/weekly/monthly, inferred from the minimum gap between each table's sorted distinct dates) without an aggregation that reconciles them. |
| `check_messy_categorical_filter` | after SQL generation | static SQL analysis + data sample | Exact-match filters (e.g. `region = 'West'`) on a column whose *actual distinct values* contain case/whitespace variants that collapse to the same normalized value — auto-detected per column, not limited to `region`. |
| `check_bad_join` | after SQL generation | static SQL analysis + data sample | Joins on columns that don't look like a real relationship — unrelated names, barely-overlapping actual values, and/or mismatched declared types — rather than a genuine primary/foreign-key link. Offers the best-overlapping, name-related column pair as a suggested fix when one exists. |

The two LLM-backed checks need semantic judgment — "top 5 customers by
revenue" and "top 5 customers" differ only in whether a metric appears
further in the sentence, which a keyword blocklist can't tell apart. Their
system prompts are passed the active schema description so clarifying
questions and scoping suggestions name real tables/columns from whatever
dataset is loaded, rather than a fixed example schema.

The three post-generation checks are static analysis over the generated SQL
(which tables it joins, which columns it filters on with an unnormalized
exact match) combined with a small data sample from the active database —
enough to estimate a table's date grain, detect that a column has
inconsistent casing, or check whether a join's columns actually correspond
to each other, without needing a model call. See the module docstring
in `guardrails/checks.py` for the detection algorithm in full.

Pre-generation checks short-circuit the pipeline entirely (`sql` stays
`None`, `clarification_needed` is populated). Post-generation checks don't
block anything — the query still runs, but `warning` is populated alongside
the real result, since a slightly-off row-count is still more useful than no
answer at all as long as it's flagged.

## Known limitations

This tool focuses on a specific set of failure modes — ambiguous questions and structural data-quality issues that cause an AI-generated SQL query to silently return a misleading answer. It does not attempt to catch every possible data quality issue. Specifically, it does not currently detect:

- **Duplicate rows** — repeated records that would inflate aggregate results (e.g., double-counted orders)
- **Unit mismatches** — e.g., a column mixing dollars and cents, or metric and imperial units, without a schema-level indicator
- **Stale or incomplete data** — e.g., a table that stopped being updated, so a question like "revenue this month" silently reflects a partial or outdated load with nothing to indicate data is missing

None of these can be inferred from the schema or a small data sample the way the four existing guardrails are — catching them reliably needs metadata the source system doesn't provide here (row provenance, unit annotations, last-updated timestamps) rather than pattern detection over the query and a data sample.

## Setup

```bash
pip install anthropic pandas faker pytest streamlit openpyxl
```

(`openpyxl` is only needed to read uploaded `.xlsx`/`.xls` files.)

Set your Anthropic API key. The pipeline itself just reads it from the
environment (`anthropic.Anthropic()` looks for `ANTHROPIC_API_KEY`); the
Streamlit app (`app/app.py`) resolves it from either of two places
depending on where it's running, so the same code works locally and on
Streamlit Cloud without changes:

- **Locally**: a gitignored `.env` file.

  ```bash
  echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # gitignored, never committed
  ```

  Then, whenever you run anything that calls the API, source it into your
  shell first:

  ```bash
  set -a && source .env && set +a
  ```

- **On Streamlit Cloud**: `st.secrets["ANTHROPIC_API_KEY"]`, set via the
  app's Settings -> Secrets in the dashboard (or a local
  `.streamlit/secrets.toml` for testing that config path - also
  gitignored, never committed).

`app.py` checks `st.secrets` first and falls back to the environment, so a
`.env`-based local setup keeps working unchanged.

Generate the database (if you haven't already, or want a fresh copy):

```bash
python3 generate_synthetic_data.py
```

## Usage

**Run the web app:**

```bash
set -a && source .env && set +a && python3 -m streamlit run app/app.py
```

This prints a local URL (typically `http://localhost:8501`) to open in your
browser. At the top, a "Data source" toggle switches between the demo
dataset and your own upload; below that is a form with a question box and an
"Ask" button, and the generated SQL is shown in a collapsible section
alongside the result. In demo mode, the sidebar also shows example questions
covering each guardrail.

Use `python3 -m streamlit run ...` rather than the bare `streamlit` command
— on at least one tested setup the installed `streamlit` console script was
broken (a stale entry point referencing `streamlit.cli`, a module the
installed version no longer has). `python3 -m streamlit` invokes the package
directly and sidesteps that.

### Bring your own data

Switch the app's "Data source" toggle to "Upload your own data" and drop in
one or more `.csv`/`.xlsx`/`.xls` files. Each file becomes a table (each
sheet of a multi-sheet Excel file becomes its own table); the table name is
the filename (or `filename_sheetname`), sanitized to a valid SQL identifier
(lowercased, non-alphanumeric characters replaced with `_`, de-duplicated on
collision). The files are loaded into a fresh temporary SQLite database via
`pandas.read_csv`/`read_excel` + `to_sql` — nothing is written into the
repo's `db/` directory, and the temp file is deleted when you switch back to
the demo dataset or upload a different set of files.

Once loaded, an expander shows the detected schema (table names, column
names, inferred SQLite types), and questions are answered against that
database exactly like the demo dataset — including all five guardrails,
which re-run their table/column/date-grain detection against whatever you
uploaded. The demo dataset's example question buttons are demo-only (they
reference `region`, `orders`, `marketing_spend`, etc. by name) and are
hidden in upload mode.

**Ask a one-off question:**

```python
from pipeline.query_engine import ask

result = ask("What was total revenue in 2024?")
print(result["sql"], result["result"])
```

**Run the manual smoke test** (3 easy questions, prints question/SQL/result):

```bash
cd pipeline && python3 test_basic.py
```

**Run the guardrail test suite:**

```bash
python3 -m pytest guardrails/test_checks.py -v
```

The static-check and mocked-LLM tests run instantly with no API key. The
live-LLM tests are skipped automatically unless `ANTHROPIC_API_KEY` is set.
Most static checks run against the bundled demo database by default; a
second in-memory dataset (the `alt_db_path` fixture — different table/column
names, a different messy column, a daily-vs-weekly rather than
daily-vs-monthly grain mismatch) is built on the fly to confirm the
detection is genuinely data-driven rather than still secretly keyed to the
demo dataset's names.

**Run the full manual evaluation** (40 questions across four of the five
guardrails, plus clean-query and edge-case checks, printed as a summary
table plus full detail for you to grade yourself — this script never
judges correctness on its own):

```bash
python3 tests/run_eval.py
```

## Guardrail impact

This 40-question hand-built evaluation suite covers four of the five
guardrails via live SQL generation (undefined metrics, out-of-scope
requests, granularity mismatches, messy categorical filters), plus
clean-query and edge-case questions checking for guardrail
over/under-triggering. The fifth guardrail (bad-join detection) is
intentionally excluded — the LLM doesn't reliably produce a
structurally-invalid join from natural language, even against a schema
built to induce one — and is covered instead by dedicated unit tests
against hand-crafted SQL, including a deliberately ambiguous synthetic
dataset (see `guardrails/test_checks.py`'s `join_db_path` fixture).

To measure whether the guardrails actually matter, every question in the
suite was run twice: once through the guarded pipeline, once through an
identical pipeline with the four guardrails active at the time disabled
(the benchmark predates `check_bad_join`). Answers were checked against
hand-written ground-truth queries (for granularity and categorical-filter
issues) or graded by hand (for undefined-metric and scope issues, and for
the handful of borderline cases that can reasonably go either way).

**Without guardrails, 56% of gradable questions (15/27) received a wrong
or misleading answer — delivered silently, with no indication anything
was off. With guardrails, 100% of those wrong answers were caught**,
either flagged with a warning or intercepted with a clarifying question
before any SQL ran.

Full methodology, caveats, and per-question results:
[`tests/benchmark_summary.md`](tests/benchmark_summary.md)

## Model

Uses `claude-sonnet-4-6` for both SQL generation and the LLM-backed
guardrail checks, at `effort: "low"` — these are short, well-scoped
classification/generation tasks, not open-ended reasoning.
