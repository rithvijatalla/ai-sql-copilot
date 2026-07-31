# AI SQL Copilot

A natural-language-to-SQL analyst copilot built on top of a small, deliberately
messy synthetic analytics database. You ask a question in plain English; it
generates a SQL query with Claude, runs it against a read-only SQLite
database, and returns the result — while a set of guardrails catches the
specific ways a technically-valid query can still be a misleading answer.

The "messiness" (inconsistent region casing, null marketing channels, a
daily-vs-monthly grain mismatch between two tables) is intentional. The point
of this project isn't just wiring an LLM up to a database — it's demonstrating
what has to be true around that wiring for the answers to be trustworthy.

## How it works

```
question
   │
   ▼
check_undefined_metric / check_out_of_scope   (pre-generation guardrails)
   │  clarification needed? → stop here, ask the user instead
   ▼
get_schema_description()                      (reads live schema from SQLite)
   │
   ▼
generate_sql()                                 (Claude → SQL SELECT statement)
   │
   ▼
check_granularity_mismatch / check_messy_categorical_filter   (post-generation guardrails)
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
generate_synthetic_data.py   Generates the 4 tables and loads them into SQLite
data/                        Generated CSVs (customers, orders, marketing_spend, support_tickets)
db/analytics.db              Generated SQLite database (read-only at query time)

pipeline/
  query_engine.py            Schema introspection, SQL generation, read-only execution, ask()
  test_basic.py               Manual smoke test — a few easy questions, prints question/SQL/result

guardrails/
  checks.py                   The 4 guardrail checks (see below)
  test_checks.py               Automated tests: deterministic for the static checks,
                                mocked-LLM + live-LLM for the two model-backed checks

tests/
  eval_questions.py            ~20 hand-written questions across every guardrail category,
                                each with an expected behavior and rationale
  run_eval.py                  Runs every question through ask() and prints a report for
                                manual grading — does NOT auto-judge correctness
```

## The data

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

| Check | Runs | Method | Catches |
|---|---|---|---|
| `check_undefined_metric` | before SQL generation | LLM judgment | Ranking language ("top", "best", "leading") with no metric specified — the generator would otherwise silently pick one. |
| `check_out_of_scope` | before SQL generation | LLM judgment | Unbounded dumps ("show me everything") or full per-row detail requested with no filter or aggregation. |
| `check_granularity_mismatch` | after SQL generation | static regex | Queries that combine `orders` and `marketing_spend` without an aggregation that reconciles the daily/monthly grain mismatch. |
| `check_messy_categorical_filter` | after SQL generation | static regex | Exact-match filters on `region` (e.g. `region = 'West'`) that don't normalize casing, and so silently undercount. |

The two LLM-backed checks need semantic judgment — "top 5 customers by
revenue" and "top 5 customers" differ only in whether a metric appears
further in the sentence, which a keyword blocklist can't tell apart. The two
static checks are schema-level facts about specific columns/tables, so a
regex is faster, cheaper, and just as reliable as a model call.

Pre-generation checks short-circuit the pipeline entirely (`sql` stays
`None`, `clarification_needed` is populated). Post-generation checks don't
block anything — the query still runs, but `warning` is populated alongside
the real result, since a slightly-off row-count is still more useful than no
answer at all as long as it's flagged.

## Setup

```bash
pip install anthropic pandas faker pytest
```

Set your Anthropic API key. The pipeline reads it from the environment
(`anthropic.Anthropic()` looks for `ANTHROPIC_API_KEY`):

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # gitignored, never committed
```

Then, whenever you run anything that calls the API, source it into your
shell first:

```bash
set -a && source .env && set +a
```

Generate the database (if you haven't already, or want a fresh copy):

```bash
python3 generate_synthetic_data.py
```

## Usage

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

**Run the full manual evaluation** (~20 questions across every guardrail
category, printed as a summary table plus full detail for you to grade
yourself — this script never judges correctness on its own):

```bash
python3 tests/run_eval.py
```

## Model

Uses `claude-sonnet-4-6` for both SQL generation and the LLM-backed
guardrail checks, at `effort: "low"` — these are short, well-scoped
classification/generation tasks, not open-ended reasoning.
