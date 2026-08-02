# Guarded vs. unguarded benchmark - final results

40 questions total, run against the demo dataset (`db/analytics.db`).

## By category

| Category | N | Gradable | Unguarded wrong | Guarded caught | Guarded missed |
|---|---|---|---|---|---|
| messy_categorical_filter | 7 | 7 | 7 (100%) | 7/7 | 0 |
| granularity_mismatch | 7 | 7 | 1 (14%) | 1/1 | 0 |
| undefined_metric | 7 | 7 | 5 (71%) | 5/5 | 0 |
| out_of_scope | 6 | 6 | 2 (33%) | 2/2 | 0 |

## Headline numbers

- **Without guardrails, 56% of gradable questions (15/27) got a wrong or misleading answer, delivered silently with no indication anything was off.**
- **With guardrails, 100% of those wrong answers (15/15) were caught** - either flagged with a warning (messy_categorical_filter, granularity_mismatch) or intercepted with a clarifying question before any SQL ran (undefined_metric, out_of_scope).
- Guardrails missed **0** of the wrong answers in this eval set - every case a human or ground-truth query identified as wrong was either flagged or avoided.

## Caveats and other findings (read before citing the headline number)

- **2 granularity_mismatch questions (#25, #27) were graded manually**, not by the automated ground-truth comparison, because the unguarded answer wasn't a single comparable number (a full per-order table instead of a total). #25's unguarded/guarded SQL is identical - the guardrail only adds a warning, it doesn't fix the query - and repeats one channel's *all-time* total on every matching row rather than computing the grand total asked for; graded 'silently wrong'. #27 is graded 'acceptable' because the one correct number it computed is actually right, though buried in a 4,000-row table - but note guarded didn't warn-and-answer here, it declined to generate SQL at all.
- **Possible guardrail over-triggering on out_of_scope**: questions [16, 18, 19, 20] were graded 'acceptable' - clear, bounded, single-table requests (e.g. 'dump the entire orders table', 'show every support ticket in full detail') that still triggered a clarifying question. 4 of the 6 out_of_scope questions in this eval set fall into this bucket, which suggests check_out_of_scope's current framing may skew toward flagging any full-table SELECT * as unbounded, even when the table itself is a reasonable, bounded unit of data. Worth tightening if reducing false-positive friction matters more than being conservative.
- **#14 ('best-performing channel this year') is a separate pipeline finding, not a guardrail failure**: the unguarded SQL used SQLite's real system clock (`strftime('%Y','now')`) to resolve "this year", which falls after the demo dataset's 2025 cutoff, so it silently returned zero rows. The pipeline never tells the model what "today" is - a relative-date resolution gap, unrelated to any of the four guardrails, that would affect the guarded path too if the question had been unambiguous enough to reach SQL generation.

## Informational only (not in the headline number)

- **clean** (7 questions, all designed to be unambiguous): 0 guardrail false-positives. No regression here.
- **edge_case** (6 questions, a mix of regression checks and deliberately borderline phrasing - not a clean-question set): 1 of 6 triggered a clarification (#37, "Which customers are most valuable?"), which is the question this eval set flagged in advance as genuinely ambiguous and expected to reasonably go either way - not a false positive. The other 5 edge cases (NULL handling, a 'top' phrasing with an explicit metric, zero-order customers, an average with a zero-order denominator, a 'top N by count' phrasing) all answered without any guardrail firing, as expected.