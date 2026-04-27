# ARCHIVED 2026-04-27 — superseded by docs/README.md
# Cleanup sweep 2026-04-27

## Update 2026-04-27 ~12:30 CDT

Change reason: broad housekeeping sweep under protected-running-eval constraints. Findings:

- Eval registry: 61 registered modes; `vectorless_keyword` is the only registered mode with no `logs/experiments.jsonl` run row. Not commented in `eval/eval_config.py` because this sweep was explicitly barred from editing eval files.
- Tests: `tests/test_formatter.py` and `tests/test_sanitizer.py` are referenced by docs and pytest config; no orphaned test file found.
- Root `*.py`: `main.py`, `rag_utils.py`, `llm_config.py`, and `web_scraper.py` are all referenced/imported; no dead root Python file found.
- Broken doc paths: no missing non-log repo paths found in Markdown backtick references; log/dataset paths were left read-only.
- Stale TODO: `docs/rigour_signoff.md` said `tests/test_formatter.py` still needed to be added; updated because the test file exists.
- SLURM dedup: no exact duplicate script hashes. Superseded comments were added only where content and recent commits show one-off wrappers were replaced by parameterized/API launchers.

Files audited: `CLAUDE.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `README.md`, `docs/experiment_overview.md`, `docs/validation_log_2026-04-25.md`, `docs/action_items.md`, `docs/meeting_2026_04_17.md`, `docs/audit_log.md`, and all last-48h target rows in `logs/experiments.jsonl` with `dataset == "musique"` or mode in `multi_hyde_diverse`, `iter_hyde`, `advisor_planning_table`.

Files modified: `CLAUDE.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `README.md`, `docs/experiment_overview.md`, `docs/validation_log_2026-04-25.md`, `docs/action_items.md`, `docs/meeting_2026_04_17.md`, `docs/audit_log.md`, and this report.

Numerical disagreements found: 3 stored-vs-current-rescore MuSiQue rows, all pre-`<span>` extractor fix. `eval_rag_simple_groq-llama70b_20260426_1945_detail.jsonl` stores 2/30 but re-scores to 6/30; `eval_rag_snap_hyde_groq-llama70b_20260426_1946_detail.jsonl` stores 3/30 but re-scores to 4/30; `eval_planning_table_groq-llama70b_20260426_1947_detail.jsonl` stores 3/30 but re-scores to 4/30. No missing detail logs. No summary/detail count mismatches. The one stored `empty_retrieval_count` matched detail records.

Placeholder echoes: 3 literal `<your answer here>` echoes were found, all in pre-`0ff67ad` rows. Post-`0ff67ad` target rows had 0 literal placeholder echoes.

Bogus-row citations found: 0. The only failed row is `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50`, tagged `_FAILED-EMPTY-RETRIEVAL`; it is not cited in `CLAUDE.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `docs/experiment_overview.md`, or `docs/validation_log_2026-04-25.md`.

Stale-claim patches applied: 5 categories. Advisor N=100 framing remains cost-parity, not lift. Failed BarExam advisor row remains do-not-cite. Llama MuSiQue pre-`<span>` rows now point to re-scored values. Phase 13.5 `multi_hyde_diverse` N=100 stats are included wherever current multi-hop claims appear. Phase 14 `iter_hyde` Gemma 3 27B negative finding replaces stale open-question framing.

Confidence statement: all cited numbers in `CLAUDE.md` / `RESEARCH.md` / `docs/experiment_overview.md` / `docs/validation_log_2026-04-25.md` are now consistent with `docs/audit_log.md` as of commit `b0f84601afe93ccebaed1780c686979112cf5d90`. Local branch refs could not be updated because this environment cannot write `.git/index.lock` or any `.git/*` file; commits were built through a temporary index/object directory.

Push status: blocked in this environment by DNS/network failure (`Could not resolve host: github.com`).
