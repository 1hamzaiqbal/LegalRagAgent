# Documentation Index — LegalRagAgent

## Read this for paper-grade results (in order)

1. **`signoff_log.md`** — paper-grade citation gate. APPROVED / WITH-CAVEAT / PENDING / REJECTED per result. **Start here.**
2. **`narrative_2026_04_27.md`** — story arc: why this work, what was tried, what we found, model assessments.
3. **`mcnemar_2026-04-27.md`** — every paired McNemar test from 2026-04-27 with b/c counts and 95% CIs.
4. **`compiled_results.md`** — per-entry audited details with direct paths to detail logs, commit SHAs, CLEAN/MINOR/MAJOR audit verdicts.

## Methodology + integrity

- `rigour_signoff.md` — methodology and pre-submission checklist
- `audit_log.md` — BarExam Tier 3 post-fix source-of-truth (cluster vLLM detail logs)
- `validation_log_2026-04-25.md` — running validation log

## Live state

- `action_items.md` — current TODOs
- `experiment_overview.md` — high-level experiment summary

## Topical analyses (paper-grade, narrower scope)

- `friend_foe_bias_analysis_2026-04-27.md` — attribution-bias structured analysis
- `methods_characterization_2026-04-26.md` — earlier characterization (still authoritative for non-2026-04-27 findings)

## Archived working docs

- `archive_2026-04-27/` — 10 point-in-time docs from this session (trackers, initial audits, preliminary briefs, mechanism)
- `archive/` — older completed or legacy materials

## Cluster / HPC

- `hpc_setup_log.md`, `hpc_throughput.md`, `hpc_qwen3_8b_eval.md`, `hpc_qwen3_8b_baseline_golden.md`, `cluster_workflow.md`

## How to validate a result before citing

1. Look it up in `signoff_log.md` — find its sign-off level
2. Cross-reference in `compiled_results.md` for per-entry audit verdict + direct log path
3. For statistical claims, verify in `mcnemar_2026-04-27.md`
4. For BarExam Tier 3, the source-of-truth is `audit_log.md` (post-fix re-scored values)

If a result is not in any of those four docs at APPROVED level, do not cite.

Branch: `hpc-setup`. Source-of-truth HEAD when the latest signoff was committed: see `signoff_log.md` header.
