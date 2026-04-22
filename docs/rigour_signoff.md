# Rigour Sign-Off Framework

Before any new eval run is trusted for the paper, these checks must pass.
Keep this doc updated as we discover new failure modes.

## Principle

We've hit three methodology bugs now (HyDE answer leak, judge INCORRECT
substring, BarExam `prompt` column dropped). Each invalidated a chunk
of prior data. To stop the cycle: **we do not generate or trust new
numbers until every known failure mode has a regression test and the
current harness is tagged at a specific commit**.

## Pre-submission checklist

Every new SLURM submission must be able to answer YES to each.

### A. Dataset integrity
- [ ] Every column in each dataset CSV is either (a) explicitly read by the
      formatter, or (b) documented as intentionally ignored with a reason
- [ ] Question + fact-pattern content is verified on 3+ sample rows per
      dataset (print the formatted prompt, eyeball that it has the
      fact-pattern-plus-question-plus-choices structure we expect)
- [ ] No silent truncation in model-input paths
      (`format_question_prompt`, `_fmt_intermediate`, any retrieved-
      passage formatter)

### B. Harness correctness
- [ ] `python tests/test_sanitizer.py` passes (sanitizer + strip + judge regression)
- [ ] Every mode runner returns the normalized schema (
      `gold_retrieved` / `retrieved_ids` / `evidence_store` always set,
      even if to defaults)
- [ ] `is_correct` derivation test passes on random sampled rows from
      the latest detail log
- [ ] HyDE/report output anti-leak on a fresh sample shows 0%
      `top_level_hyde_artifacts`

### C. Code pinning
- [ ] A specific git commit is tagged (e.g., `clean-rerun-v2`,
      `clean-rerun-v3` — incrementing each time a new bug is fixed)
- [ ] Cluster repo is at that commit (verified via `git log --oneline
      -1` on the cluster before submitting)
- [ ] SLURM script wallclock is sized for the worst mode (no more
      wallclock-kill incidents like 50867)

### D. Methodology
- [ ] Seed specified (seed=42 is the default; seed=99 is the
      repeatability seed)
- [ ] N=1195 for the full BarExam set; anything smaller is explicitly
      labeled as a smoke or N=200 mini-eval
- [ ] Across-mode comparisons use the same seed + N on the same model
      (not mixing seed=42 rag_simple with seed=99 rag_hyde)

### E. Documentation discipline
- [ ] Any number from before the current code tag is explicitly labeled
      `[pre-<tag>]` in docs and tables
- [ ] Deltas from bug fixes are quantified where possible (run the
      same N=200 subset pre and post, report the delta)
- [ ] In-flight jobs are listed with expected completion and the
      code-tag they started from

## Known bugs catalogued + their regression tests

| Date | Bug | Fix commit | Regression test |
|---|---|---|---|
| 2026-04-20 | HyDE `Answer: (X)` leak at 100%/74% | `e508765`, `951729d`, `bf89b78` | `tests/test_sanitizer.py` — 3 tests on sanitizer + strip |
| 2026-04-21 | `_judge_open_answer` scored INCORRECT as CORRECT | `7a2ee28` | `tests/test_sanitizer.py::test_open_answer_judge_handles_INCORRECT` |
| 2026-04-21 | No-retrieval modes omit `gold_retrieved`/`retrieved_ids`/`evidence_store` | `e9ed9ab` | (manual — check via `compare_mini_eval.py` output) |
| 2026-04-21 | Snap-key schema inconsistency (snap1 vs snap_answer) | `17127c0` | (no test — normalization at harness level) |
| 2026-04-21 | Entity-search hard-codes barexam paths | `17127c0` | (manual — fallback now auditable via `entity_fallback` field) |
| 2026-04-22 | BarExam `prompt` column dropped (37% of questions missing fact pattern) | `f95f316` | **ADD** `tests/test_formatter.py` — test that format_question_prompt includes prompt content when non-empty |

## Clean-rerun protocol

When we decide to run the definitive post-fix matrix:

1. Re-run every pre-submission check above. List passes/fails.
2. Tag a commit (`git tag clean-rerun-v<N>`)
3. Submit SLURM jobs at that tag.
4. Label detail logs with the tag for clarity (TAG_SUFFIX includes the version).
5. Any new number from that tag is reported with confidence intervals
   (N=1195 → ±2.8% at 95% CI).
6. Relative comparisons (mode-vs-mode, size-vs-size) report signed
   delta + confidence band. Claims need |delta| ≥ 2× the CI to be
   "significant."

## Current status (2026-04-22)

- Last known bug: `prompt` column (fixed in f95f316)
- Codex audit in flight (aaf6cb75992bbc6be) — sweeping all 5 datasets
  for similar silent-column-drop bugs
- Regression test for the prompt-column fix: **TODO** (will add now)
- Tag `clean-rerun-v1` not yet applied — waiting for Codex sweep
- 7 in-flight jobs (50858/65/68, 50986, 50991-2, 51023-4) are all
  pre-f95f316 and will be labeled as `[pre-prompt-fix]`
- Validation runs 51179 (E4B N=200) + 51180 (31B N=200) are post-f95f316
  and will give the delta. These count as pre-`clean-rerun-v1` also
  because Codex may find more issues before we tag.

## Signing off

> **I, the rigour architect, sign off on this rerun when:**
> (1) All checklist items A-E are ✓
> (2) The Codex silent-drop audit returns with no new MAJOR findings
> (3) A regression test for every known bug exists and passes
> (4) The code tag matches between local and cluster repos
