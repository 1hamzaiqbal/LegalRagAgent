# Writers' TODO — Diagnosing Legal RAG paper

Items that the **authors** need to think about, but that do **not** belong in
the paper itself. Kept here so the draft stays focused on the science.

Audit cadence: every time a pending row lands, every time a result claim
changes, every time someone surfaces a missing experiment, update this file.

## A. Engineering hiccups (deliberately omitted from paper)

These were real fixes during the project but they are infrastructure debt,
not contributions:

- **OpenRouter `max_tokens` cap routing**. LangChain rewrites `max_tokens`
  to `max_completion_tokens`; OpenRouter only honors `max_tokens` through
  `extra_body`. `llm_config.py` now routes through `extra_body`. Without
  this, multi-hundred-thousand-character runaway answers occurred on
  CaseHOLD and SCALR.
- **GTE `position_ids` RoPE buffer repair** in `rag_utils.py` (max_seq_length
  capped at 512, fp16 disabled by default).
- **HousingQA state-filter casing fix** (`California` vs `california`).
  Converts $0/200$ empty retrieval to $81/200$ gold retrieved at $k=5$.
  This one *could* be mentioned in the paper as "metadata casing matters
  for jurisdiction-scoped retrieval" — but stripped from the current
  draft per the May 11 editorial direction.
- **BarExam prompt-column fix** (commit `3d5ff05`, 2026-04-22). 445/1195
  rows carry the fact pattern in a separate `prompt` column; the harness
  formatter previously read only `row["question"]`. All pre-fix BarExam
  absolute numbers are invalid; relative rankings are preserved.
- **Bad cluster nodes** to exclude: `r28-1801`, `a100-2207`, `a100s-2307`,
  `a40-2206`.

## B. Audit / source-gating pipeline (deliberately omitted)

- SLURM exit `0:0` required.
- `analyze_detail_flags.py` — rows match, no missing-prediction spike, no
  empty retrieval on retrieval-bearing modes, no runaway final answer.
- `audit_adaptive_hyre_logs.py` — structural integrity of snap-answer and
  hypothetical-passage fields.
- `compute_mcnemar.py` — paired test with bootstrap CI.
- "wrapper-caveated" verdicts (detail log clean, post-processing script
  missing) — currently `67864` is the only one; cited internally only.

## C. Pending experimental work (intellectual gaps — flag in paper)

These belong in the paper as limitations or future work — but as items WE
owe, not bugs WE fixed.

- **N>=500 validation on all four legal benchmarks**. Currently running:
  - `67897` LegalBench-SCALR full-corpus capped baseline + frontier
    (`rag_simple` half at 419/571 = 73.4% locally, frontier still running).
  - `67911` BarExam baseline/rewrite/adaptive-v2 N=500.
  - `67912` HousingQA state-filter/rewrite/verifier N=500.
  - `67913` CaseHOLD baseline/rewrite/diverse N=500.
  - `67915` LegalBench-SCALR rewrite retry at N=571.

  *In the paper:* "results validated on calibration N=200 and held-out N=50;
  larger-N evaluations are in progress."

- **BarExam held-out route instability**. Calibration picks
  `adaptive_snap_hyre_v2` (86.0 vs baseline 80.0) but held-out shows query
  rewrite at 90.0 vs Snap-HyRE-v2 at 76.0. The controller picks the wrong
  family on this slice. *In the paper:* limitation — "the routing policy
  is rule-based and exhibits one held-out misroute on BarExam."

- **SCALR exact-replay route ties baseline on held-out** (80.0 vs 80.0)
  while the frontier component reaches 84.0. The selector is choosing the
  wrong arm of the frontier portfolio. *In the paper:* limitation — same
  language.

- **Cross-model CaseHOLD diverse-HyRE does not transfer cleanly** (rejected
  by health gates on Llama~70B held-out). *In the paper:* limitation —
  "cross-model coverage is partial; one of four routes did not transfer."

- **No comparison against Self-RAG, CRAG, Speculative RAG on legal**.
  Would require retraining or per-question critique infrastructure. *In the
  paper:* future work.

- **Controller is rule-based, not learned**. *In the paper:* limitation —
  "learning the routing policy from calibration signals is future work."

## D. Rejected / superseded runs (cite-internal only, NOT in paper)

These exist for our reproducibility records but should not appear in the
paper or its appendix.

- `67828` SCALR HyRE-only uncapped — 267,458-character runaway final
  answer. Rejected.
- `67866` CaseHOLD snap-only first capped attempt — 157,678-character
  answer at row 12. Rejected. Superseded by `67867` at 145/200 = 72.5%.
- `67863` full-SCALR — both halves had runaway output. Rejected.
- `67837` Llama~70B CaseHOLD diverse-HyRE held-out — 2 errors, 2 empty
  retrieval, 2 missing predictions. Rejected.
- `67914` SCALR rewrite — died on bad cluster node. Resubmitted as `67915`.

## E. Numbers to backfill when N>=500 lands

Once each pending job clears local validation, replace these in the paper
draft (currently using N=200/N=50 numbers):

- Calibration Table (sections/5Analysis.tex, `tab:calibration`):
  BarExam, HousingQA, CaseHOLD, SCALR macro at N>=500.
- Held-out Table (`tab:heldout`): same datasets at the larger slice.
- Appendix `tab:app_full_rows` rows now marked with stand-ins.

## F. Pending writer decisions

Things the writing team should decide together:

1. **Anonymity:** the current `\usepackage{icml2026}` invocation auto-blinds
   for review; names are kept in the source. Switch to `[accepted]` only
   for camera-ready.
2. **Page target:** ICML is 8 pages main + unlimited references/appendix.
   Currently overweight; we are deliberately keeping content at ~10-15
   pages so reviewers can pick which sections to compress.
3. **MuSiQue:** cut from this submission per the legal-only framing
   decision; if a reviewer asks for multi-hop evidence, mention the
   bottleneck-typed retrieval mechanism without numbers.
4. **BarExam Tier 3 N=1195 historical reference:** currently in appendix
   §B "BarExam Tier 3 Full-Corpus Reference Result." Decide if it stays
   (strongest single absolute lift) or goes (predates the diagnostic
   pivot).

## G. Things to revisit after colleague review

- Whether the paper title and abstract over- or under-claim relative to
  the actual lift size and held-out instability.
- Whether to demote SCALR/BarExam to "open routing-policy cases" earlier
  in the paper rather than burying that in §5.2.
- Whether to add a learned-router proof-of-concept before submission to
  preempt that limitation.
