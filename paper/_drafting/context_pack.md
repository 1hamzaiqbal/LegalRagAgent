# Paper Context Pack — Diagnosing Legal RAG: Bottleneck-Aware Routing of Snap-HyRE

ICML 2026 submission. 8-page main paper (excluding references, impact statement,
appendices). Double-blind. Authors: Hamza Iqbal, Hanson Li, Josh Li
(Washington University in St. Louis). icml2026.sty auto-blinds at review time.

This pack is the **single source of truth** for any subagent writing a section.
All numbers cited in prose must come from this pack or from the source docs
listed at the bottom. Do not invent numbers. If a claim isn't here or in a
listed source doc, do not make it.

---

## A. North-star problem statement

Legal RAG is not bottlenecked by a single failure mode. Across four legal
benchmarks (BarExamQA, HousingQA, CaseHOLD, LegalBench-SCALR), the *useful
intervention* changes with the benchmark, the model, and the slice. Some tasks
need deeper retrieval; some need better query formulation; some need
jurisdiction-scoped metadata filters; some need an option-grounding or
verifier policy; some are already at the parametric ceiling and adding
retrieval hurts. A method tuned to one bottleneck systematically fails on the
others.

We operationalize this through a **source-gated diagnostic adaptation
framework**: run a small calibration trace on each benchmark slice, measure a
fixed set of signals (accuracy, gold exposure, conditional accuracy,
method-pair disagreement, parse/error counts, average LLM calls), label the
active bottleneck, and route each row to the cheapest plausible intervention
among:

- baseline retrieval,
- legal query rewrite,
- Snap-HyRE / HyRE (HyDE-style hypothetical passage generation conditioned on
  a snap answer),
- jurisdiction-scoped metadata filtering,
- option grounding,
- conservative yes/no verifier,
- disagreement arbitration,
- reject/escalate.

The earlier framing was "Snap-HyRE wins legal RAG"; the meeting-grade evidence
now refutes that universal claim and instead supports "Snap-HyRE is one route
inside a bottleneck-aware controller."

---

## B. Contribution list

1. **A bottleneck taxonomy for legal RAG.** Eight named bottlenecks—retrieval
   depth, candidate-set saturation, query/legal-reasoning formulation,
   metadata scope, statutory entailment, answer-option conversion, method
   disagreement, parametric floor—each identifiable from cheap calibration
   signals.

2. **A diagnostic adaptation framework.** A source-gated controller that types
   each benchmark from calibration traces and routes each row to the cheapest
   matching intervention. Evaluated on four legal benchmarks.

3. **Snap-HyRE positioned as one route, not the recipe.** Clean negative
   controls on HousingQA and CaseHOLD; route-instability on BarExam; the
   headline lift requires routing.

4. **Source-gated audit hygiene.** Long-answer outlier detection, OpenRouter
   max-tokens patching, multi-stage validation (`analyze_detail_flags.py` +
   `audit_adaptive_hyre_logs.py`); a methodological contribution for legal
   RAG research.

5. **Cross-model sanity transfer.** Held-out controller routes partially
   transfer to Llama~3.3~70B; the HousingQA verifier and SCALR frontier lift
   are model-stable while BarExam Snap-HyRE-v2 is not.

---

## C. Bottleneck taxonomy table (use this exactly)

| Bottleneck | Diagnostic signal | Intervention | Dominates on |
|---|---|---|---|
| Retrieval depth | Catastrophic top-1 vs top-5 drop on baseline | Larger k, smarter passage selection | (depth-sensitive datasets generally) |
| Candidate-set size then saturated | top-1 collapses; top-5/top-10 tie; gold-hit rises but answer flat | Restrict to small reranked candidate set | LegalBench-SCALR |
| Query / legal-reasoning formulation | Rewrite vs HyRE asymmetry; snap reasoning lifts > HyDE retrieval alone | `rag_rewrite` or adaptive Snap-HyRE v2; *route* between them | BarExam |
| Metadata / jurisdiction scope | Empty-retrieval or near-zero state-match rate; explicit state filter beats generic top-k | State-filtered retrieval | HousingQA |
| Statutory entailment / false-positive yes | Verifier route lifts over plain state-filter; fixed HyRE underperforms baseline | Conservative yes/no verifier on top of state-filter | HousingQA |
| Answer-option conversion | Gold-retrieval gains do not convert to answer gains (CaseHOLD gold-hit 16.0\% -> 47.0\% but EM 69.5\% -> 72.0\%, p=0.4421) | Diverse HyRE or option-table prompting (direct option-table is a clean negative) | CaseHOLD |
| Method disagreement / candidate exposure | High disagreement across HyRE-family rows; oracle row-level headroom unrealized by simple ensembles | Disagreement arbitration or frontier replay selector | LegalBench-SCALR |
| Parametric floor / depth-flat | top-1 ~ top-5 ~ llm_only; snap-only matches RAG | Trust priors; use snap reasoning for anchoring, not retrieval | BarExam |

---

## D. Calibration table (Gemma 4 26B, OpenRouter; rows 0--199, N=200 each)

Matched baseline = `rag_simple` on BarExam/CaseHOLD/SCALR, `rag_state_filter`
on HousingQA. Query-rewrite is N=50 on HousingQA / CaseHOLD / SCALR and N=200
on BarExam; flag as control.

| Method | BarExam | HousingQA | CaseHOLD | SCALR | Macro avg. | Calls |
|---|---:|---:|---:|---:|---:|---:|
| Matched baseline | 80.0 | 60.5 | 73.0 | 74.0 | 71.9 | 1.00 |
| + snap-only reasoning | 85.5 | 55.0 | 72.5 | 72.5 | 71.4 | 2.00 |
| + legal query rewrite (control) | 82.0 | 58.0\* | 72.0\* | 76.0\* | 72.0 | 2.00 |
| + preselected HyRE-family route | 86.0 | 63.5 | 73.5 | 76.0 | 74.8 | 2.00 |
| **+ diagnostic controller routes** | **86.0** | **74.5** | **73.5** | **77.5** | **77.9** | **1.30** |

\* N=50 for HousingQA/CaseHOLD/SCALR; N=200 BarExam.

### Expanded ladder per dataset

| Dataset | Baseline | Snap-only | HyRE-only | Fixed Snap-HyRE | Best controller route |
|---|---:|---:|---:|---:|---:|
| BarExam | 80.0 | 85.5 | 82.0 | 84.5 | 86.0 |
| HousingQA | 60.5 | 55.0 | 50.0 | 51.5 | 74.5 |
| CaseHOLD | 73.0 | 72.5 | 71.5 | 72.0 | 73.5 |
| LegalBench-SCALR | 74.0 | 72.5 | 74.0\*\* | 76.0 | 77.5 |

\*\* SCALR HyRE-only is the wrapper-caveated capped rerun (SLURM 67864):
detail-log clean at 148/200 = 74.0% but the SLURM wrapper failed after results
were written. The uncapped HyRE-only row is rejected because one final answer
ran away to 267,458 chars.

### CaseHOLD snap-only supersession

Capped replacement `67867` (145/200 = 72.5%) supersedes the earlier 74.0% row
which had a 41,898-char outlier. The failed intermediate `67866` remains
rejected (157,678-char answer at row 12). Replacement is clean: errors 0,
missing predictions 0, no long-answer rows.

---

## E. Held-out table (Gemma 4 26B; rows 200--249, N=50 per dataset)

| Method | BarExam | HousingQA | CaseHOLD | SCALR | Macro avg. | Calls |
|---|---:|---:|---:|---:|---:|---:|
| Held-out baseline | 76.0 | 62.0 | 68.0 | 80.0 | 71.5 | 1.00 |
| + legal query rewrite | 90.0 | 58.0 | 76.0 | 78.0 | 75.5 | 2.00 |
| + selected diagnostic routes | 76.0 | 76.0 | 78.0 | 80.0 | 77.5 | 1.54 |

Interpretation:
- HousingQA cleanest controller lift: **+14pp** over baseline (with one
  unparseable verifier answer counted wrong in the source audit).
- CaseHOLD: diverse-HyRE route adds **+10pp** over baseline and +2pp over
  rewrite.
- BarExam: route-unstable. Held-out query rewrite wins (90.0) while selected
  Snap-HyRE-v2 ties baseline (76.0). Argues for a rewrite-vs-HyRE *selector*,
  not either alone.
- SCALR: exact selected route ties baseline (80.0). Frontier component
  reaches 84.0 but the controller chooses the wrong route; routing policy
  needs refinement.

---

## F. Cross-model sanity (Groq Llama 3.3 70B; rows 200--249, N=50 per dataset)

| Dataset | Baseline | Selected route | Transfer read |
|---|---:|---:|---|
| BarExam | `rag_simple` 76.0 | `adaptive_snap_hyre_v2` 72.0 | Selected route slice-unstable. |
| HousingQA | `rag_state_filter` 44.0 | verifier 60.0 | Verifier lift transfers directionally (+16pp). |
| CaseHOLD | `rag_simple` 66.0 | diverse HyRE rejected (2 errors, 2 empty retrieval) | Health-gated; do not promote. |
| LegalBench-SCALR | `rag_simple` 82.0 | frontier 88.0 | Frontier lift transfers directionally (+6pp). |

Use as cross-model coverage sanity, not as the main table.

---

## G. Pending N>=500 jobs (mark with \pending{...} in tables)

| Job | Dataset | Modes | Status |
|---|---|---|---|
| `67897` | LegalBench-SCALR | capped full-corpus `rag_simple` + frontier | `rag_simple` half copied & validated 419/571; frontier half running |
| `67911` | BarExam | baseline/rewrite/adaptive-v2 N=500 | running |
| `67912` | HousingQA | state-filter/rewrite/verifier N=500 | running |
| `67913` | CaseHOLD | baseline/rewrite/diverse N=500 | running |
| `67915` | LegalBench-SCALR | rewrite retry N=571 (after `67914` died on bad node) | running |

Rejected (do NOT cite):
- `67828` SCALR HyRE-only uncapped (267,458-char runaway final answer).
- `67866` CaseHOLD snap-only first cap (157,678-char answer at row 12).
- `67863` full-SCALR (both halves had runaway output; rag_simple half
  structurally complete at 424/571 = 74.3\% but health-gated by three
  long-answer rows up to 233,166 chars; frontier half cancelled at row 296
  after a 232,797-char answer).
- Llama 70B CaseHOLD diverse-HyRE held-out (`67837`).

---

## H. Methodology / audit hygiene story

These were *fixed* and matter as methodological contributions:

1. **OpenRouter `max_tokens` patch.** LangChain rewrites `max_tokens` to
   `max_completion_tokens`, but OpenRouter only respects
   `extra_body.max_tokens`. Patched `llm_config.py` to send the cap through
   `extra_body`. All cite-clean replacement runs use
   `LLM_MAX_COMPLETION_TOKENS=4096`.

2. **GTE query embedding repair.** Default GTE remote-code crashed before
   retrieval due to a corrupted `position_ids` buffer; `rag_utils.py`
   reinitializes RoPE `position_ids`, sets `max_seq_length=512`, disables fp16
   by default. Smoke jobs verified finite 1024-d unit-norm embeddings.

3. **HousingQA state-filter casing fix.** Question rows store
   display-case state names (`California`); statute metadata is lowercase
   (`california`). `_housing_state_where` now lowercases the question state
   before constructing the Chroma filter. Converted the state filter from a
   parametric fallback to real retrieval (0/200 empty retrieval -> 81/200
   gold retrieval at k=5).

4. **BarExam prompt-column fix** (pre-pivot, historical). 445/1195 BarExam
   rows carry a shared fact pattern in the `prompt` column;
   `format_question_prompt` previously read only `row["question"]`. Eleven
   retrieval/rerank call sites also dropped the prompt column. All BarExam
   numbers before commit `3d5ff05` are pre-fix references.

5. **Bad cluster nodes** to exclude: `r28-1801`, `a100-2207`, `a100s-2307`,
   `a40-2206`. Always use `--exclude` for these.

6. **Validation pipeline.** Every cite-clean number must (i) come from a
   SLURM job with exit `0:0` (or wrapper-caveated detail-log-clean), (ii)
   pass `analyze_detail_flags.py` (rows match, no missing-prediction spike,
   no empty-retrieval for retrieval-bearing modes, no runaway final-answer
   length), and (iii) pass `audit_adaptive_hyre_logs.py` for HyRE-family
   rows.

---

## I. Bibliography (keys + 1-line descriptions)

All citations are in `paper/references.bib`. Use \citep{key} for the standard
parenthetical and \citet{key} for inline.

### RAG / HyDE / agentic
- `lewis2020rag` — Lewis et al. 2020 NeurIPS. RAG foundational.
- `izacard2021fid` — Izacard \& Grave 2021 EACL. Fusion-in-Decoder.
- `gao2023hyde` — Gao et al. 2023 ACL. HyDE: hypothetical document embeddings.
- `wang2023query2doc` — Wang et al. 2023 EMNLP. Query2Doc.
- `yu2023genread` — Yu et al. 2023 ICLR. GenRead generate-then-read.
- `zhou2024hyqe` — Zhou et al. 2024 EMNLP Findings. Hypothetical query embeddings.
- `jiang2023flare` — Jiang et al. 2023 EMNLP. FLARE active retrieval.
- `asai2024selfrag` — Asai et al. 2024 ICLR. Self-RAG reflection-token retrieval.
- `yan2024crag` — Yan et al. 2024. Corrective RAG with evidence grading.
- `jeong2024adaptive` — Jeong et al. 2024 NAACL. Adaptive-RAG by question complexity.
- `wang2025specrag` — Wang et al. 2025 ICLR. Speculative RAG drafter+verifier.
- `shao2023iterretgen` — Shao et al. 2023 EMNLP Findings. Iter-RetGen.
- `lei2025adaptivehyde` — Lei et al. 2025 arXiv. Adaptive HyDE / "Never Come Up Empty".

### Reasoning
- `wei2022cot` — Wei et al. 2022 NeurIPS. Chain-of-thought.
- `wang2023selfconsistency` — Wang et al. 2023 ICLR. Self-consistency voting.
- `trivedi2023ircot` — Trivedi et al. 2023 ACL. IRCoT.
- `press2023selfask` — Press et al. 2023 EMNLP Findings. Self-Ask.
- `yao2023react` — Yao et al. 2023 ICLR. ReAct.
- `gao2023rarr` — Gao et al. 2023 ACL. RARR researching-and-revising verifier.

### Legal NLP
- `guha2023legalbench` — Guha et al. 2023 NeurIPS D\&B. LegalBench (incl. SCALR).
- `zheng2021casehold` — Zheng et al. 2021 ICAIL. CaseHOLD.
- `zheng2025legalretrieval` — Zheng et al. 2025 CS\&Law. Reasoning-focused legal
  retrieval benchmark (BarExamQA + HousingQA).
- `chalkidis2022lexglue` — Chalkidis et al. 2022 ACL. LexGLUE.
- `henderson2022pileoflaw` — Henderson et al. 2022 NeurIPS D\&B. Pile of Law.
- `pipitone2024legalbenchrag` — Pipitone \& Alami 2024. LegalBench-RAG.
- `butler2026legalragbench` — Butler \& Butler 2026. Legal RAG Bench.
- `vaddi2026smallmodels` — Vaddi 2026. Task-dependent legal retrieval.
- `magesh2024hallucinations` — Magesh et al. 2024. Stanford hallucination audit.

### RAG eval / diagnostic
- `ru2024ragchecker` — Ru et al. 2024. RAGChecker fine-grained diagnostic.
- `es2024ragas` — Es et al. 2024 EACL. RAGAS automated eval.
- `saadfalcon2024ares` — Saad-Falcon et al. 2024 NAACL. ARES.
- `liu2024lostmiddle` — Liu et al. 2024 TACL. Lost in the middle.

### Multi-hop QA (reference only)
- `trivedi2022musique` — Trivedi et al. 2022 TACL. MuSiQue.

---

## J. Figures available in `paper/figures/`

- `12_diagnostic_adaptation_calibration_ablation.png` --- Calibration ablation
  table (Gemma 4 26B legal-only N=200).
- `13_diagnostic_adaptation_heldout_ablation.png` --- Held-out (rows 200-249,
  N=50 per dataset).
- `14_diagnostic_controller_macro_lift.png` --- Macro accuracy vs avg LLM
  calls; controller dominates Pareto frontier.
- `15_bottleneck_diagnostic_route_map.png` --- Evidence signal -> bottleneck
  label -> routed policy.
- `16_method_ladder_flowchart.png` --- Inherited method ladder.
- `02_barexam_cross_size.png` --- BarExam Tier 3 full-corpus cross-size lift
  (Gemma 4 26B-A4B + E4B). For appendix.
- `06_barexam_26b_full_matrix.png` --- BarExam Gemma 4 26B-A4B Tier 3 method
  matrix (N=1195). For appendix.
- `11_barexam_golden_snap_mechanism.png` --- BarExam golden-passage
  mechanism / oracle paradox.
- `01_depth_and_conversion.png` --- Legacy depth+conversion figure.
- `04_bottleneck_route_map.png` --- Older route map (use 15 in main body).

---

## K. Mandatory non-claims (do NOT promote any of these)

1. Do NOT claim Snap-HyRE / Snap-HyDE is universally better than baseline RAG
   on legal tasks. HousingQA HyRE-only 50.0; HousingQA fixed Snap-HyRE 51.5;
   CaseHOLD HyRE-only 71.5; CaseHOLD fixed Snap-HyRE 72.0. Negative controls.
2. Do NOT claim full-corpus results. No clean full-corpus number exists for
   any of the four legal benchmarks. Cell placeholders only.
3. Do NOT promote rejected SLURM rows (67828, 67866, 67863, 67837).
4. Do NOT claim a *learned* router. The controller is rule/evidence-summary
   based.
5. Do NOT claim cross-model lift uniformly transfers. BarExam Snap-HyRE-v2 is
   72.0 on Llama 70B vs 76.0 baseline.
6. Do NOT promote pure rewrite as the winning policy. On HousingQA rewrite is
   58.0 vs verifier 76.0.
7. Do NOT claim option-table prompting works on CaseHOLD. Direct option-table
   `67744` is 70.0, below rewrite 76.0 and diverse-HyRE 78.0.
8. Do NOT mix MuSiQue into the paper. This submission is legal-only.

---

## L. Source documents (for verification)

All paths relative to `/Users/hamzaiqbal/grad/LegalRagAgent/`:

- `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` — primary
- `docs/meeting_eval_expansion_status_2026-05-11.md` — pending job status
- `docs/meeting_package_audit_2026-05-11.md` — completion audit
- `docs/signoff_log.md` — cite-or-not gate
- `docs/snap_hyde_2call_2026-04-28.md` — Snap-HyRE 2-call mechanism
- `docs/top1_ablation_2026-04-28.md` — depth ablation
- `docs/casehold_repaired_rerun_2026-05-01.md` — CaseHOLD
- `docs/housing_state_filter_followup_2026-05-01.md` — HousingQA state filter
- `docs/compiled_results.md` — historical audit (BarExam Tier 3 N=1195)
- `CLAUDE.md` — methodology integrity and run-control notes
- `reports/final_class_report/main.tex` — prior class report draft (use as a
  prose source for *legal-RAG framing language*, but not for the new
  diagnostic-adaptation contribution)
