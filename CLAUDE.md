# CLAUDE.md

## Update 2026-05-11 (diagnostic-adaptation meeting package)

**Current north star**: source-gated diagnostic adaptation for legal RAG. The
meeting package frames Snap-HyRE/HyRE as one intervention family inside a
bottleneck-aware controller: calibration traces route each benchmark toward
baseline RAG, legal query rewrite, Snap-HyRE/HyRE, state metadata filtering,
option grounding, verifier policies, disagreement arbitration, or
reject/escalate.

**Start with these current docs**:
- `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` - meeting brief,
  ablation tables, bottleneck summary, diagrams, and defensible narrative.
- `docs/meeting_eval_expansion_status_2026-05-11.md` - live expansion status,
  active SLURM jobs, invalid-run exclusions, full-corpus feasibility, and
  validation gates.
- `docs/meeting_package_audit_2026-05-11.md` - requirement-by-requirement
  completion audit with exact validation commands.
- `docs/signoff_log.md` - cite-or-not gate for any reported number.

**Latest source-gated deltas**:
- Snap-only controls are complete across the four legal benchmarks with copied
  detail logs and `analyze_detail_flags.py` validation: BarExam 85.5%,
  HousingQA 55.0%, CaseHOLD 74.0%, and LegalBench-SCALR 72.5%, all at 2.00
  calls.
- BarExam HyRE-only (`rag_hyde`) is a modest positive retrieval control at
  82.0%, above baseline retrieval but below snap-only reasoning and the
  stronger fixed Snap-HyRE v2 route.
- HousingQA HyRE-only (`rag_hyde`) is a clean negative control at 50.0%,
  below snap-only, state-filter retrieval, and the verifier route.
- CaseHOLD HyRE-only (`rag_hyde`) is a weak/negative control at 71.5%, below
  current baseline retrieval, snap-only, and diverse HyRE-family rows.
- HousingQA fixed Snap-HyRE (`rag_snap_hyde_2call`) is a clean negative
  control at 51.5%; CaseHOLD fixed Snap-HyRE is weak/negative at 72.0%;
  BarExam fixed Snap-HyRE lands at 84.5%, below snap-only and adaptive
  Snap-HyRE v2.
- Groq Llama 70B held-out sanity rows are mostly clean: Housing verifier and
  SCALR frontier transfer directionally, BarExam selected route underperforms
  its baseline slice, and CaseHOLD selected route is rejected due health-gate
  failures.
- SCALR HyRE-only completed at 71.0% but is rejected as a clean report row
  because one final answer ran away to 267,458 chars / 70,593 output tokens.
  Capped rerun `67864` is live with `LLM_MAX_COMPLETION_TOKENS=4096`.
- CaseHOLD direct option-table is repaired and clean but weak at 70.0% on the
  held-out slice; cite it as an option-conversion bottleneck signal, not as a
  positive route.

**Do not promote pending rows**: SCALR capped HyRE-only rerun and the targeted
full-SCALR sanity job must
remain pending until their stdout, detail logs, and local validation all pass.
Use `docs/meeting_eval_expansion_status_2026-05-11.md` for the current queue
state before citing anything.

## Update 2026-05-01 (meeting prep)

**Current meeting frame**: lead with bottleneck-typed retrieval, not a finished
new RAG recipe. `snap_hyde_2call` is a useful fixed-cost probe, but the forward
method hypothesis is adaptive snap-HyDE: use the HyDE/reasoning budget on top of
the active bottleneck, whether that is query framing, query diversity, candidate
depth, metadata filtering, or answer-option conversion. The defensible current
contribution is the diagnostic matrix that separates retrieval depth,
candidate-set size, query formulation, evidence use, metadata filtering, and
answer-option anchoring.

**Latest source-gated deltas**:
- CaseHOLD repaired cluster job `58283` landed: `rag_simple` 69.5% vs
  `rag_snap_hyde_2call` 72.0%, +2.5pp, b/c=16/11, p=0.4421. Gold retrieval is
  now meaningful for this pair and jumps 16.0% -> 47.0%, but answer accuracy is
  still not a reliable lift. See `docs/casehold_repaired_rerun_2026-05-01.md`.
- Housing state-filter job `58282` is invalid as a method result: both k=5 and
  k=10 were `_FAILED-EMPTY-RETRIEVAL`. Root cause was state-metadata casing
  (`California` vs `california`). The fixed k=5 run `58799` landed at 61.5%,
  and the chunked k=10 completion `58937` landed at 62.5% with 0 empty
  retrieval rows and 98/200 gold retrieved. Cite as a metadata-filtering signal,
  not as a generic deeper-retrieval result. See
  `docs/housing_state_filter_followup_2026-05-01.md`.
- Top-k sensitivity should be described as a retrieval-policy stress test or
  first-pass bottleneck signal, not a complete causal diagnosis.

Current citation gates: `docs/signoff_log.md`,
`docs/meeting_state_2026-05-01.md`, `docs/snap_hyde_2call_2026-04-28.md`,
`docs/top1_ablation_2026-04-28.md`, `docs/compiled_results.md`, and
`logs/experiments.jsonl`.

## Update 2026-04-28 (post-2026-04-27 meeting)

**Headline shift**: snap_hyde_2call is the new MuSiQue Llama 70b winner at **+9.5pp p=0.008** (vs prior multi_hyde_diverse +8.0pp p=0.0195). The paper-grade story has pivoted from "multi_hyde_diverse wins multi-hop" to a **bottleneck taxonomy** measurable via a single ablation: top-1 vs top-5 retrieval depth.

**Cleanest cross-dataset evidence** (added 2026-04-28):
- MuSiQue × Llama 70b rag_simple top-1 vs top-5 = **-14.5pp p=4.18e-07** (catastrophic — retrieval-bottlenecked)
- BarExam × Gemma 4 26B rag_simple top-1 vs top-5 = **-0.5pp p=1.00 NS** (depth-flat; full-corpus lift is better read as answer anchoring / evidence use)
- ~14pp gap is the bottleneck-taxonomy signature, method-independent.

**Mechanism finding** (new): the Llama-vs-Gemma snap_hyde_2call split on MuSiQue is explained by HyDE-passage gold-recall delta. Llama 70b's HyDE passages improve gold-hit by +2.5pp; Gemma 27B's degrade it by -7.5pp. Same parametric floor (snap_only_in_final: Llama 9.5%, Gemma 9.0%), opposite retrieval and EM outcomes.

**Open**: full-corpus N=2400 Llama 70b paired blocked by Groq RPD limit (1K/day vs 4800 calls needed). Cerebras would unlock it but no API key configured.

Historical citation gates for the 2026-04-28 pivot were
`docs/signoff_log.md`, `docs/snap_hyde_2call_2026-04-28.md`,
`docs/lit_review_2026-04-28.md`, `docs/top1_ablation_2026-04-28.md`, and
`docs/meeting_notes_042726.md`; prefer the 2026-05-01 gates above for current
meeting claims.

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Environment Note

`uv` may not be on PATH in every shell. Prefer `uv` when available, otherwise fall back to `~/.local/bin/uv`.

## Project Summary

Legal RAG research repo with two distinct surfaces:
- `main.py` = the full LangGraph agentic pipeline / demo system
- `eval/` = the current research harness, where adaptive retrieval variants are compared under a fixed evaluation setup

Current research direction: the original heavy pipeline underperformed, but the long-term goal is still a strong full agentic system. For now, the project rebuilds toward that goal **atomically**: simpler adaptive retrieval strategies are the default baseline, and extra structure only stays when it proves itself in `eval/eval_harness.py`.

## Key Documentation

- `docs/README.md` — concise documentation map. Start here when deciding
  which docs are current vs historical.
- `docs/meeting_state_2026-05-01.md` — meeting-ready state of findings,
  blockers, open jobs, and defensible interpretation.
- `docs/signoff_log.md` — cite-or-not gate for result claims.
- `docs/compiled_results.md` and `logs/experiments.jsonl` — audited ledger and
  machine-readable run summaries.
- `docs/benchmark_method_birdseye_2026-04-30.md` — benchmark/method map and
  harness coverage.
- `reports/final_class_report/main.pdf` and
  `reports/final_class_report/main.tex` — current class-report draft.
- `docs/housing_state_filter_followup_2026-05-01.md` and
  `docs/casehold_repaired_rerun_2026-05-01.md` — latest dataset-specific gates.
- `docs/hpc_setup_log.md` and `docs/cluster_workflow.md` — cluster paths,
  venvs, bad nodes, and launch workflow.
- `RESEARCH.md` and `EXPERIMENTS.md` — historical running logs, not current
  claim entrypoints.

## Runtime Architecture

Source of truth: `main.py`

### Graph

```
START → router_node → planner_node → parallel_executor_node → parallel_synthesizer_node
                          ↑                                          |
                          └── parallel_replanner_node ←──────────────┘ (if incomplete)
                                                                     └→ END (if complete)
```

### Nodes

**router_node** — Lightweight LLM call to choose which ChromaDB collection(s) to search. Current registry: `legal_passages`, `housing_statutes`. Falls back to `legal_passages`.

**planner_node** — Decomposes the question into `PlanningStep`s. Outputs:
- `complexity`: `"simple"` / `"moderate"` / `"complex"` (LLM decides)
- Steps: each with `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`
- Hard cap: 5 steps max. `max_retries` capped at 3.

Loads `skills/planner.md`. Falls back to a single-step plan on parse failure.

**parallel_executor_node** — Executes ALL pending steps, each with its own internal escalation chain via `_execute_step_with_escalation()`:

Per-step execution:
- `rag_search`: LLM query rewrite → multi-query retrieval → cited synthesis → judge
- `web_search`: DuckDuckGo → scrape top 2 URLs (trafilatura) → cited synthesis → judge
- `direct_answer`: LLM answers from doctrine (no retrieval) → judge

Per-step escalation (if judge says insufficient, up to `max_retries`):
- `rag_search` → rewrite query → `direct_answer` (web skipped for doctrinal queries)
- `web_search` → `direct_answer`

**parallel_synthesizer_node** — Aggregates all completed steps into an IRAC answer (`skills/synthesizer.md`), then runs a completeness check. If gaps identified, returns `missing_topics` and routes back to replanner. Max 3 rounds.

**parallel_replanner_node** — Creates new `PlanningStep`s from the synthesizer's `missing_topics` and feeds them back to the executor.

### Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps
- `inputs` — `{"question": "..."}`
- `run_config` — `{"max_steps": 7, "max_parallel_rounds": 3}`
- `collections` — chosen search collection(s), populated by `router_node`
- `planning_table` — list of `PlanningStep`
- `evidence_store` — accumulated retrieved passages (all steps)
- `final_answer` — synthesizer output
- `audit_log` — per-node trace entries with timestamps
- `completeness_verdict` — synthesizer's completeness check result
- `parallel_round` — current planner→executor→synthesizer iteration

`PlanningStep` fields: `step_id`, `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`, `rewrite_attempt`, `status`, `result`, `confidence`, `evidence_ids`, `retry_of`, `judge_verdict`.

### Logging

Two modes controlled by `--verbose` CLI flag or `VERBOSE=1` env var:
- **Compact** (default): step breakdown with evidence source counts, LLM call/token totals
- **Verbose**: full passage text with cross-encoder scores, query rewrite alternatives, web search URLs and scraped content previews, sub-answer previews, per-LLM-call token counts

## Skills

4 prompt files in `skills/`, loaded by `main.py`:

| Skill file | Loaded as | Purpose |
|---|---|---|
| `planner.md` | `planner` | Decompose question into research steps |
| `query_rewriter.md` | `query_rewriter` | Rewrite sub-question into primary + 2 alternative queries (JSON) |
| `synthesize_and_cite.md` | `synthesize_and_cite` | Per-step cited synthesis with `[Source N]` format |
| `synthesizer.md` | `synthesizer` | Final IRAC synthesis with `[Evidence N]` citations |

## Retrieval Stack

Source of truth: `rag_utils.py`

- **Vector store**: ChromaDB persisted to `./chroma_db/`
  - `legal_passages`: 686,324 barexam passages
  - `housing_statutes`: 1,837,403 housing statutes
- **Embedding model**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid retrieval**: BM25 + bi-encoder candidates pooled, deduplicated by `idx`, cross-encoder reranks to top k
- **BM25 corpus cap**: BM25 skipped for collections >1M docs (housing_statutes uses dense-only)
- **Multi-query retrieval**: pools candidates across all query variants, deduplicates, reranks against primary query
- **Cross-step dedup**: `exclude_ids` filters out passages already retrieved in prior steps

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER` env var (default: `deepseek`)
- `get_llm()` returns a cached `ChatOpenAI` instance (LRU cache keyed on temperature + provider)
- `_llm_call()` adds retry handling (3 attempts for transient errors)
- Run `uv run python llm_config.py` to list all providers

## Commands

```bash
# Install
uv sync

# Configure
cp .env.example .env   # then add API keys

# Download datasets
uv run python utils/download_data.py                # BarExam QA
uv run python utils/download_housingqa.py           # HousingQA
uv run python utils/download_new_datasets.py        # CaseHOLD, Legal-RAG-QA, Australian Legal QA

# Build vector stores (GPU-optimized)
uv run python utils/fast_embed.py barexam           # Full barexam (~2.2 hr on RTX 3070)
uv run python utils/fast_embed.py housing           # Full housing (~6 hr on RTX 3070)
uv run python utils/fast_embed.py housing --resume  # Resume interrupted embedding
uv run python utils/fast_embed.py status            # Check collection sizes

# Run agent
uv run python main.py simple                        # Simple doctrinal question
uv run python main.py multi_hop                      # Multi-step reasoning
uv run python main.py medium                         # Medium complexity
uv run python main.py simple --verbose               # Verbose output

# Evals (all via eval_harness.py)
uv run python eval/eval_harness.py --mode llm_only --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100 --dataset housing
uv run python eval/eval_harness.py --mode golden_passage --provider groq-llama70b --questions 100

# List providers
uv run python llm_config.py

# Regression tests (sanitizer + snap stripping — lock in leak fixes)
uv run python tests/test_sanitizer.py
```

## Methodology integrity — read this BEFORE running new evals

Two harness bugs were patched on 2026-04-22 that invalidated all
prior BarExam numbers:

1. `f95f316` — `format_question_prompt` and `_fmt_intermediate` were
   reading `row["question"]` but never `row["prompt"]`. 445/1195
   BarExam rows (37%) carry a shared fact pattern in the `prompt`
   column; without it the model saw stems like
   `"Is Farmer obligated to make the $4,000 payment?"` (47 chars)
   with no facts.

2. `3d5ff05` — 11 retrieval/rerank call sites used
   `raw_question = str(row["question"])` (e.g., `snap_rag`, gap
   investigations, entity search). They also dropped the prompt
   column, so the vector store was being queried with the bare
   47-char stem too.

Every BarExam result before commit `3d5ff05` is a pre-prompt-fix
reference. Relative rankings (mode-vs-mode, size-vs-size) survive
because the bug hit all modes equally; absolute numbers do not.

Run `python tests/test_formatter.py` and `python tests/test_sanitizer.py`
before any new submission. The full pre-submission checklist lives in
`docs/rigour_signoff.md`.

## Result Snapshot / Direction Notes

**Source of truth**: `docs/signoff_log.md` for cite-or-not status,
`docs/meeting_state_2026-05-01.md` for current meeting wording,
`docs/compiled_results.md` for audit details, and
`docs/snap_hyde_2call_2026-04-28.md` /
`docs/top1_ablation_2026-04-28.md` for the bottleneck-taxonomy pivot. Numbers
below are post-fix BarExam Tier 3 or MuSiQue Tier 2 unless explicitly marked
direction-only.

### Llama 70b MuSiQue (Tier 2 N=200, current method vehicle)

| Mode | EM | Δ vs `rag_simple` | McNemar p | Verdict |
|---|---:|---:|---:|---|
| `rag_simple` | 27.5% | — | — | baseline |
| **`snap_hyde_2call`** | **37.0%** | **+9.5pp** | **0.0079** | **SIG; current MuSiQue vehicle** |
| `iterative_planning_table` | 36.0% | +8.5pp | 0.0533 | TRENDING-SIG |
| `multi_hyde_diverse` | 35.5% | +8.0pp | 0.0195 | SIG — superseded headline, still citeable |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | NS |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | NS |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | NS |
| `subagent_rag` | 15.5% | -12.0pp | 0.0007 | SIG negative |

### Gemma 4 26B-A4B BarExam (post-fix N=1195, 8/8 modes landed)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **81.17%** | **+3.09pp** ← winning method |
| `snap_only_in_final` | 80.59% | +2.51pp |
| `llm_only` | 79.75% | +1.67pp |
| `rag_hyde` | 78.91% | +0.83pp |
| `golden_passage` (oracle) | 78.66% | +0.58pp |
| `subagent_rag` | 78.16% | +0.08pp |
| `rag_simple` (baseline) | 78.08% | — |
| `subagent_hybrid` | 74.23% (rescored; stored 74.14%) | -3.85pp |

### Gemma 4 E4B BarExam (post-fix N=1195, 8 modes landed)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **62.18%** | **+3.69pp** ← same winner as 26B |
| `subagent_rag` | 60.92% | +2.43pp |
| `snap_hyde_report` | 60.75% | +2.26pp |
| `rag_hyde` | 60.59% | +2.10pp |
| `subagent_hyde` | 60.17% | +1.68pp |
| `subagent_hybrid` | 58.83% | +0.34pp |
| `rag_simple` (baseline) | 58.49% | — |
| `snap_only_in_final` | 57.82% | -0.67pp |

### Cross-family BarExam llm_only N=100 board

| Model | EM | Architecture |
|---|---|---|
| Llama 3.3 70b | 81% | 70B dense |
| **Gemma 4 26B-A4B** | **79.75%** | **25B/3.8B-active MoE** (cluster N=1195) |
| Qwen3 30B MoE | 70% | 30B/3B-active MoE (N=100) |
| Qwen3 32b dense | 68% (caveat: 13/100 truncated mid-`<think>` at 2048-token cap; true likely 70-78%) | 32B dense |
| Gemma 3 27b | 68% | 27B dense |
| Llama 4 Scout 17b | 67% | 17B MoE |

### MuSiQue (multi-hop) — older direction-only rows

| Mode | Gemma 4 26B | Gemma 3 27B | Llama 70b |
|---|---|---|---|
| `rag_simple` (baseline) | 26.7% (N=30) | **22.0%** (N=100) | **21.0%** (N=100) |
| `rag_multi_query` | 23.3% (N=30) | — | 20.0% (N=30) |
| `planning_table_no_snap` v2 | 23.3% (N=30) | — | 20.0% (N=30) |
| `iterative_planning_table` | 20.0% (N=30) | — | 23.3% (N=30) |
| `advisor_planning_table` (cheap-plan) | 23.3% (N=30) | — | 23.0% (N=100, +2pp p=0.82, cost-parity not lift) |
| **`multi_hyde_diverse`** (direction-only N=100 context) | — | **30.0% (N=100, +8pp p=0.134 trending)** | **33.0% (N=100, +12pp p=0.023)** |
| `rag_snap_hyde` | 20.0% (N=30) | — | 13.3% (N=30) |
| `golden_passage` (oracle) | 62% (N=30) | — | 47% (N=30) |

These N=100/N=30 rows are preserved as direction-only context. They are superseded for confirmed claims by the Llama 70b N=200 table above and the Gemma 3 27B N=200 NULL.

`advisor_planning_table` is a cost-parity method (86% strong-LLM input-token / 43% output-token reduction vs `iter_ptable` at parity EM, audit `a5bbd0b5840ac0da6`), not an accuracy lift method.

### Working interpretation (current)

- **Current frame:** retrieval behavior is bottleneck-typed. MuSiQue is
  retrieval-depth sensitive; BarExam is depth-flat; CaseHOLD/SCALR mainly probe
  option disambiguation and answer anchoring.
- **`rag_snap_hyde` is the proven winner on legal MC**: +3-4pp over
  `rag_simple` at both E4B and 26B sizes (clean cross-size lift, not noise).
- **Multi-hop QA split:** `snap_hyde_2call` is the current MuSiQue Llama 70B
  winner, while the earlier `multi_hyde_diverse` lift remains useful as a
  mechanism/superseded-headline result.
- **Bug-fix decomposition**: formatter (`f95f316`) added +5.44pp at 26B llm_only/snap_only_in_final identically; retrieval-query (`3d5ff05`) added +1.85pp marginal on RAG modes
- **Showing snap answer letter to the final agent always hurts** — strip the letter, keep the reasoning (regression-tested in `tests/test_sanitizer.py`)
- **Cross-model**: Gemma 4 26B-A4B beats Qwen3 30B MoE by +9.75pp at the same MoE class; +12pp over Gemma 3 27b dense

Use `docs/README.md` and `docs/meeting_state_2026-05-01.md` for the current
queue/handoff. Use `RESEARCH.md` and `EXPERIMENTS.md` as historical process logs
only.

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_harness.py` | Unified multi-model harness (65 modes, 7 datasets) |
| `eval/eval_config.py` | Config, question loading, answer extraction, EVAL_MODES dict |
| `eval/eval_analyze.py` | Post-hoc analysis of JSONL logs |
| `eval/run_experiment_queue.py` | Queue runner for batched eval submissions |
| `eval/run_embedding_comparison.py` | Embedding A/B harness for retrieval comparisons |

## Running Evals

### Environment requirements

- **HuggingFace offline mode**: HF Hub may be unreachable from this network. Always set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when running evals. The embedding model (`gte-large-en-v1.5`) is cached locally in `~/.cache/huggingface/hub/`.
- **uv**: Prefer `uv`; if it is missing from PATH in the current shell, use `~/.local/bin/uv`.
- **API keys**: All in `.env`. Groq, DeepSeek, Google, OpenRouter, OpenAI, Cerebras.

### Known Cluster Issues

- Node `r28-1801`: RTX 2080, exclude for >4B models
- Node `a100-2207`: vLLM engine init fails. Exclude.
- Node `a100s-2307`: bad vLLM node. Exclude.
- Always use `--exclude=r28-1801,a100-2207,a100s-2307`

### Launch pattern (IMPORTANT for agents)

```bash
# Single run (recommended — monitor before scaling up)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode decompose_rag --provider groq-scout --questions 200 --dataset barexam

# Background run — do NOT pipe through grep/tail (eats errors and buffers output)
# Instead, run directly and redirect to a file:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python eval/eval_harness.py \
  --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset barexam \
  > /tmp/eval_run.log 2>&1 &
```

**Common pitfalls:**
- Piping through `grep` or `tail` buffers stdout and hides errors — run failed silently
- Launching 6+ concurrent Groq calls hits rate limits (Llama: 1K RPD / 100K TPD, Scout: 1K RPD / 500K TPD)
- Parallel runs that share the same Groq model will contend on rate limits — run one per provider at a time
- Detail log and experiments.jsonl are written at END of run (not incrementally) — killing mid-run loses all results

### Groq rate limits

| Provider | Model | RPD | TPD |
|---|---|---|---|
| groq-llama70b | llama-3.3-70b-versatile | 1,000 | 100,000 |
| groq-scout | llama-4-scout-17b-16e-instruct | 1,000 | 500,000 |

Decompose_rag uses ~8 LLM calls per question (split + 3×snap + 3×hyde + synthesize). N=200 = ~1,600 calls. Fits within 1K RPD only because retry logic spaces them out — but running 2 evals on the same model will exceed limits.

### Monitoring a running eval

```bash
# Count completed questions in a background run's output
grep -c "PASS\|FAIL" /tmp/eval_run.log

# Current accuracy
echo "$(grep -c PASS /tmp/eval_run.log) / $(grep -c 'PASS\|FAIL' /tmp/eval_run.log)"

# Check for errors
grep -i "error\|traceback\|rate.limit" /tmp/eval_run.log | tail -5

# Watch for running eval processes
pgrep -a python | grep eval
```

### Output files

- **Detail log**: `logs/eval_{mode}_{provider}_{YYYYMMDD_HHMM}_detail.jsonl` — one JSON record per question
- **Summary**: appended to `logs/experiments.jsonl` — one JSON record per run
- Both written ONLY when run completes successfully

### Analyzing results

Use `eval/eval_analyze.py` for post-hoc analysis of JSONL detail logs.

```bash
# List all results for a mode
python3 -c "import json; [print(f\"{d['timestamp']} {d['mode']:25s} {d['provider']:20s} acc={d['accuracy']}  N={d['n_questions']}\") for d in (json.loads(l) for l in open('logs/experiments.jsonl')) if 'decompose' in d.get('mode','')]"
```

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| MuSiQue | in-row BM25 / `musique_passages` on cluster | per-row passages | short-answer multi-hop | local CSV build |
| LegalBench-SCALR | `legalbench_scalr_holdings` | 571 | MC (A-E) | LegalBench SCALR |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` (case_hold) |

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits
- `datasets/housing_qa/` — Statute CSVs and QA pairs
- `datasets/casehold/`, `datasets/musique/`, `datasets/legalbench_scalr/` —
  added evaluation datasets
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs

## Editing Guidance

- `main.py` is the source of truth for the pipeline (all runtime logic is currently here). Verify architecture claims against it before updating docs.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `web_scraper.py` is a standalone module (testable via CLI) imported by main.py for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — sentence-transformers with fp16 + chunked processing. Supports `--resume`.
- `vectorless` modes (`vectorless_direct`, `vectorless_role`, etc.) are multi-turn LLM reasoning, not real corpus search. The name is historical.
- Real structured search (entity graph, case summaries) is being built in `utils/build_entity_graph.py` and `utils/build_case_summaries.py`.
- Validity checklist: check answer change rate > 0%, evidence retrieval > 50%,
  snap accuracy consistency, and empty-retrieval guards. See
  `docs/rigour_signoff.md` and `docs/signoff_log.md`.
- Verify the current working branch with `git branch --show-current` before
  relying on branch-specific notes.
- Sequential pipeline code archived in branch `archive/sequential-pipeline`.
- See `docs/README.md` for the current documentation path.
