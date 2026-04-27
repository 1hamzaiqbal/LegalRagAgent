# LegalRagAgent

Legal RAG research project studying **when retrieval helps legal QA and when it hurts**.

## 🎯 Start here — meeting / paper-grade reading order

| Read in order | What's there |
|---|---|
| 1. [`docs/signoff_log.md`](docs/signoff_log.md) | Cite-or-not gate: APPROVED / WITH-CAVEAT / PENDING / REJECTED per result |
| 2. [`docs/narrative_2026_04_27.md`](docs/narrative_2026_04_27.md) | Story arc: why this work, what was tried, what we found, model assessments |
| 3. [`docs/mcnemar_2026-04-27.md`](docs/mcnemar_2026-04-27.md) | Every paired McNemar test with b/c counts and 95% CIs |
| 4. [`docs/compiled_results.md`](docs/compiled_results.md) | Per-entry audited details with detail-log paths, commit SHAs, audit verdicts |

**Quick navigation: [`docs/presentation/00_index.md`](docs/presentation/00_index.md)** — landing page with Q&A cheatsheet pointing to results tables, methods explained, takeaways, datasets+models, log index, next steps, and figures.

**Temporary 2026-04-27 handoff: [`docs/meeting_notes_042726.md`](docs/meeting_notes_042726.md)** — latest meeting asks for the HPC-access agent: golden-passage sanity, top-1 vs top-5 retrieval depth, Snap-HyDE 2-call ablation, dataset×model×method planning, deadlines/authorship.

**Figures**: [`docs/presentation/figures/`](docs/presentation/figures/) — 7 PNGs (paper headline matrix, BarExam cross-size, mechanism decomposition, cross-domain specificity, cross-family check, BarExam full matrix, cost vs accuracy). Captions in [`figures/captions.md`](docs/presentation/figures/captions.md).

**Log viewer**: `python scripts/log_viewer.py` → http://localhost:8765 — drag-drop or path-entry, PASS/FAIL filter, navigation, pretty-printed records. See [`docs/presentation/log_viewer.md`](docs/presentation/log_viewer.md).

## Headline results (audited 2026-04-27)

**BarExam Tier 3 (full corpus N=1195)** — `rag_snap_hyde` wins cross-size on Gemma 4:
- Gemma 4 26B-A4B: 78.08% → **81.17%** (+3.09pp)
- Gemma 4 E4B: 58.49% → **62.18%** (+3.69pp)

**MuSiQue Tier 2 (N=200 paired McNemar) — Llama 3.3 70b dense paper headline:**
- `rag_simple` 27.5% (baseline)
- **`multi_hyde_diverse`** 35.5%, **+8.0pp p=0.0195 SIG** ✅ * (full-corpus replicate would solidify)
- `iterative_planning_table` 36.0%, +8.5pp p=0.0533 TRENDING-SIG * (full-corpus replicate would solidify)
- `subagent_rag` 15.5%, **-12.0pp p=0.0007 SIG NEGATIVE** (gap-routing over-abstains, implementation caveat)
- All other methods NS

**Mechanism (Tier 2)**: mhd's +8pp lift = ~+1.5pp from query diversity (NS) + ~+6.5pp from HyDE-style answer-bearing passages. **HyDE-style passages do ~80% of the work.**

**Cross-domain**: BarExam method (snap+HyDE) does NOT carry to MuSiQue (-3.5pp NS). MuSiQue method (mhd) does NOT carry to BarExam paired (-2.5pp NS). Methods are domain-specific.

**Cross-family caveat**: Llama 70b SIG +8pp; Gemma 3 27B NULL +2.5pp p=0.59. Multi-hop lift is not yet universal across dense families. Full-corpus runs in flight.

## Methodology hardening

- **Tier system**: N=100 directional only, N=200+ paired McNemar = citeable (Tier 2), full corpus = paper headline (Tier 3)
- **Per-entry audits**: 30 detail logs spot-checked for truncation / `<think>` leakage / empty preds / silent fallbacks (`docs/audits/`)
- **Pre-flight gate**: catches API auth fails (DeepSeek), runaway-rate limits (Venice 429), empty retrieval before logging garbage
- **Paired McNemar infrastructure**: `scripts/compute_mcnemar.py`
- **OR-Gemma serving caveat**: runaway-loop generations on iterative methods caught and documented; cluster vLLM remains gold-standard for Gemma 4

The repo contains two layers:
- `main.py` — the full LangGraph agentic pipeline / demo system
- `eval/` — the current research loop, where simpler adaptive methods are benchmarked against heavier agentic variants

**Project direction:** the long-term goal is still a strong full agentic pipeline, but the current research program is rebuilding toward it atomically from smaller, controlled retrieval strategies and only keeping improvements that survive fixed-eval scrutiny.

**Other domains (historical, not paper-grade)**: HousingQA `rag_snap_hyde` on Llama 70B = 56% (N=200); CaseHOLD `llm_only` / `confidence_gated` = 72.5% (N=200). See `RESEARCH.md` for the running state and `EXPERIMENTS.md` for the full keep/discard history.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/shrango/adaptive-plan-and-solve-agent.git
cd adaptive-plan-and-solve-agent
uv sync
```

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

### 2. Configure an LLM provider

```bash
cp .env.example .env
# Edit .env and add at least one provider API key
```

Default provider: `deepseek`. Run `uv run python llm_config.py` to list configured providers.

### 3. Download datasets

```bash
uv run python utils/download_data.py          # BarExam QA
uv run python utils/download_housingqa.py      # HousingQA
uv run python utils/download_new_datasets.py   # CaseHOLD, Legal-RAG-QA, Australian Legal QA
```

### 4. Build vector stores

```bash
uv run python utils/fast_embed.py barexam      # ~2.2 hr on RTX 3070
uv run python utils/fast_embed.py housing      # ~6 hr on RTX 3070
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py status
```

## Running

```bash
# Demo questions (full agentic pipeline)
uv run python main.py simple
uv run python main.py multi_hop
uv run python main.py medium
uv run python main.py simple --verbose

# Evals (all via eval_harness.py — 53 modes, 5 datasets)
uv run python eval/eval_harness.py --mode llm_only --provider groq-llama70b --questions 200
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset housing
uv run python eval/eval_harness.py --mode confidence_gated --provider groq-llama70b --questions 200

# List providers
uv run python llm_config.py
```

## Audited paper-grade results (full table)

For the full audited matrix — including BarExam Tier 3 N=1195 on both Gemma 4 sizes and the Llama 70b N=200 method matrix with paired McNemar p-values — see [`docs/presentation/01_results_tables.md`](docs/presentation/01_results_tables.md). Methodology and citation rules are in [`docs/signoff_log.md`](docs/signoff_log.md).

## Pipeline Architecture

Source of truth: `main.py`

```
START → router_node → planner_node → parallel_executor_node → parallel_synthesizer_node
                          ↑                                          |
                          └── parallel_replanner_node ←──────────────┘ (if incomplete)
                                                                     └→ END (if complete)
```

- **router_node** — Chooses ChromaDB collection(s): `legal_passages`, `housing_statutes`
- **planner_node** — Decomposes question into 1-5 PlanningSteps
- **parallel_executor_node** — Executes steps with per-step escalation (rag_search → web_search → direct_answer)
- **parallel_synthesizer_node** — IRAC synthesis + completeness check (max 3 rounds)

## Retrieval Stack

Source of truth: `rag_utils.py`

- ChromaDB persisted in `./chroma_db/` (configurable via `CHROMA_DB_DIR` env var)
- Default embedding: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- Embedding A/B testing: multiple models in `utils/fast_embed.py`, override via `EVAL_EMBEDDING_MODEL` env var
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Dense retrieval (k=15) → cross-encoder rerank (top 5). BM25 available but disabled by default

## Project Structure

```
main.py                    # Full pipeline: graph, nodes, executor, synthesizer (1000 LOC)
rag_utils.py               # ChromaDB, BM25, cross-encoder, multi-query retrieval
llm_config.py              # 30+ LLM provider configs, LRU-cached
web_scraper.py             # DuckDuckGo + trafilatura for web_search steps
skills/                    # 4 prompt files: planner, query_rewriter, synthesize_and_cite, synthesizer
eval/
  eval_harness.py          # Unified eval: 61 modes, 5 datasets, JSONL logging
  eval_config.py           # Config, question loaders, answer extractors
  eval_analyze.py          # Post-hoc JSONL analysis
  curate_questions.py      # One-time question curation utility
utils/
  fast_embed.py            # GPU bulk embedding with resume + A/B testing support
  build_case_summaries.py  # Case-level summary builder for structured-search experiments
  build_entity_graph.py    # Entity-graph / inverted-index builder for structured search
  download_data.py         # BarExam dataset fetcher
  download_housingqa.py    # HousingQA dataset fetcher
  download_new_datasets.py # CaseHOLD, Legal-RAG-QA, Australian Legal QA fetcher
scripts/hpc/               # SLURM job scripts for WashU HPC cluster
scripts/log_viewer.py      # Local detail-log viewer (drag-drop, PASS/FAIL filter, prev/next nav)
scripts/compute_mcnemar.py # Paired McNemar exact 2-sided test + bootstrap CI
scripts/analyze_friend_foe_bias.py  # Attribution-bias structured analysis
docs/
  README.md                # Documentation index — start here
  signoff_log.md           # Cite-or-not gate (paper-grade)
  narrative_2026_04_27.md  # Story arc + model assessments
  mcnemar_2026-04-27.md    # Paired statistical tests
  compiled_results.md      # Per-entry audited details
  audit_log.md             # BarExam Tier 3 source-of-truth (post-fix)
  rigour_signoff.md        # Methodology + pre-submission checklist
  presentation/            # 6 docs + figures/ + log_viewer guide
  audits/                  # Per-log audit reports (4 files, ~30 logs sampled)
  archive_2026-04-27/      # Superseded working docs (kept for traceability)
RESEARCH.md                # Research state, experiment queue, session handoff
EXPERIMENTS.md             # Full experiment log (hypothesis → result → verdict)
CLAUDE.md                  # Operational source of truth
tests/                     # Regression tests for formatter and sanitizer hardening
logs/                      # Eval output (gitignored)
datasets/                  # Downloaded data (gitignored)
chroma_db/                 # Vector store (gitignored)
```

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
