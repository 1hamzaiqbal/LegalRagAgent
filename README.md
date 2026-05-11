# LegalRagAgent

Legal RAG research repo for studying when retrieval helps legal QA, when it is
flat, and when it hurts.

## Start Here

The current navigation path, in priority order:

1. [`paper/main.pdf`](paper/main.pdf) and
   [`paper/main.tex`](paper/main.tex) - **ICML 2026 submission draft**.
   Headline: a bottleneck-aware diagnostic controller routes among baseline
   RAG, query rewrite, Snap-HyRE/HyRE, state-filter, verifier, option
   grounding, disagreement arbitration, and reject/escalate. See also
   [`paper/README.md`](paper/README.md) and
   [`paper/TODO_for_writers.md`](paper/TODO_for_writers.md) for build
   instructions and outstanding decisions.
2. [`docs/README.md`](docs/README.md) - documentation map; evidence
   ledgers, validation rules, current meeting state.
3. [`docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`](docs/meeting_prep_2026-05-11_diagnostic_adaptation.md)
   - May 11 meeting brief: bottleneck taxonomy, calibration/held-out tables,
   north-star goal.
4. [`docs/signoff_log.md`](docs/signoff_log.md) - cite-or-not gate for result
   claims.
5. [`docs/compiled_results.md`](docs/compiled_results.md) and
   [`logs/experiments.jsonl`](logs/experiments.jsonl) - audited result ledger
   and raw run summaries.

`RESEARCH.md`, `EXPERIMENTS.md`, and `reports/final_class_report/main.pdf`
are historical: useful for process and provenance, superseded for current
claims by the docs above. The pre-pivot class report's Tier-3 BarExam result
($N{=}1{,}195$, cross-size) is preserved in the paper's appendix as a
robustness check for the BarExam route.

## Current Research Frame

The active framing is a bottleneck taxonomy, not a universal new RAG recipe.
The work asks which failure mode a dataset is exposing:

- MuSiQue x Llama 70B is retrieval-depth sensitive and benefits from
  `snap_hyde_2call` at N=200.
- BarExam x Gemma 4 26B is retrieval-depth flat, so its RAG lift is better read
  as answer anchoring or evidence-use behavior rather than simple top-k recall.
- CaseHOLD and LegalBench-SCALR are option-disambiguation replicates, with
  CaseHOLD now repaired enough to show gold-retrieval movement without a
  significant answer-accuracy lift.
- HousingQA is the active metadata/filtering stress test; the first state-filter
  run failed empty retrieval and was resubmitted after a casing fix.

For numbers, p-values, and caveats, use
[`docs/signoff_log.md`](docs/signoff_log.md), not this README.

## Setup

```bash
git clone https://github.com/shrango/adaptive-plan-and-solve-agent.git
cd adaptive-plan-and-solve-agent
uv sync
```

Requires Python 3.11-3.13 and [`uv`](https://docs.astral.sh/uv/). If `uv` is
missing from PATH in this environment, use `~/.local/bin/uv`.

Configure at least one provider:

```bash
cp .env.example .env
uv run python llm_config.py
```

## Data And Vector Stores

Download local datasets:

```bash
uv run python utils/download_data.py          # BarExam QA
uv run python utils/download_housingqa.py      # HousingQA
uv run python utils/download_new_datasets.py   # CaseHOLD, Legal-RAG-QA, Australian Legal QA
```

Build Chroma collections:

```bash
uv run python utils/fast_embed.py barexam
uv run python utils/fast_embed.py housing
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py status
```

When running evals, keep offline cache flags set unless you intentionally want
to download models:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py ...
```

## Running

Demo pipeline:

```bash
uv run python main.py simple
uv run python main.py multi_hop
uv run python main.py medium
uv run python main.py simple --verbose
```

Evaluation harness:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode llm_only --provider groq-llama70b --questions 200 --dataset barexam

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset housing
```

Supported datasets are defined in `eval/eval_config.py` and currently include
`barexam`, `housing`, `legal_rag`, `australian`, `casehold`, `musique`, and
`legalbench_scalr`.

## Architecture

The repo has two main surfaces:

- `main.py` - the full LangGraph legal agent demo.
- `eval/` - the controlled research harness where retrieval variants are
  benchmarked.

Runtime graph in `main.py`:

```text
START -> router_node -> planner_node -> parallel_executor_node -> parallel_synthesizer_node
                         ^                                           |
                         +-- parallel_replanner_node <---------------+
                                                                     |
                                                                    END
```

Retrieval stack in `rag_utils.py`:

- ChromaDB under `./chroma_db/`, configurable with `CHROMA_DB_DIR`.
- Default embedding model: `Alibaba-NLP/gte-large-en-v1.5`.
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Dense retrieval plus optional BM25 pooling, deduplication, and top-k rerank.

## Repository Map

```text
main.py                    # Full LangGraph pipeline
rag_utils.py               # ChromaDB, BM25, reranking, multi-query retrieval
llm_config.py              # Provider configs and cached ChatOpenAI creation
eval/
  eval_harness.py          # Multi-method eval harness
  eval_config.py           # Dataset loading, formatting, answer extraction
  eval_analyze.py          # Post-hoc JSONL analysis
utils/
  fast_embed.py            # Bulk embedding with resume support
  download_data.py         # BarExam fetcher
  download_housingqa.py    # HousingQA fetcher
  download_new_datasets.py # CaseHOLD, Legal-RAG-QA, Australian Legal QA
scripts/
  compute_mcnemar.py       # Paired McNemar tests
  log_viewer.py            # Local JSONL detail-log viewer
  hpc/                     # SLURM scripts
paper/
  main.tex, main.pdf       # ICML 2026 submission draft
  references.bib           # 33-entry bibliography
  sections/                # 8-file split: 0Abstract..6Conclusion + Appendix
  figures/                 # Paper figures (10 PNG)
  README.md                # Build/Overleaf instructions
  TODO_for_writers.md      # Process notes, pending HPC jobs, open decisions
  diagnosing_legal_rag_overleaf.zip   # Pre-built Overleaf bundle
docs/
  README.md                # Documentation map
  signoff_log.md           # Cite-or-not gate
  meeting_prep_2026-05-11_diagnostic_adaptation.md
  meeting_state_2026-05-01.md
  compiled_results.md      # Audited result ledger
  presentation/            # Presentation figures (paper sources)
  archive*/                # Historical docs retained for traceability
reports/final_class_report/
  main.tex, main.pdf       # Pre-pivot class report (historical)
RESEARCH.md                # Historical research log
EXPERIMENTS.md             # Historical experiment chronology
CLAUDE.md                  # Agent operational context
tests/                     # Formatter and sanitizer regressions
logs/                      # Eval outputs
datasets/                  # Downloaded datasets
chroma_db/                 # Local vector stores
```

## Datasets

| Dataset | Collection / retrieval path | QA format |
|---|---|---|
| BarExam QA | `legal_passages` | Multiple choice A-D |
| HousingQA | `housing_statutes` | Yes/No |
| CaseHOLD | `casehold_holdings` | Multiple choice A-E |
| MuSiQue | in-row BM25 / `musique_passages` on cluster | Short-answer multi-hop |
| LegalBench-SCALR | `legalbench_scalr_holdings` | Multiple choice A-E |
| Legal-RAG-QA | `legal_rag_passages` | Open-ended |
| Australian Legal QA | `australian_legal` | Open-ended |

## Development Checks

```bash
uv run python tests/test_formatter.py
uv run python tests/test_sanitizer.py
git diff --check
```

Before adding new result claims, update the relevant evidence doc and then gate
the claim through [`docs/signoff_log.md`](docs/signoff_log.md).
