# LegalRagAgent

Legal RAG research repo for studying when retrieval helps legal QA, when it is
flat, and when it hurts.

## Start Here

Do not start from older narrative docs. The current navigation path is:

1. [`docs/README.md`](docs/README.md) - concise map of current docs, evidence
   ledgers, historical notes, and validation rules.
2. [`docs/meeting_state_2026-05-01.md`](docs/meeting_state_2026-05-01.md) -
   meeting-ready synthesis of current findings and blockers.
3. [`docs/signoff_log.md`](docs/signoff_log.md) - cite-or-not gate for result
   claims.
4. [`docs/compiled_results.md`](docs/compiled_results.md) and
   [`logs/experiments.jsonl`](logs/experiments.jsonl) - audited result ledger
   and raw run summaries.
5. [`docs/final_class_report_2026-04-30.pdf`](docs/final_class_report_2026-04-30.pdf)
   - current class-report draft.

`RESEARCH.md` and `EXPERIMENTS.md` are historical running logs. They are useful
for process and provenance, but current result claims should be checked through
the docs above.

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
docs/
  README.md                # Documentation map, start here
  signoff_log.md           # Cite-or-not gate
  meeting_state_2026-05-01.md
  compiled_results.md      # Audited result ledger
  presentation/            # Presentation docs and figures
  archive*/                # Historical docs retained for traceability
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
