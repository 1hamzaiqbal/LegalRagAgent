# MuSiQue Golden-Passage Control - 2026-04-30

Purpose: test whether the BarExam golden-passage paradox generalizes to a
retrieval-depth-limited multi-hop task. It does not. On MuSiQue, providing the
labeled gold context is strongly helpful, which supports the claim that context
utility depends on the active bottleneck and model prior strength.

## Run

| Field | Value |
|---|---|
| Dataset | MuSiQue |
| Provider | `groq-llama70b` |
| Mode | `golden_passage` |
| N | 200 |
| Seed | 42 |
| Tag | `golden-musique-n200` |
| Detail log | `logs/eval_golden_passage_groq-llama70b_20260430_1515_detail.jsonl` |
| Summary row | `logs/experiments.jsonl` run `20260430_1515_golden_passage_groq-llama70b_golden-musique-n200` |

Smoke run before launch: `logs/eval_golden_passage_groq-llama70b_20260430_1513_detail.jsonl`
at N=25 scored 12/25 = 48.0%.

## Headline

`golden_passage` reaches **113/200 = 56.5% EM** on the same N=200 MuSiQue
slice used by the current Llama 70B RAG matrix.

| Comparator | Baseline | Treatment | Delta | b/c | McNemar p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| `rag_simple` -> `golden_passage` | 27.5% | 56.5% | +29.0pp | 64/6 | 2.44e-13 | [+22.0, +36.0] pp |
| `snap_hyde_2call` -> `golden_passage` | 37.0% | 56.5% | +19.5pp | 47/8 | 8.07e-08 | [+12.5, +26.0] pp |

Speculative-RAG-aligned metrics from the same logs:

| Mode | EM | F1 | Contains gold | Gold hit | Calls/q | In tok/q | Out tok/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rag_simple` | 27.5% | 36.9% | 35.0% | 84.0% | 1.00 | 833 | 84 |
| `golden_passage` | 56.5% | 69.1% | 72.0% | 100.0% | 1.00 | 440 | 43 |
| `snap_hyde_2call` | 37.0% | 48.2% | 48.0% | 86.5% | 2.00 | 1135 | 249 |

## Interpretation

- BarExam and MuSiQue disagree in the expected direction. On BarExam, the
  single gold-passage control is noisy and slightly worse than `llm_only`;
  on MuSiQue, it is a large positive control.
- The BarExam paradox should be framed as task/model-specific context
  anchoring, not as a general statement that gold context is bad.
- The MuSiQue result confirms that the multi-hop slice is still missing-fact
  and evidence-utilization limited: if the answer-bearing context is injected,
  the model gains +29pp over normal RAG.
- `golden_passage` is not a deployable retrieval method. It is a diagnostic
  control with privileged context access.

## Verification

Commands run:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode golden_passage --provider groq-llama70b --questions 200 \
  --dataset musique --seed 42 --tag golden-musique-n200

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/compute_mcnemar.py \
  logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl \
  logs/eval_golden_passage_groq-llama70b_20260430_1515_detail.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/compute_mcnemar.py \
  logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl \
  logs/eval_golden_passage_groq-llama70b_20260430_1515_detail.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 uv run python scripts/analyze_detail_flags.py \
  logs/eval_golden_passage_groq-llama70b_20260430_1515_detail.jsonl
```

Audit flags: 200 rows, 113/200 correct, no top-level HyDE/report/knowledge
artifacts reported by `scripts/analyze_detail_flags.py`.
