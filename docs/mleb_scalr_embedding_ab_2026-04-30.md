# MLEB-SCALR Embedding A/B - 2026-04-30

Purpose: keep a pure retrieval-only calibration channel separate from the QA
harness. This avoids spending LLM calls to learn that the retriever itself is
weak or insensitive.

## Runs

| Embedder | Collection | Detail log | Status |
|---|---|---|---|
| `gte-large` | `mleb_scalr_holdings` | `logs/eval_retrieval_only_mleb_scalr_gte-large_20260430_detail.jsonl` | completed |
| `all-MiniLM-L6-v2` | `mleb_scalr_holdings__all_minilm_l6_v2` | `logs/eval_retrieval_only_mleb_scalr_all-minilm_20260430_detail.jsonl` | completed |
| `bge-large` | `mleb_scalr_holdings__bge_large_en_v1_5` | - | attempted locally; stopped after model load/download blocked for ~2 min |
| `legal-bert` | `mleb_scalr_holdings__legal_bert_base_uncased` | - | attempted locally; stopped after fresh wrapper/download path blocked |

## Result

| Embedder | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `gte-large` | 34.17% | 65.00% | 72.50% | 45.77% | 52.18% |
| `all-MiniLM-L6-v2` | 31.67% | 57.50% | 67.50% | 43.10% | 48.94% |

## Read

- `gte-large` beats the speed baseline by +2.50pp Recall@1, +7.50pp
  Recall@5, and +5.00pp Recall@10.
- The gap is meaningful but not huge; retrieval variance exists, but the
  current gte-large setup is not obviously dominated by a tiny generic model.
- This is still a narrow A/B. Larger or legal-specialized retrievers should run
  on cluster or in a longer local model-download window, not inline with API
  evaluation.

## Next

Run `bge-m3`, `bge-large`, and `legal-bert` when model download time is
acceptable. Score them with the same qrels path before using any answer-quality
benchmark to judge retriever quality.
