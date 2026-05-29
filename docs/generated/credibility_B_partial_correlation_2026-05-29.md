# Credibility Battery Phase B - Partial Correlation

Read-only analysis over existing affinity-margin points plus regenerated BM25 gold-affinity controls. No `paper/` files were edited.

## Verdict

- Mechanism circularity check: **survives**. Pooled gold-affinity-delta partial-R2 after CE(raw,gold) and BM25 controls is `0.096`.
- Kill criterion: below 0.05 means the gold-affinity-delta mechanism is mostly mechanical after controlling for raw closeness and BM25-space affinity.

## OLS: Retrieval Gain on Geometry and Controls

| Dataset | N | R2 | Gold-delta beta | Gold-delta partial-R2 | CE(raw,gold) beta | BM25 scope beta | BM25 raw beta |
|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | 3585 | 0.147 | 0.343 | 0.064 | 0.144 | 0.105 | 0.003 |
| FiQA | 648 | 0.275 | 0.606 | 0.102 | 0.096 | 0.017 | 0.033 |
| NFCorpus | 323 | 0.132 | 0.510 | 0.101 | 0.342 | -0.084 | -0.013 |
| SciDocs | 1000 | 0.137 | 0.482 | 0.074 | 0.170 | 0.107 | -0.015 |
| SciFact | 300 | 0.271 | 0.703 | 0.210 | 0.352 | -0.113 | 0.222 |
| TREC-COVID | 50 | 0.063 | 0.196 | 0.014 | 0.134 | 0.213 | -0.166 |
| HousingQA state-filtered | 20559 | 0.228 | 0.464 | 0.114 | 0.002 | 0.071 | -0.049 |
| Pooled | 26465 | 0.203 | 0.440 | 0.096 | 0.023 | 0.049 | -0.033 |

## P4 Failure Model With Controls

Target is `1[CE deltaM < 0]`. The controlled geometry model adds CE(raw,gold), BM25(scope,gold), and BM25(raw,gold) to the prior geometry features.

| Dataset | Feature set | N | Failures | AUC | Pseudo-R2 | Key partial-R2 |
|---|---|---:|---:|---:|---:|---|
| BarExamQA | OOV + logPPL | 3585 | 2519 | 0.588 | 0.015 | `oov_rate`=0.001; `log_perplexity`=0.007 |
| BarExamQA | Geometry + CE/BM25 controls | 3585 | 2519 | 0.954 | 0.597 | `ce_margin_raw`=0.204; `ce_scope_gold`=0.335; `ce_raw_gold`=0.000; `bm25_scope_gold`=0.000; `bm25_raw_gold`=0.004 |
| FiQA | OOV + logPPL | 648 | 423 | 0.527 | 0.003 | `oov_rate`=0.003; `log_perplexity`=0.000 |
| FiQA | Geometry + CE/BM25 controls | 648 | 423 | 0.933 | 0.520 | `ce_margin_raw`=0.190; `ce_scope_gold`=0.117; `ce_raw_gold`=0.015; `bm25_scope_gold`=0.005; `bm25_raw_gold`=0.001 |
| NFCorpus | OOV + logPPL | 307 | 188 | 0.548 | 0.008 | `oov_rate`=0.000; `log_perplexity`=0.006 |
| NFCorpus | Geometry + CE/BM25 controls | 307 | 188 | 0.896 | 0.413 | `ce_margin_raw`=0.261; `ce_scope_gold`=0.059; `ce_raw_gold`=0.011; `bm25_scope_gold`=0.025; `bm25_raw_gold`=0.000 |
| SciDocs | OOV + logPPL | 1000 | 369 | 0.507 | 0.001 | `oov_rate`=0.000; `log_perplexity`=0.000 |
| SciDocs | Geometry + CE/BM25 controls | 1000 | 369 | 0.920 | 0.480 | `ce_margin_raw`=0.265; `ce_scope_gold`=0.146; `ce_raw_gold`=0.035; `bm25_scope_gold`=0.000; `bm25_raw_gold`=0.001 |
| SciFact | OOV + logPPL | 300 | 213 | 0.572 | 0.024 | `oov_rate`=0.023; `log_perplexity`=0.001 |
| SciFact | Geometry + CE/BM25 controls | 300 | 213 | 0.918 | 0.468 | `ce_margin_raw`=0.133; `ce_scope_gold`=0.148; `ce_raw_gold`=0.070; `bm25_scope_gold`=0.005; `bm25_raw_gold`=0.000 |
| TREC-COVID | OOV + logPPL | 27 | 13 | 0.665 | 0.069 | `oov_rate`=0.000; `log_perplexity`=0.069 |
| TREC-COVID | Geometry + CE/BM25 controls | 27 | 13 | 0.912 | 0.441 | `ce_margin_raw`=0.202; `ce_scope_gold`=0.003; `ce_raw_gold`=0.001; `bm25_scope_gold`=0.047; `bm25_raw_gold`=0.068 |
| HousingQA state-filtered | OOV + logPPL | 20559 | 9766 | 0.558 | 0.015 | `oov_rate`=0.000; `log_perplexity`=0.010 |
| HousingQA state-filtered | Geometry + CE/BM25 controls | 20559 | 9766 | 0.931 | 0.516 | `ce_margin_raw`=0.285; `ce_scope_gold`=0.279; `ce_raw_gold`=0.004; `bm25_scope_gold`=0.000; `bm25_raw_gold`=0.000 |
| Pooled | OOV + logPPL | 26426 | 13491 | 0.565 | 0.013 | `oov_rate`=0.000; `log_perplexity`=0.011 |
| Pooled | Geometry + CE/BM25 controls | 26426 | 13491 | 0.912 | 0.458 | `ce_margin_raw`=0.253; `ce_scope_gold`=0.206; `ce_raw_gold`=0.000; `bm25_scope_gold`=0.005; `bm25_raw_gold`=0.000 |

## BM25 Control Notes

- BM25 controls score the raw question and SCOPE passage against the gold passage set using corpus-wide document-frequency statistics for the query terms.
- BEIR corpora are read from `datasets/beir/*/corpus.csv`; legal corpora are streamed from the Chroma collections.
- Row-level points: `docs/generated/credibility_B_partial_correlation_2026-05-29_points.jsonl`

