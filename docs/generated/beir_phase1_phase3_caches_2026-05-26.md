# BEIR Phase 1 - Phase 3 Caches

Phase 3 built full-query BEIR retrieval and generation caches for the five Phase 1 subsets. No files under `paper/` were edited.

## Inputs

| Dataset | Eval queries | Raw retrieval cache | HyDE generation cache | SCOPE generation cache |
|---|---:|---|---|---|
| SciFact | 300 | `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl` | `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| NFCorpus | 323 | `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl` | `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| FiQA | 648 | `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl` | `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| TREC-COVID | 50 | `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl` | `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |
| SciDocs | 1000 | `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl` | `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` |

## Generation Health

Generation used `or-gemma4-26b` with `OPENROUTER_PROVIDER_ONLY=Cloudflare`, `NO_SILENT_FALLBACK=1`, `EVAL_GENERATION_FORMAT_RETRY=1`, and `EVAL_CONCURRENCY=4`. Rows were generated in deterministic chunks with `--resume`; output row order is by dataset loader order.

| Dataset | Mode | Rows | Errors | Cloudflare route rows | Parse-bad rows | Format-retry rows |
|---|---|---:|---:|---:|---:|---:|
| SciFact | HyDE | 300 | 0 | 300 | 0 | 0 |
| SciFact | SCOPE | 300 | 0 | 300 | 0 | 0 |
| NFCorpus | HyDE | 323 | 0 | 323 | 0 | 0 |
| NFCorpus | SCOPE | 323 | 0 | 323 | 0 | 0 |
| FiQA | HyDE | 648 | 0 | 648 | 0 | 0 |
| FiQA | SCOPE | 648 | 0 | 648 | 0 | 2 |
| TREC-COVID | HyDE | 50 | 0 | 50 | 0 | 0 |
| TREC-COVID | SCOPE | 50 | 0 | 50 | 0 | 0 |
| SciDocs | HyDE | 1000 | 0 | 1000 | 0 | 0 |
| SciDocs | SCOPE | 1000 | 0 | 1000 | 0 | 0 |

An earlier unpinned OpenRouter full-SCOPE attempt failed closed on an upstream DeepInfra `401` under `NO_SILENT_FALLBACK=1`. The committed generation caches supersede that attempt and are Cloudflare-pinned throughout.

## Derived Retrieval Caches

Derived retrieval used the same Chroma collections and retrieval stack as the raw-question cache: `Alibaba-NLP/gte-large-en-v1.5` dense retrieval followed by `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking, `k=10`.

| Dataset | HyDE retrieval cache | SCOPE retrieval cache |
|---|---|---|
| SciFact | `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| NFCorpus | `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| FiQA | `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| TREC-COVID | `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |
| SciDocs | `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` |

Retrieval-cache health: every raw, HyDE, and SCOPE cache has the expected row count, zero duplicate labels, zero rows with fewer than 10 retrieved ids, and zero rows without qrel gold ids. Derived HyDE/SCOPE retrieval rows record the generation cache path, provider `or-gemma4-26b`, and source mode (`rag_hyde` or `snap_hyre`) on every row.

## Retrieval Snapshot

`RI` is Collins-Thompson robustness index against raw-question retrieval at Hit@5: `(help - hurt) / N`.

| Dataset | Method | N | Hit@5 | Hit@10 | MRR@10 | RI vs raw Hit@5 | Help | Hurt | Same |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | raw | 300 | 0.820 | 0.890 | 0.693 | +0.000 | 0 | 0 | 0 |
| SciFact | HyDE | 300 | 0.350 | 0.483 | 0.249 | -0.470 | 12 | 153 | 135 |
| SciFact | SCOPE | 300 | 0.657 | 0.773 | 0.500 | -0.163 | 12 | 61 | 227 |
| NFCorpus | raw | 323 | 0.693 | 0.743 | 0.579 | +0.000 | 0 | 0 | 0 |
| NFCorpus | HyDE | 323 | 0.334 | 0.446 | 0.254 | -0.359 | 6 | 122 | 195 |
| NFCorpus | SCOPE | 323 | 0.650 | 0.746 | 0.537 | -0.043 | 20 | 34 | 269 |
| FiQA | raw | 648 | 0.662 | 0.770 | 0.510 | +0.000 | 0 | 0 | 0 |
| FiQA | HyDE | 648 | 0.323 | 0.424 | 0.210 | -0.340 | 38 | 258 | 352 |
| FiQA | SCOPE | 648 | 0.352 | 0.477 | 0.234 | -0.310 | 25 | 226 | 397 |
| TREC-COVID | raw | 50 | 0.980 | 1.000 | 0.907 | +0.000 | 0 | 0 | 0 |
| TREC-COVID | HyDE | 50 | 0.700 | 0.740 | 0.551 | -0.280 | 1 | 15 | 34 |
| TREC-COVID | SCOPE | 50 | 0.960 | 0.980 | 0.810 | -0.020 | 1 | 2 | 47 |
| SciDocs | raw | 1000 | 0.490 | 0.642 | 0.336 | +0.000 | 0 | 0 | 0 |
| SciDocs | HyDE | 1000 | 0.255 | 0.393 | 0.170 | -0.235 | 58 | 293 | 649 |
| SciDocs | SCOPE | 1000 | 0.471 | 0.604 | 0.320 | -0.019 | 87 | 106 | 807 |

## Commands

Raw retrieval caches were built with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_SILENT_FALLBACK=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset <beir_dataset> \
  --questions full \
  --seed 42 \
  --query-type raw_question \
  --max-k 10 \
  --collection <beir_collection> \
  --out caches/retrieval/full/<dataset>_qfull_seed42_raw_question_k10.jsonl \
  --resume
```

Generation caches were built in chunks with:

```bash
OPENROUTER_PROVIDER_ONLY=Cloudflare \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NO_SILENT_FALLBACK=1 EVAL_GENERATION_FORMAT_RETRY=1 EVAL_CONCURRENCY=4 \
uv run python scripts/build_generation_cache.py \
  --mode <rag_hyde|snap_hyre> \
  --provider or-gemma4-26b \
  --dataset <beir_dataset> \
  --questions full \
  --seed 42 \
  --sample-start <start> \
  --sample-end <end> \
  --out caches/generation/full/<dataset>_qfull_seed42_or-gemma4-26b_<mode>.jsonl \
  --resume \
  --concurrency 4
```

Derived retrieval caches were built with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_SILENT_FALLBACK=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset <beir_dataset> \
  --questions full \
  --seed 42 \
  --query-type <hyde_cache|hyre_cache> \
  --max-k 10 \
  --hyre-cache-path caches/generation/full/<dataset>_qfull_seed42_or-gemma4-26b_<mode>.jsonl \
  --expected-provider or-gemma4-26b \
  --out caches/retrieval/full/<dataset>_qfull_seed42_or-gemma4-26b_<mode>_k10.jsonl \
  --resume
```
