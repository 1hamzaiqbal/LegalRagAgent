# HousingQA Gemma 26B Completion Audit - 2026-05-23

This is the source-gated checkpoint for HousingQA `or-gemma4-26b` after the
May 23 completion cycle. The canonical row order is
`caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
with 6853 labels.

## Citable Rows

| Mode | Full-N value | Retrieval exposure | Source |
|---|---:|---:|---|
| `llm_only` | 3846/6853 = 56.1% | n/a | `logs/merged/housing_or-gemma4-26b_llm_only_full_20260523_114720_detail.jsonl` |
| `rag_simple` | 4531/6853 = 66.1% | Hit@5 0.3695 / MRR@5 0.2330 | `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl` |
| `rag_hyde` | 4456/6853 = 65.0% | Hit@5 0.3063 / MRR@5 0.1964 | `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl` |
| `snap_hyre` | 4458/6853 = 65.1% | Hit@5 0.3807 / MRR@5 0.2452 | `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl` |

`golden_passage` remains non-citable for HousingQA Gemma 26B: no matching full
or partial detail log exists under `logs/` or `logs/merged/`.

## Merge Provenance

`llm_only` was merged from three clean, non-overlapping canonical spans:

| Span | Rows | Correct | Route | Source |
|---|---:|---:|---|---|
| 0:10 | 10 | 6 | Cloudflare | `logs/eval_llm_only_or-gemma4-26b_20260520_060947_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl` |
| 10:3680 | 3670 | 2019 | Parasail | `logs/eval_llm_only_or-gemma4-26b_20260520_061243_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl` |
| 3680:6853 | 3173 | 1821 | Cloudflare | `logs/eval_llm_only_or-gemma4-26b_20260523_040509_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5-s3680-eend_detail.jsonl` |

Merge command:

```bash
python3 scripts/merge_detail_logs.py --key label --on-duplicate error \
  --output logs/merged/housing_or-gemma4-26b_llm_only_full_20260523_114720_detail.jsonl \
  logs/eval_llm_only_or-gemma4-26b_20260520_060947_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl \
  logs/eval_llm_only_or-gemma4-26b_20260520_061243_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl \
  logs/eval_llm_only_or-gemma4-26b_20260523_040509_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5-s3680-eend_detail.jsonl
```

Merge output:

```text
n=6853
correct=3846
accuracy=0.561214
empty_retrieval=6853
gold_retrieved=0
avg_llm_calls=1.000
avg_tokens=447.4
```

`snap_hyre` was merged earlier from canonical spans 0:3942, 3942:4793,
4793:4845, and 4845:6853 into
`logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl`.

## Audit Commands And Results

LLM-only chunk audits:

```bash
python3 scripts/audit_housing_llm_only_detail.py \
  --canonical-cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --canonical-start 0 --allow-partial --expected-rows 10 \
  logs/eval_llm_only_or-gemma4-26b_20260520_060947_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl

python3 scripts/audit_housing_llm_only_detail.py \
  --canonical-cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --canonical-start 10 --allow-partial --expected-rows 3670 \
  logs/eval_llm_only_or-gemma4-26b_20260520_061243_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5_detail.jsonl

python3 scripts/audit_housing_llm_only_detail.py \
  --canonical-cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --canonical-start 3680 --allow-partial --expected-rows 3173 \
  logs/eval_llm_only_or-gemma4-26b_20260523_040509_housing_local-snap-hyre-or-gemma4-26b-housing-llm_only-nfull-k5-s3680-eend_detail.jsonl
```

Key output: every chunk had `canonical_order_match=True`, wrong
provider/mode/dataset 0, missing predictions 0, errors 0, missing exact final
answers 0, fallback 0, think tags 0, and evidence payload 0.

Full LLM-only audit:

```bash
python3 scripts/analyze_detail_flags.py \
  logs/merged/housing_or-gemma4-26b_llm_only_full_20260523_114720_detail.jsonl

python3 scripts/audit_housing_llm_only_detail.py \
  --canonical-cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --expected-rows 6853 \
  logs/merged/housing_or-gemma4-26b_llm_only_full_20260523_114720_detail.jsonl
```

Key output:

```text
rows=6853
accuracy=3846/6853 = 0.561214
provider_route_counts={"Cloudflare": 3183, "Parasail": 3670}
canonical_order_match=True
wrong_provider=0
wrong_mode=0
wrong_dataset=0
missing_prediction=0
error=0
missing_exact_final=0
fallback=0
think_tag=0
evidence_payload=0
answer_format_retries=2
near_cap_rows=0
```

Full state-filter answer audits:

```bash
python3 scripts/analyze_detail_flags.py \
  logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl

python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode rag_simple --expected-rows 6853 \
  logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl

python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode rag_hyde --expected-rows 6853 --require-hyre-cache \
  logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl

python3 scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b --mode snap_hyre --expected-rows 6853 --require-hyre-cache \
  logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl
```

Key output:

```text
rag_simple:
rows=6853
accuracy=4531/6853 = 0.661170
gold_hit@5=2532/6853 = 0.369473
wrong_provider=0
wrong_mode=0
wrong_dataset=0
missing_state_filter=0
retrieval_cache_miss=0
doc_cache_miss=0
bad_evidence_len=0
missing_exact_final=0
fallback=0
think_tag=0

rag_hyde:
rows=6853
accuracy=4456/6853 = 0.650226
gold_hit@5=2099/6853 = 0.306289
wrong_provider=0
wrong_mode=0
wrong_dataset=0
missing_state_filter=0
retrieval_cache_miss=0
doc_cache_miss=0
hyre_cache_miss=0
bad_evidence_len=0
missing_exact_final=0
fallback=0
think_tag=0

snap_hyre:
rows=6853
accuracy=4458/6853 = 0.650518
gold_hit@5=2609/6853 = 0.380709
wrong_provider=0
wrong_mode=0
wrong_dataset=0
missing_prediction=0
error=0
missing_state_filter=0
retrieval_cache_miss=0
doc_cache_miss=0
hyre_cache_miss=0
bad_evidence_len=0
missing_exact_final=0
fallback=0
think_tag=0
```

Retrieval-cache audits:

```bash
python3 scripts/audit_retrieval_cache.py --dataset housing --ks 1,3,5,10 \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl
python3 scripts/audit_retrieval_cache.py --dataset housing --ks 1,3,5,10 \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl
python3 scripts/audit_retrieval_cache.py --dataset housing --ks 1,3,5,10 \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl
```

All three caches pass with 6853 rows, 0 duplicate keys, 0 missing indices, 0
empty retrieval rows, 0 short rows, and 0 rows without gold.

## Paper Impact

Table 1 now includes HousingQA Gemma 26B `llm_only` at 56.1% and
`snap_hyre` at 65.1%. Raw-question RAG remains the strongest deployable answer
row at 66.1%; Snap-HyRE is essentially tied with HyDE on answer accuracy
(65.1% vs 65.0%) while improving retrieval exposure over raw RAG (Hit@5 38.1%
vs 36.9%, MRR@5 24.5% vs 23.3%).

Table 2 values are unchanged because the Gemma retrieval caches were already
full-N. Figure 3 and the top-k appendix table now include the full HousingQA
Gemma 26B HyDE and Snap-HyRE caches in the HousingQA generated-query means.

## Prompt-To-Artifact Checklist

| Requirement | Evidence |
|---|---|
| Find current HousingQA Gemma 26B detail logs | Log inventory from `rg --files logs logs/merged | rg 'housing.*or-gemma4-26b.*detail\\.jsonl$'`; citable rows listed above. |
| Verify 6853 unique HousingQA rows | Full LLM-only, raw RAG, HyDE, and Snap-HyRE audits show rows 6853; the LLM-only audit also reports canonical order match. |
| Verify provider/model/mode/dataset | LLM-only and state-filter audits report wrong provider/mode/dataset 0. |
| Verify state filtering where applicable | State-filter audit reports missing state filter 0 for `rag_simple`, `rag_hyde`, and `snap_hyre`. |
| Verify no missing predictions/errors | `analyze_detail_flags.py` and audits report errors 0 and missing predictions 0. |
| Verify exact `Answer: Yes/No` format | LLM-only and state-filter audits report missing exact final answers 0. |
| Verify no silent fallback or think tags | Audits report fallback 0 and think tags 0. |
| Verify retrieval/doc/HyRE cache hits where required | State-filter audits report retrieval/doc cache misses 0 for raw RAG, HyDE, and Snap-HyRE, and HyRE cache misses 0 for HyDE/Snap-HyRE. |
| Merge partial logs only when clean | LLM-only and Snap-HyRE were merged from non-overlapping canonical spans; raw logs were not overwritten. |
| Update source-gated artifacts | `docs/signoff_log.md`, `current_status.md`, paper lineage docs, and paper tables were updated after full audits. |
| Keep retrieval exposure separate from answer accuracy | This report and the paper distinguish Table 1 answer accuracy from Table 2/Figure 3 retrieval exposure. |
