# Raw + SCOPE Union CE-Rerank Full-N Status - 2026-05-25

## Scope

This results-lane pass scales the deterministic selection part of the requested arm to full N on BarExamQA and HousingQA state-filtered, across `groq-llama8b`, `or-gemma4-26b`, and `groq-llama70b`.

- Candidate pool: raw-question top-10 union canonical SCOPE / `snap_hyre` top-10 from signed retrieval caches.
- Selection arms: CE-rerank to top-5 and RRF to top-5.
- Baselines: signed raw RAG and canonical SCOPE downstream detail logs; no baseline reruns.
- Paper files: no files under `paper/` were edited.

## Completion Status

Full-N retrieval selection metrics are complete for every requested dataset/model cell. Full-N downstream answer accuracy for the new CE/RRF union arms is not complete in this lane: answering both arms for all requested cells would require 48,288 new answer-model calls before retries, and no full treatment answer log exists to support full-N McNemar tests. I therefore do not substitute the q200 probe as a full-N result.

| Dataset | Rows | Models | Arms needing answer calls | New full-N treatment answer calls required | Full-N downstream status |
|---|---:|---:|---:|---:|---|
| BarExamQA | 1,195 | 3 | 2 | 7,170 | Not completed; no full CE/RRF treatment logs found |
| HousingQA state-filtered | 6,853 | 3 | 2 | 41,118 | Not completed; no full CE/RRF treatment logs found |
| **Total** | 8,048 | 3 | 2 | **48,288** | Full downstream accuracy and McNemar p-values unavailable for the new arms |

No provider rate-limit error was hit during this completed selection-metrics lane because it did not launch the full treatment answer calls. Any CE/RRF downstream numbers below are explicitly labeled as the earlier q200 Gemma probe, not a full-N substitute.

## BarExamQA Full-N Selection Metrics

Accuracy columns are signed downstream baseline accuracy. Hit/Recall/MRR columns are retrieval exposure of the final top-5 selected evidence set. CE/RRF rows have full-N retrieval exposure only; their full-N answer accuracy is not available yet.

| Model | Row | Answer acc. | Hit@5 | Recall@5 | MRR@5 | Avg pool size | Avg raw/SCOPE overlap@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Groq Llama 8B | Raw question RAG | 651/1195 (54.5%) | 1.4% | 1.4% | 0.7% |  |  |
| Groq Llama 8B | Canonical SCOPE | 680/1195 (56.9%) | 9.5% | 9.5% | 4.7% |  |  |
| Groq Llama 8B | Union + CE-rerank | -- | 2.8% | 2.8% | 0.9% | 19.71 | 0.29 |
| Groq Llama 8B | Union + RRF | -- | 5.4% | 5.4% | 2.2% | 19.71 | 0.29 |
| Gemma 4 26B | Raw question RAG | 932/1195 (78.0%) | 1.4% | 1.4% | 0.7% |  |  |
| Gemma 4 26B | Canonical SCOPE | 980/1195 (82.0%) | 12.1% | 12.1% | 6.0% |  |  |
| Gemma 4 26B | Union + CE-rerank | -- | 3.5% | 3.5% | 1.2% | 19.65 | 0.35 |
| Gemma 4 26B | Union + RRF | -- | 6.2% | 6.2% | 2.7% | 19.65 | 0.35 |
| Groq Llama 70B | Raw question RAG | 891/1195 (74.6%) | 1.4% | 1.4% | 0.7% |  |  |
| Groq Llama 70B | Canonical SCOPE | 953/1195 (79.7%) | 11.0% | 11.0% | 5.6% |  |  |
| Groq Llama 70B | Union + CE-rerank | -- | 3.3% | 3.3% | 1.2% | 19.64 | 0.36 |
| Groq Llama 70B | Union + RRF | -- | 5.9% | 5.9% | 2.5% | 19.64 | 0.36 |

## HousingQA state-filtered Full-N Selection Metrics

Accuracy columns are signed downstream baseline accuracy. Hit/Recall/MRR columns are retrieval exposure of the final top-5 selected evidence set. CE/RRF rows have full-N retrieval exposure only; their full-N answer accuracy is not available yet.

| Model | Row | Answer acc. | Hit@5 | Recall@5 | MRR@5 | Avg pool size | Avg raw/SCOPE overlap@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Groq Llama 8B | Raw question RAG | 4269/6853 (62.3%) | 36.9% | 24.1% | 23.3% |  |  |
| Groq Llama 8B | Canonical SCOPE | 4043/6853 (59.0%) | 29.6% | 18.9% | 18.9% |  |  |
| Groq Llama 8B | Union + CE-rerank | -- | 35.5% | 23.3% | 21.6% | 17.75 | 2.25 |
| Groq Llama 8B | Union + RRF | -- | 38.6% | 25.6% | 24.8% | 17.75 | 2.25 |
| Gemma 4 26B | Raw question RAG | 4531/6853 (66.1%) | 36.9% | 24.1% | 23.3% |  |  |
| Gemma 4 26B | Canonical SCOPE | 4458/6853 (65.1%) | 38.1% | 25.0% | 24.5% |  |  |
| Gemma 4 26B | Union + CE-rerank | -- | 37.8% | 24.6% | 23.0% | 17.51 | 2.49 |
| Gemma 4 26B | Union + RRF | -- | 44.2% | 29.7% | 28.9% | 17.51 | 2.49 |
| Groq Llama 70B | Raw question RAG | 4258/6853 (62.1%) | 36.9% | 24.1% | 23.3% |  |  |
| Groq Llama 70B | Canonical SCOPE | 4087/6853 (59.6%) | 23.1% | 14.7% | 14.3% |  |  |
| Groq Llama 70B | Union + CE-rerank | -- | 33.1% | 21.4% | 19.9% | 18.13 | 1.87 |
| Groq Llama 70B | Union + RRF | -- | 36.4% | 23.8% | 23.1% | 18.13 | 1.87 |

## What The Full-N Selection Pass Says

- BarExamQA: canonical SCOPE remains the strongest retrieval-exposure row among the tested top-5 sets at every model size. CE-rerank and RRF both improve over raw retrieval exposure, but neither approaches canonical SCOPE exposure on the full set.
- HousingQA: RRF is the strongest full-N retrieval-exposure selector for Gemma 26B and Llama 8B, and it is approximately raw-level on Llama 70B. CE-rerank is not the full-N retrieval-exposure winner, despite being the best q200 downstream-answer arm for Gemma.
- The earlier Housing q200 result is therefore best read as an answer-ordering/context-selection signal, not a simple Hit@5 story: CE-rerank won q200 answer accuracy while RRF and LLM-judge had stronger retrieval exposure.
- Given the call volume, the next practical downstream scale should be narrower: HousingQA Gemma CE-rerank first, then HousingQA Gemma RRF if the CE result is real. BarExamQA all-model scaling is lower priority because full-N retrieval exposure remains below canonical SCOPE.

## q200 Downstream Probe Reference

These are the earlier `or-gemma4-26b` q200 downstream rows from `docs/generated/raw_scope_union_downstream_2026-05-25.md`, included only to preserve the decision trail. They are not full-N results.

| Dataset | Arm | Correct | Accuracy | Raw q200 acc. | SCOPE q200 acc. | McNemar vs raw b/c, p | McNemar vs SCOPE b/c, p | Avg calls | Avg input toks | Avg output toks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | Union + CE-rerank | 167/200 | 83.5% | 81.0% | 88.0% | 13/8, p=0.383 | 9/18, p=0.122 | 1.00 | 1125 | 632 |
| BarExamQA | Union + RRF | 174/200 | 87.0% | 81.0% | 88.0% | 21/9, p=0.0428 | 8/10, p=0.815 | 1.00 | 1142 | 644 |
| HousingQA state-filtered | Union + CE-rerank | 130/200 | 65.0% | 62.0% | 59.0% | 15/9, p=0.307 | 21/9, p=0.0428 | 1.00 | 2787 | 479 |
| HousingQA state-filtered | Union + RRF | 121/200 | 60.5% | 62.0% | 59.0% | 12/15, p=0.701 | 12/9, p=0.664 | 1.00 | 2534 | 466 |

## Source Files

Retrieval caches and signed baseline logs used by the full-N selection pass:

- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`
- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_raw_question_k10_doc_cache.jsonl`
- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10_doc_cache.jsonl`
- `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`
- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10_doc_cache.jsonl`
- `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10_doc_cache.jsonl`
- `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
- `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl`

q200 downstream scratch rows used only for the probe-reference table:

- `/tmp/raw_scope_union_downstream_2026-05-25b_rows.jsonl`
- `/tmp/raw_scope_union_downstream_2026-05-25b_housing_rows.jsonl`

Full-N selection metric scratch payload:

- `/tmp/raw_scope_union_cererank_full_metrics_2026-05-25.json`

## Repro Notes

- Environment for selection metrics: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_MAX_CHARS=4096 CROSS_ENCODER_BATCH_SIZE=16`.
- HousingQA used the state-filtered retrieval and doc caches listed above.
- BarExamQA passage text was hydrated by cached Chroma document id lookup via `rag_utils.get_documents_by_idx`, matching the q200 probe path.
- RRF selection itself has no model-call cost, but downstream RRF answer accuracy would still require one answer call per row.
- Full-N McNemar p-values for CE/RRF cannot be computed until full treatment answer logs exist with one paired prediction per baseline label.
