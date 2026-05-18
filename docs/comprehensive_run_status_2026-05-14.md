# Comprehensive Run Status - 2026-05-14

Purpose: live checkpoint for the Snap-HyRE comprehensive evaluation. This file
tracks launch decisions, active cells, blocked rows, and signed-off rows so the
run does not drift while long API jobs execute.

## Fixed Launch Contract

- Branch: `snap_hyre_comprehensive`
- Benchmarks: BarExamQA, HousingQA, CaseHOLD, LegalBench-SCALR
- Current launch priority: finish LegalBench-SCALR first, then prioritize
  BarExamQA and CaseHOLD. HousingQA remains in scope, but is lower priority
  because of its size and format caveats unless a specific comparison needs it.
- Providers: `or-ministral-8b`, `or-gemma4-26b`, `groq-llama70b`
- Canonical modes: `llm_only`, `rag_simple`, `rag_hyde`, `snap_hyre`,
  `golden_passage`, `golden_plus_neighbors`, `rag_rewrite`
- Main answer depth: `RETRIEVAL_K=5`
- Answer cap: `LLM_MAX_COMPLETION_TOKENS=2048`
- Required guards: `EVAL_FINAL_FORMAT_RETRY=1`,
  `EVAL_GENERATION_FORMAT_RETRY=1`, `NO_SILENT_FALLBACK=1`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- Strict replay caches are required where the mode supports them.

## Dataset Sizes

| Dataset | Full rows |
|---|---:|
| BarExamQA | 1195 |
| HousingQA | 6853 |
| CaseHOLD | 3600 |
| LegalBench-SCALR | 571 |

## Cache Status

| Dataset | Cache | Status |
|---|---|---|
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_raw_question_k10.jsonl` | clean, 571 rows |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_golden_neighbors_k10.jsonl` | clean, 571 rows |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl` | clean, 571 generated rows; one explicit same-model generation-format retry on `scalr_273` for `missing_passage_block` |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` | clean, 571 rows; Hit@5 0.7268, Hit@10 0.7828, MRR@10 0.6242 |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` | clean, 571 generated rows; zero retries/errors/missing passages/artifacts |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | clean, 571 rows; Hit@5 0.7075, Hit@10 0.7688, MRR@10 0.6383 |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-ministral-8b_rag_hyde.jsonl` | clean, 571 generated rows; zero retries/errors/missing passages/fallbacks/artifacts |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-ministral-8b_rag_hyde_k10.jsonl` | clean, 571 rows; Hit@5 0.6025, Hit@10 0.6865, MRR@10 0.4506 |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-ministral-8b_snap_hyre.jsonl` | clean, 571 generated rows; zero retries/errors/missing passages/fallbacks/artifacts |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-ministral-8b_snap_hyre_k10.jsonl` | clean, 571 rows; Hit@5 0.6200, Hit@10 0.7040, MRR@10 0.5110 |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_groq-llama70b_rag_hyde.jsonl` | clean, 571 generated rows; zero retries/errors/missing passages/fallbacks/parse failures/artifacts |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | clean, 571 rows; Hit@5 0.6147, Hit@10 0.6953, MRR@10 0.5015 |
| LegalBench-SCALR | `caches/hyre/full/legalbench_scalr_qfull_seed42_groq-llama70b_snap_hyre.jsonl` | clean, 571 generated rows; zero retries/errors/missing passages/fallbacks/parse failures/artifacts |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | clean, 571 rows; Hit@5 0.5517, Hit@10 0.6462, MRR@10 0.4126 |
| BarExamQA | `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_rag_hyde.jsonl` | clean, 1195 generated rows; zero errors/missing passages/fallback keys/parse failures/answer artifacts; passage chars min/p50/p95/max 466/703/866/1169 |
| BarExamQA | `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | clean, 1195 rows; Hit@5 0.1046, Hit@10 0.1757, MRR@10 0.0609 |
| BarExamQA | `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_snap_hyre.jsonl` | clean, 1195 generated rows; zero errors/missing passages/fallback keys/parse failures/answer artifacts; passage chars min/p50/p95/max 281/447/562/666 |
| BarExamQA | `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | clean, 1195 rows; Hit@5 0.1105, Hit@10 0.1849, MRR@10 0.0663 |
| CaseHOLD | `caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl` | clean, 3600 rows |
| CaseHOLD | `caches/retrieval/full/casehold_qfull_seed42_golden_neighbors_k10.jsonl` | clean, 3600 rows |
| CaseHOLD | `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_rag_hyde.jsonl` | clean, 3600 generated rows; zero errors/missing passages/fallbacks/parse failures/answer artifacts/think artifacts; one call per row |
| CaseHOLD | `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | clean, 3600 rows; Hit@5 0.5122, Hit@10 0.5914, MRR@10 0.4090 |
| CaseHOLD | `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_snap_hyre.jsonl` | clean after strict snap-final-line repair, 3600 generated rows; zero errors/missing passages/missing snap letters/fallbacks/parse failures/answer artifacts/think artifacts; one call per row |
| CaseHOLD | `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | clean, 3600 rows; Hit@5 0.4497, Hit@10 0.5289, MRR@10 0.3390 |
| HousingQA | `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl` | copied from equivalent full raw cache, 6853 rows; qfull rebuild partial preserved as `.partial_20260514T1936Z` |
| HousingQA | `caches/retrieval/full/housing_qfull_seed42_golden_neighbors_k10.jsonl` | clean, 6853 rows; rebuilt consistently with explicit `retrieval_backend=stored_gold_embedding` and `cross_encoder_max_chars=4096` |

The qfull SCALR raw cache audited at Hit@5 0.4956 and Hit@10 0.5937. The
golden-neighbor control audited at Hit@k/MRR@k 1.0000 because gold ids are
injected before neighbor evidence.

The qfull SCALR `or-gemma4-26b` `snap_hyre` generated cache is signed off for
answer replay. Generation audit: 571/571 unique numeric indices, zero errors,
zero missing HyRE passages, zero parse failures, zero answer-artifact passages,
and one logged same-model generation-format retry on `scalr_273`
(`missing_passage_block`; input prediction `E`). Retrieval audit:
duplicate_keys=0, missing_idx=0, empty_retrieval=0, rows_shorter_than_min_k=0,
rows_without_gold=0; Hit@1 0.5517, Hit@5 0.7268, Hit@10 0.7828, and MRR@10
0.6242. Compared with raw question retrieval, `snap_hyre` improves SCALR Hit@5
by +0.2312 and Hit@10 by +0.1891 for the default Gemma 26B row.

The qfull SCALR `or-gemma4-26b` `rag_hyde` generated cache is signed off for
answer replay. Generation audit: 571/571 unique numeric indices, zero errors,
zero missing HyDE passages, zero parse failures, zero answer-artifact passages,
zero generation-format retries, and compact passage lengths
(min/p50/p95/max chars 226/376/496/551). Retrieval audit: duplicate_keys=0,
missing_idx=0, empty_retrieval=0, rows_shorter_than_min_k=0, rows_without_gold=0;
Hit@1 0.5814, Hit@5 0.7075, Hit@10 0.7688, and MRR@10 0.6383. Compared with
raw question retrieval, `rag_hyde` improves SCALR Hit@5 by +0.2119 and Hit@10
by +0.1751. Compared with `snap_hyre`, it is lower on Hit@5 and Hit@10 but
higher on MRR@10.

The qfull SCALR `or-ministral-8b` generated caches are signed off for answer
replay. `rag_hyde` generation audit: 571/571 unique indices, zero errors,
missing passages, fallbacks, parse failures, answer-artifact passages, or
retries; passage chars min/p50/p95/max 439/667/872/1043. Retrieval audit:
duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0; Hit@1 0.3415,
Hit@5 0.6025, Hit@10 0.6865, and MRR@10 0.4506. `snap_hyre` generation audit:
571/571 unique indices, zero errors, missing passages, fallbacks, parse
failures, answer-artifact passages, or retries; passage chars min/p50/p95/max
290/466/605/686. Retrieval audit: duplicate_keys=0, missing_idx=0, empty=0,
short=0, no_gold=0; Hit@1 0.4256, Hit@5 0.6200, Hit@10 0.7040, and MRR@10
0.5110. Compared with raw question retrieval, small-model `snap_hyre` improves
SCALR Hit@5 by +0.1244 and Hit@10 by +0.1103; compared with small-model
`rag_hyde`, it is +0.0175 Hit@5 and +0.0604 MRR@10.

The qfull SCALR `groq-llama70b` generated caches are signed off for answer
replay. `rag_hyde` generation audit: 571/571 unique indices, zero errors,
missing passages, fallbacks, parse failures, answer-artifact passages, or
retries; passage chars min/p50/p95/max 437/695/846/957. Retrieval audit:
duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0; Hit@1 0.4046,
Hit@5 0.6147, Hit@10 0.6953, and MRR@10 0.5015. `snap_hyre` generation audit:
571/571 unique indices, zero errors, missing passages, fallbacks, parse
failures, answer-artifact passages, or retries; passage chars min/p50/p95/max
244/461/582/714. Retrieval audit: duplicate_keys=0, missing_idx=0, empty=0,
short=0, no_gold=0; Hit@1 0.3135, Hit@5 0.5517, Hit@10 0.6462, and MRR@10
0.4126. Compared with raw question retrieval, Groq `snap_hyre` improves SCALR
Hit@5 by +0.0561 and Hit@10 by +0.0525; compared with Groq `rag_hyde`, it is
-0.0630 Hit@5 and -0.0889 MRR@10.

The qfull CaseHOLD raw cache audited at Hit@5 0.1794 and Hit@10 0.2639. The
golden-neighbor control audited at Hit@k/MRR@k 1.0000 because gold ids are
injected before neighbor evidence.

The qfull CaseHOLD `groq-llama70b` generated caches are signed off for answer
replay. `rag_hyde` generation audit: 3600/3600 rows, zero errors, missing
passages, fallbacks, parse failures, answer-artifact passages, or think
artifacts; max output 193 tokens. Retrieval audit: duplicate_keys=0,
missing_idx=0, empty=0, short=0, no_gold=0; Hit@1 0.3303, Hit@5 0.5122,
Hit@10 0.5914, and MRR@10 0.4090. `snap_hyre` generation audit after strict
snap-final-line repair: 3600/3600 rows, zero errors, missing passages, missing
snap letters, fallbacks, parse failures, answer-artifact passages, or think
artifacts; max output 304 tokens. Two malformed snap metadata rows
(`ch_test_1108`, `ch_test_3118`) were regenerated with the same provider/model
and merged before signoff. Retrieval audit: duplicate_keys=0, missing_idx=0,
empty=0, short=0, no_gold=0; Hit@1 0.2608, Hit@5 0.4497, Hit@10 0.5289, and
MRR@10 0.3390. Compared with raw question retrieval, Groq `snap_hyre` improves
CaseHOLD Hit@5 by +0.2703 and Hit@10 by +0.2650; compared with Groq
`rag_hyde`, it is -0.0625 Hit@5 and -0.0700 MRR@10.

The qfull HousingQA golden-neighbor cache audited at Hit@1 1.0000, Recall@5
0.9893, Hit@10 1.0000, and MRR@10 1.0000. The original text-embedding build
OOM-killed on the 1.8M-document Housing Chroma index. The completed cache uses
the persisted Chroma embedding for the gold statute ids as the neighbor query,
then keeps MiniLM cross-encoder reranking enabled with a 4096-character
cross-encoder-only input cap. This cap does not truncate final LLM evidence.
Cache metadata records `retrieval_backend=stored_gold_embedding` on all 6853
rows; 3588 rows had the reranker query capped and 5598 rows had at least one
candidate document capped for reranking.

## Recent Jobs

| Started UTC | Dataset | Provider | Mode | Command status |
|---|---|---|---|---|
| 2026-05-14T19:01Z | LegalBench-SCALR | `groq-llama70b` | `llm_only` | completed cleanly; moved to completed rows |
| 2026-05-14T19:14Z | LegalBench-SCALR | `or-gemma4-26b` | `llm_only` | initial run blocked on missing final `Answer:` marker; superseded by retry-hardened merged clean row |
| 2026-05-14T19:15Z | CaseHOLD | n/a | qfull raw/golden retrieval caches | completed cleanly |
| 2026-05-14T19:36Z | HousingQA | n/a | qfull golden-neighbor retrieval cache | completed cleanly; merged stored-backend chunks and audited final 6853-row cache |
| 2026-05-15T00:31Z | LegalBench-SCALR | `or-gemma4-26b` | `llm_only` row 421 and tail rerun | completed cleanly; merged full detail log moved to completed rows |
| 2026-05-15T00:58Z | LegalBench-SCALR | `or-ministral-8b` | `llm_only` | completed cleanly; moved to completed rows |
| 2026-05-15T01:46Z | LegalBench-SCALR | `or-gemma4-26b` | `rag_simple` | completed cleanly with strict raw retrieval-cache replay; moved to completed rows |
| 2026-05-15T03:15Z | LegalBench-SCALR | `or-gemma4-26b` | `golden_plus_neighbors` | completed cleanly with strict golden-neighbor cache replay; moved to completed rows |
| 2026-05-15T04:59Z | LegalBench-SCALR | `or-gemma4-26b` | `golden_passage` | completed cleanly; moved to completed rows |
| 2026-05-15T07:03Z | LegalBench-SCALR | `or-gemma4-26b` | `snap_hyre` generation/retrieval cache | completed cleanly; cache signed off for answer replay |
| 2026-05-15T07:05Z | LegalBench-SCALR | `or-gemma4-26b` | `snap_hyre` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T09:04Z | LegalBench-SCALR | `or-gemma4-26b` | `rag_hyde` generation/retrieval cache | completed cleanly; cache signed off for answer replay |
| 2026-05-15T09:27Z | LegalBench-SCALR | `or-gemma4-26b` | `rag_hyde` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T11:22Z | LegalBench-SCALR | `or-gemma4-26b` | `rag_rewrite` answer row | completed with explicit retry caveat; moved to completed rows |
| 2026-05-15T14:34Z | LegalBench-SCALR | `or-ministral-8b` | `rag_simple` answer row | strict final-line rerun completed with explicit retry/truncation caveat; moved to completed rows and supersedes the 2026-05-15T08:29Z row |
| 2026-05-15T15:26Z | LegalBench-SCALR | `or-ministral-8b` | `golden_passage` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T16:28Z | LegalBench-SCALR | `or-ministral-8b` | `golden_plus_neighbors` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T17:14Z | LegalBench-SCALR | `or-ministral-8b` | `rag_hyde`/`snap_hyre` generation/retrieval caches | completed cleanly; caches signed off for answer replay |
| 2026-05-15T18:02Z | LegalBench-SCALR | `or-ministral-8b` | `rag_hyde` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T19:02Z | LegalBench-SCALR | `or-ministral-8b` | `snap_hyre` answer row | completed with explicit retry/truncation caveat; moved to completed rows |
| 2026-05-15T19:56Z | LegalBench-SCALR | `or-ministral-8b` | `rag_rewrite` answer row | initial full/tail attempts hit no-silent rewrite JSON blockers; after explicit partial-JSON repair, merged clean full row moved to completed rows |
| 2026-05-15T22:12Z | LegalBench-SCALR | `groq-llama70b` | `rag_simple` answer row | completed cleanly with strict raw retrieval-cache replay; moved to completed rows |
| 2026-05-15T22:30Z | LegalBench-SCALR | `groq-llama70b` | `golden_passage` answer row | completed with explicit retry caveat; moved to completed rows |
| 2026-05-15T22:46Z | LegalBench-SCALR | `groq-llama70b` | `golden_plus_neighbors` answer row | completed with explicit retry caveat; moved to completed rows |
| 2026-05-15T23:01Z | LegalBench-SCALR | `groq-llama70b` | `rag_hyde`/`snap_hyre` generation/retrieval caches | completed cleanly; caches signed off for answer replay |
| 2026-05-15T23:18Z | LegalBench-SCALR | `groq-llama70b` | `rag_hyde` answer row | completed cleanly; moved to completed rows |
| 2026-05-15T23:46Z | LegalBench-SCALR | `groq-llama70b` | `snap_hyre` answer row | completed cleanly; moved to completed rows |
| 2026-05-16T00:09Z | LegalBench-SCALR | `groq-llama70b` | `rag_rewrite` answer row | completed with explicit retry caveat; moved to completed rows |
| 2026-05-16T00:15Z | BarExamQA | `or-gemma4-26b` | `llm_only` partial pacing probe | manually stopped at 9/1195 because row speed projected to multi-hour runtime; do not cite |
| 2026-05-16T00:47Z | BarExamQA | `groq-llama70b` | `llm_only` answer row | completed cleanly; moved to completed rows |
| 2026-05-16T01:26Z | BarExamQA | `groq-llama70b` | `rag_simple` answer row | completed with explicit retry caveat; downstream-negative vs `llm_only`; moved to completed rows |
| 2026-05-16T02:05Z | BarExamQA | `groq-llama70b` | `golden_plus_neighbors` answer row | completed with explicit retry caveat; positive vs `rag_simple`, flat vs `llm_only`; moved to completed rows |
| 2026-05-16T02:41Z | BarExamQA | `groq-llama70b` | `golden_passage` answer row | completed with explicit retry caveat; positive vs `rag_simple`, flat vs `llm_only`; moved to completed rows |
| 2026-05-16T03:25Z | BarExamQA | `groq-llama70b` | `rag_hyde`/`snap_hyre` generation/retrieval caches | completed cleanly; caches signed off for answer replay |
| 2026-05-16T04:03Z | BarExamQA | `groq-llama70b` | `rag_hyde` answer row | completed with explicit retry caveat; positive vs `rag_simple`, directionally above `llm_only`; moved to completed rows |
| 2026-05-16T04:40Z | BarExamQA | `groq-llama70b` | `snap_hyre` answer row | completed with explicit retry caveat; positive vs `rag_simple`, flat/slightly below `rag_hyde`; moved to completed rows |
| 2026-05-16T05:34Z | BarExamQA | `groq-llama70b` | `rag_rewrite` answer row | completed with explicit retry/repair caveat; positive vs `rag_simple`, below generated `rag_hyde`/`snap_hyre`; moved to completed rows |
| 2026-05-16T07:02Z | CaseHOLD | `groq-llama70b` | `llm_only` answer row | completed with explicit retry caveat; moved to completed rows |
| 2026-05-16T08:23Z | CaseHOLD | `groq-llama70b` | `rag_simple` answer row | completed with explicit retry caveat; downstream-flat/negative vs `llm_only`; moved to completed rows |
| 2026-05-16T09:47Z | CaseHOLD | `groq-llama70b` | `golden_passage` answer row | completed with explicit retry caveat; oracle-positive vs `rag_simple` and `llm_only`; moved to completed rows |
| 2026-05-16T10:59Z | CaseHOLD | `groq-llama70b` | `golden_plus_neighbors` answer row | completed with explicit retry caveat; positive vs raw/LLM-only but far below gold-only; moved to completed rows |
| 2026-05-16T12:43Z | CaseHOLD | `groq-llama70b` | `rag_hyde`/`snap_hyre` generation/retrieval caches | completed; `rag_hyde` clean immediately, `snap_hyre` required strict two-row snap-final-line repair before signoff |
| 2026-05-16T12:49Z | CaseHOLD | `groq-llama70b` | `snap_hyre` strict cache repair | completed cleanly; regenerated `ch_test_1108` and `ch_test_3118`, merged generation/retrieval rows, and recompiled Groq retrieval matrix |
| 2026-05-16T12:52Z | CaseHOLD | `groq-llama70b` / `or-llama70b-paid` | `rag_hyde` answer row | Groq stopped on `spend_limit_reached` after 2639 valid rows; explicitly replayed `ch_test_2639` and the remaining tail on the same Llama 3.3 70B model through OpenRouter, with final segment pinned to `OPENROUTER_PROVIDER_ONLY=AkashML`; merged full row moved to completed rows |
| 2026-05-16T16:07Z | CaseHOLD | `or-llama70b-paid` / `groq-llama70b` | `snap_hyre` answer row | initial same-model OpenRouter prefix hit the strict no-silent snap-final-line guard at `ch_test_581`; after formatting-only generation-cache repair and the user Groq spend reset, `ch_test_581` plus the tail ran on Groq; merged full row moved to completed rows |
| 2026-05-16T20:26Z | CaseHOLD | `groq-llama70b` | `rag_rewrite` answer row | completed with explicit retry caveat; dynamic rewrite retrieval landed at Hit@5 0.4508 / MRR@5 0.3319 and answer accuracy 2542/3600 = 70.6%; moved to completed rows |
| 2026-05-16T21:38Z | BarExamQA | `or-gemma4-26b` | `llm_only` answer row | initial unconstrained OpenRouter prefix stopped on DekaLLM 401 under `NO_SILENT_FALLBACK`; clean 51-row prefix was explicitly merged with the 1144-row same-model Cloudflare-only tail, yielding 966/1195 = 80.8%; moved to completed rows |
| 2026-05-17T03:22Z | HousingQA | `groq-llama70b` | `llm_only` answer row | completed cleanly at 3067/6853 = 44.8%; no retrieval evidence by design, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/retries/near-cap outputs; moved to completed rows |
| 2026-05-17T04:33Z | BarExamQA | `or-gemma4-26b` | `golden_passage` answer row | completed with retry/near-cap caveat at 939/1195 = 78.6%; oracle gold retrieved on all rows, answer-flat vs `rag_simple` and directionally below `llm_only`; moved to completed rows |
| 2026-05-17T08:06Z | BarExamQA | `or-gemma4-26b` | `golden_plus_neighbors` answer row | completed with retry/near-cap caveat at 964/1195 = 80.7%; strict golden-neighbor cache replay retrieved gold on all rows and improved over `rag_simple` by +2.68pp; moved to completed rows |
| 2026-05-17T12:43Z | BarExamQA | `or-gemma4-26b` | `rag_hyde` answer row | completed with retry/near-cap caveat at 959/1195 = 80.3%; strict HyDE/retrieval cache replay, 1195/1195 cache hits, and 136/1195 gold retrieved; directionally positive vs `rag_simple`, flat vs `llm_only` and `golden_plus_neighbors`; moved to completed rows |
| 2026-05-17T17:44Z | BarExamQA | `or-gemma4-26b` | `snap_hyre` answer row | completed with retry/near-cap/transient caveat at 980/1195 = 82.0%; strict HyRE/retrieval cache replay, 1195/1195 cache hits, and 144/1195 gold retrieved; positive vs `rag_simple`, directionally above `rag_hyde` and `llm_only`; moved to completed rows |
| 2026-05-17T22:20Z | BarExamQA | `or-gemma4-26b` | `rag_rewrite` answer row | completed with retry/near-cap/transient caveat at 964/1195 = 80.7%; dynamic rewrite retrieval had 1195/1195 rewrite JSON parses, 146/1195 gold retrieved, Hit@5 0.1222 / MRR@5 0.0604; tied `golden_plus_neighbors`, positive vs `rag_simple`, below `snap_hyre`; moved to completed rows |
| 2026-05-18T00:50Z | BarExamQA | `or-ministral-8b` | `llm_only` answer row | completed with retry/near-cap caveat at 679/1195 = 56.8%; no retrieval evidence by design, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; 74 explicit answer-format retries and 9 near-cap accounting rows; moved to completed rows |
| 2026-05-18T03:51Z | BarExamQA | `or-ministral-8b` | `rag_simple` answer row | completed with retry/near-cap caveat at 680/1195 = 56.9%; strict raw retrieval-cache replay, 1195/1195 cache hits, 17/1195 gold retrieved, retrieval exposure Hit@5 0.0142 / MRR@5 0.0068; zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; 27 answer-format retries and 22 near-cap accounting rows; paired vs `llm_only`: +0.08pp, b/c=156/155, p=1.000; moved to completed rows |
| 2026-05-18T06:46Z | BarExamQA | `or-ministral-8b` | `golden_passage` answer row | completed with retry/near-cap caveat at 772/1195 = 64.6%; oracle gold retrieved on all rows and improved over `rag_simple` by +7.70pp (b/c=205/113, p=2.78e-07) and `llm_only` by +7.78pp (b/c=206/113, p=2.14e-07); moved to completed rows |
| 2026-05-18T08:25Z | HousingQA | `groq-llama70b` | `rag_simple` answer row | completed cleanly at 3241/6853 = 47.3%; strict raw retrieval-cache replay, 6853/6853 cache hits, 193/6853 gold retrieved, retrieval exposure Hit@5 0.0282 / MRR@5 0.0148; zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/retries/near-cap outputs; paired vs `llm_only`: +2.54pp, b/c=894/720, p=1.63e-05; moved to completed rows |
| 2026-05-18T09:59Z | BarExamQA | `or-ministral-8b` | `golden_plus_neighbors` answer row | completed with retry/near-cap caveat at 755/1195 = 63.2%; strict golden-neighbor retrieval-cache replay, 1195/1195 cache hits, 1195/1195 gold retrieved, retrieval exposure Hit@5/MRR@5 1.0000; zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows and zero runner near-cap failures; 30 answer-format retries and 28 near-1900 accounting rows; paired vs `rag_simple`: +6.28pp, b/c=205/130, p=4.93e-05; paired vs `llm_only`: +6.36pp, b/c=199/123, p=2.70e-05; paired vs `golden_passage`: -1.42pp, b/c=111/128, p=0.301; moved to completed rows |

## Blocked Rows

No current Groq blocker is active. Historical note: Groq `groq-llama70b` API
access hit a spend-alert threshold during the CaseHOLD `rag_hyde` answer row
(`spend_limit_reached` at `ch_test_2639`) and again blocked the initial
CaseHOLD `snap_hyre` answer replay. Both rows were recovered through explicit
same-model routing and/or Groq replay after the user reset the spend limit; do
not treat either recovery as a silent fallback.

The prior OpenRouter SCALR `llm_only`
blockers were superseded by retry-hardened full reruns and signed off below.
The initial 2026-05-15T08:29Z `or-ministral-8b` SCALR `rag_simple` row is
superseded and should not be cited: the stricter exact-final-line audit found
six rows where the stored prediction or final line was not source-safe. The
2026-05-15T14:34Z full rerun listed below is the current cite gate.
The initial 2026-05-15T14:56Z / 16:19Z / 16:34Z `or-ministral-8b` SCALR
`rag_rewrite` partial logs are superseded for the failed rows `scalr_110`,
`scalr_431`, and `scalr_538`; cite only the merged detail log listed below.
The 2026-05-16T00:13Z BarExamQA `or-gemma4-26b` `llm_only` partial log stopped
at 9/1195 rows and is not a blocked validity issue, but it is not citable. It
was an explicit pacing probe after OpenRouter rows took roughly 8-16 seconds
each under the no-silent-fallback launch contract.
The 2026-05-16T13:18Z BarExamQA `or-gemma4-26b` `llm_only` prefix is superseded
as a standalone row because DekaLLM returned 401 at `mbe_60` and
`NO_SILENT_FALLBACK` blocked the missing answer. The first 51 clean rows from
that prefix are included only in the merged Cloudflare-tail cite gate below.

## Completed Rows

| Dataset | Provider | Mode | Accuracy | Detail log | Health |
|---|---|---|---:|---|---|
| BarExamQA | `or-ministral-8b` | `llm_only` | 679/1195 = 56.8% | `logs/eval_llm_only_or-ministral-8b_20260517_172732_barexam_local-snap-hyre-or-ministral-8b-barexam-llm_only-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: no retrieval evidence by design, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; 74 logged answer-format retries, 9 rows at >=1900 total output tokens, max output 2070 tokens, max final-answer chars 9911, avg calls 1.06 |
| BarExamQA | `or-ministral-8b` | `rag_simple` | 680/1195 = 56.9% | `logs/eval_rag_simple_or-ministral-8b_20260517_195925_barexam_local-snap-hyre-or-ministral-8b-barexam-rag_simple-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: strict raw retrieval-cache replay, 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 17/1195 gold retrieved, retrieval exposure Hit@5 0.0142 / MRR@5 0.0068 from `docs/generated/retrieval_qrels_barexam_or-ministral-8b_rag_simple.md`; 27 logged answer-format retries, 22 rows at >=1900 total output tokens, max output 2091 tokens, max final-answer chars 9682; zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; paired vs `llm_only`: +0.08pp, b/c=156/155, p=1.000 |
| BarExamQA | `or-ministral-8b` | `golden_passage` | 772/1195 = 64.6% | `logs/eval_golden_passage_or-ministral-8b_20260517_225625_barexam_local-snap-hyre-or-ministral-8b-barexam-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: oracle gold injected/retrieved on 1195/1195 rows, retrieved list length 1 on all rows, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; 37 logged answer-format retries, 34 rows at >=1900 total output tokens, max output 2112 tokens, max final-answer chars 10173 at verbose `mbe_233` with intact final `Answer: (C)`; paired vs `rag_simple`: +7.70pp, b/c=205/113, p=2.78e-07; paired vs `llm_only`: +7.78pp, b/c=206/113, p=2.14e-07 |
| BarExamQA | `or-ministral-8b` | `golden_plus_neighbors` | 755/1195 = 63.2% | `logs/eval_golden_plus_neighbors_or-ministral-8b_20260518_015518_barexam_local-snap-hyre-or-ministral-8b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: strict golden-neighbor retrieval-cache replay, 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1195/1195 gold retrieved, retrieval exposure Hit@5/MRR@5 1.0000; 30 logged answer-format retries, 28 rows at >=1900 total output tokens, zero runner near-cap failures under the 2048-token margin, max output 2087 tokens, max final-answer chars 9848; zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; paired vs `rag_simple`: +6.28pp, b/c=205/130, p=4.93e-05; paired vs `llm_only`: +6.36pp, b/c=199/123, p=2.70e-05; paired vs `golden_passage`: -1.42pp, b/c=111/128, p=0.301 |
| LegalBench-SCALR | `groq-llama70b` | `llm_only` | 425/571 = 74.4% | `logs/eval_llm_only_groq-llama70b_20260514_1414_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` | clean: zero errors, missing predictions, parse failures, long rows, fallback markers, and near-cap outputs |
| LegalBench-SCALR | `groq-llama70b` | `rag_simple` | 416/571 = 72.9% | `logs/eval_rag_simple_groq-llama70b_20260515_171203_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | clean: 571/571 raw retrieval-cache hits, 0 empty retrieval rows, 283/571 gold retrieved, retrieval exposure Hit@5 0.4956 / MRR@5 0.3447, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/answer retries; max output 760 tokens; paired vs `llm_only`: -1.58pp, b/c=29/38, p=0.328 |
| LegalBench-SCALR | `groq-llama70b` | `golden_passage` | 534/571 = 93.5% | `logs/eval_golden_passage_groq-llama70b_20260515_173003_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry caveat: 571/571 gold injected, 571/571 evidence rows, 8 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 603 tokens and no near-cap repairs; paired vs `rag_simple`: +20.67pp, b/c=119/1, p=1.82e-34; paired vs `llm_only`: +19.09pp, b/c=110/1, p=8.63e-32 |
| LegalBench-SCALR | `groq-llama70b` | `golden_plus_neighbors` | 474/571 = 83.0% | `logs/eval_golden_plus_neighbors_groq-llama70b_20260515_174642_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry caveat: 571/571 golden-neighbor cache hits, 571/571 gold retrieved, 571/571 evidence rows, 2 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 750 tokens and no near-cap repairs; paired vs `rag_simple`: +10.16pp, b/c=73/15, p=2.56e-10; paired vs `golden_passage`: -10.51pp, b/c=1/61, p=2.73e-17; paired vs `llm_only`: +8.58pp, b/c=63/14, p=1.41e-08 |
| LegalBench-SCALR | `groq-llama70b` | `rag_hyde` | 402/571 = 70.4% | `logs/eval_rag_hyde_groq-llama70b_20260515_181837_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | clean: 571/571 HyDE cache hits and retrieval-cache hits, 0 empty retrieval rows, 351/571 gold retrieved, retrieval exposure Hit@5 0.6147 / MRR@10 0.5015, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/answer retries; max output 724 tokens; paired vs `rag_simple`: -2.45pp, b/c=23/37, p=0.0925; paired vs `llm_only`: -4.03pp, b/c=13/36, p=0.00140; paired vs `golden_plus_neighbors`: -12.61pp, b/c=8/80, p=4.60e-16 |
| LegalBench-SCALR | `groq-llama70b` | `snap_hyre` | 407/571 = 71.3% | `logs/eval_snap_hyre_groq-llama70b_20260515_183326_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | clean: 571/571 HyRE cache hits and retrieval-cache hits, 0 empty retrieval rows, 315/571 gold retrieved, retrieval exposure Hit@5 0.5517 / MRR@10 0.4126, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/answer retries; max output 789 tokens; paired vs `rag_simple`: -1.58pp, b/c=23/32, p=0.281; paired vs `llm_only`: -3.15pp, b/c=19/37, p=0.0222; paired vs `rag_hyde`: +0.88pp, b/c=24/19, p=0.542; paired vs `golden_plus_neighbors`: -11.73pp, b/c=10/77, p=5.92e-14 |
| LegalBench-SCALR | `groq-llama70b` | `rag_rewrite` | 409/571 = 71.6% | `logs/eval_rag_rewrite_groq-llama70b_20260515_184842_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_rewrite-nfull-k5_detail.jsonl` | signed with explicit retry caveat: dynamic rewrite retrieval, 571/571 rewrite JSON parses, 0 rewrite retries, 0 partial-JSON repairs, 0 empty retrieval rows, 329/571 gold retrieved, retrieval exposure Hit@5 0.5762 / MRR@5 0.4327 from `docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_rewrite.md`; 5 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 859 tokens and no near-cap repairs; paired vs `rag_simple`: -1.23pp, b/c=28/35, p=0.450; paired vs `snap_hyre`: +0.35pp, b/c=27/25, p=0.890; paired vs `rag_hyde`: +1.23pp, b/c=27/20, p=0.382; paired vs `llm_only`: -2.80pp, b/c=18/34, p=0.0365; paired vs `golden_plus_neighbors`: -11.38pp, b/c=13/78, p=1.85e-12 |
| BarExamQA | `groq-llama70b` | `llm_only` | 940/1195 = 78.7% | `logs/eval_llm_only_groq-llama70b_20260515_191548_barexam_local-snap-hyre-groq-llama70b-barexam-llm_only-nfull-k5_detail.jsonl` | clean: zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/answer retries; max output 738 tokens, max final-answer chars 3681, avg 1.00 calls; no long rows or near-cap outputs |
| BarExamQA | `groq-llama70b` | `rag_simple` | 891/1195 = 74.6% | `logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict raw retrieval-cache replay, 1195/1195 cache hits, 0 empty retrieval rows, 17/1195 gold retrieved, retrieval exposure Hit@5/Recall@5 0.0142 / MRR@5 0.0068 from `docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_simple.md`; 2 logged answer-format retries, retry max 5 tokens, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 1169 tokens and no near-cap repairs; paired vs `llm_only`: -4.10pp, b/c=66/115, p=0.000334 |
| BarExamQA | `groq-llama70b` | `golden_plus_neighbors` | 930/1195 = 77.8% | `logs/eval_golden_plus_neighbors_groq-llama70b_20260515_202827_barexam_local-snap-hyre-groq-llama70b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict golden-neighbor cache replay, 1195/1195 cache hits, 1195/1195 gold retrieved, retrieval exposure Hit@5/MRR@5 1.0000 from `docs/generated/retrieval_qrels_barexam_groq-llama70b_golden_plus_neighbors.md`; 2 logged answer-format retries, retry max 5 tokens, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 1112 tokens and no near-cap repairs; paired vs `rag_simple`: +3.26pp, b/c=136/97, p=0.0126; paired vs `llm_only`: -0.84pp, b/c=97/107, p=0.529 |
| BarExamQA | `groq-llama70b` | `golden_passage` | 946/1195 = 79.2% | `logs/eval_golden_passage_groq-llama70b_20260515_210741_barexam_local-snap-hyre-groq-llama70b-barexam-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry caveat: oracle gold injected/retrieved on 1195/1195 rows, 0 empty retrieval rows, 3 logged answer-format retries (`mbe_2`, `mbe_197`, `mbe_1125`), retry max 5 tokens, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 1029 tokens and no near-cap repairs; paired vs `rag_simple`: +4.60pp, b/c=137/82, p=0.000246; paired vs `llm_only`: +0.50pp, b/c=100/94, p=0.720; paired vs `golden_plus_neighbors`: +1.34pp, b/c=76/60, p=0.198 |
| BarExamQA | `groq-llama70b` | `rag_hyde` | 958/1195 = 80.2% | `logs/eval_rag_hyde_groq-llama70b_20260515_222654_barexam_local-snap-hyre-groq-llama70b-barexam-rag_hyde-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict generated/retrieval-cache replay, 1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, 0 empty retrieval rows, 125/1195 gold retrieved, retrieval exposure from signed cache Hit@5 0.1046 / MRR@10 0.0609; 3 logged answer-format retries (`mbe_90`, `mbe_95`, `mbe_131`), retry max 5 tokens, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 1061 tokens and no near-cap repairs; paired vs `rag_simple`: +5.61pp, b/c=137/70, p=3.73e-06; paired vs `llm_only`: +1.51pp, b/c=101/83, p=0.210; paired vs `golden_passage`: +1.00pp, b/c=106/94, p=0.437 |
| BarExamQA | `groq-llama70b` | `snap_hyre` | 953/1195 = 79.7% | `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict generated/retrieval-cache replay, 1195/1195 HyRE cache hits, 1195/1195 retrieval-cache hits, 0 empty retrieval rows, 132/1195 gold retrieved, retrieval exposure from signed cache Hit@5 0.1105 / MRR@10 0.0663; 1 logged answer-format retry (`mbe_2`), retry max 5 tokens, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/answer artifacts; max output 1265 tokens, max final-answer chars 6534, no long rows or near-cap repairs; paired vs `rag_simple`: +5.19pp, b/c=138/76, p=2.70e-05; paired vs `llm_only`: +1.09pp, b/c=103/90, p=0.388; paired vs `rag_hyde`: -0.42pp, b/c=79/84, p=0.754; paired vs `golden_passage`: +0.59pp, b/c=110/103, p=0.681; paired vs `golden_plus_neighbors`: +1.92pp, b/c=114/91, p=0.124 |
| BarExamQA | `groq-llama70b` | `rag_rewrite` | 923/1195 = 77.2% | `logs/eval_rag_rewrite_groq-llama70b_20260515_234357_barexam_local-snap-hyre-groq-llama70b-barexam-rag_rewrite-nfull-k5_detail.jsonl` | signed with explicit retry/repair caveat: dynamic rewrite retrieval, 1195/1195 rewrite JSON parses, 0 rewrite-format retries, 1 logged partial-JSON repair, 0 empty retrieval rows, 146/1195 gold retrieved, retrieval exposure Hit@5 0.1222 / MRR@5 0.0565 from `docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_rewrite.md`; 11 logged answer-format retries (`mbe_19`, `mbe_450`, `mbe_606`, `mbe_716`, `mbe_724`, `mbe_754`, `mbe_1081`, `mbe_1106`, `mbe_890`, `mbe_956`, `mbe_1141`), zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 1004 tokens and no near-cap repairs; paired vs `rag_simple`: +2.68pp, b/c=133/101, p=0.0425; paired vs `llm_only`: -1.42pp, b/c=102/119, p=0.282; paired vs `rag_hyde`: -2.93pp, b/c=84/119, p=0.0168; paired vs `snap_hyre`: -2.51pp, b/c=85/115, p=0.0400; paired vs `golden_passage`: -1.92pp, b/c=105/128, p=0.149 |
| BarExamQA | `or-gemma4-26b` | `llm_only` | 966/1195 = 80.8% | `logs/merged/eval_llm_only_or-gemma4-26b_cloudflare_tail_20260516_barexam_nfull_k5_detail.jsonl` | signed with explicit same-model route caveat: first 51 clean rows from the initial prefix plus 1144-row `OPENROUTER_PROVIDER_ONLY=Cloudflare` tail on `google/gemma-4-26b-a4b-it`; failed DekaLLM 401 row `mbe_60` excluded and superseded after `NO_SILENT_FALLBACK` blocked it. `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, think tags 0, 3 valid answer-format retries, 4 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2062 tokens, max final-answer chars 7959; one naive fallback-text hit on `mbe_608` is incidental legal explanation text |
| BarExamQA | `or-gemma4-26b` | `rag_simple` | 932/1195 = 78.0% | `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict raw retrieval-cache replay, 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty retrieval rows, 17/1195 gold retrieved, retrieval exposure Hit@5 0.0142 / MRR@5 0.0068 from `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_simple.md`; 3 logged answer-format retries (`mbe_576`, `mbe_989`, `mbe_1124`), zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags; max output 2052 tokens, max final-answer chars 7739, 3 rows at >=1900 output tokens with intact final `Answer:` lines; paired vs `llm_only`: -2.85pp, b/c=78/112, p=0.0164 |
| BarExamQA | `or-gemma4-26b` | `golden_passage` | 939/1195 = 78.6% | `logs/eval_golden_passage_or-gemma4-26b_20260516_200935_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: oracle gold injected/retrieved on 1195/1195 rows, retrieved list length 1 on all rows, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; 4 logged answer-format retries (`mbe_83`, `mbe_312`, `mbe_625`, `mbe_989`), 4 original outputs at >=1900 tokens with intact final `Answer:` lines, max output 2023 tokens, max final-answer chars 8849; paired vs `rag_simple`: +0.59pp, b/c=102/95, p=0.669; paired vs `llm_only`: -2.26pp, b/c=78/105, p=0.0543 |
| BarExamQA | `or-gemma4-26b` | `golden_plus_neighbors` | 964/1195 = 80.7% | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260516_233647_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: strict golden-neighbor retrieval-cache replay, retrieved list length 5 on all rows, 1195/1195 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; 5 logged answer-format retries (`mbe_532`, `mbe_562`, `mbe_989`, `mbe_1131`, `mbe_563`), 3 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2007 tokens, max final-answer chars 9033, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags; paired vs `rag_simple`: +2.68pp, b/c=116/84, p=0.0281; paired vs `llm_only`: -0.17pp, b/c=91/93, p=0.941; paired vs `golden_passage`: +2.09pp, b/c=91/66, p=0.0551 |
| BarExamQA | `or-gemma4-26b` | `rag_hyde` | 959/1195 = 80.3% | `logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap caveat: strict HyDE generation/retrieval-cache replay, 1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 136/1195 gold retrieved, retrieval exposure Hit@5 0.1138 / MRR@5 0.0542; 5 logged answer-format retries (`mbe_141`, `mbe_291`, `mbe_576`, `mbe_899`, `mbe_989`), 5 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2103 tokens, max final-answer chars 8665, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags; paired vs `rag_simple`: +2.26pp, b/c=113/86, p=0.0650; paired vs `llm_only`: -0.59pp, b/c=94/101, p=0.668; paired vs `golden_plus_neighbors`: -0.42pp, b/c=94/99, p=0.773; paired vs `golden_passage`: +1.67pp, b/c=113/93, p=0.185 |
| BarExamQA | `or-gemma4-26b` | `snap_hyre` | 980/1195 = 82.0% | `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap/transient caveat: strict HyRE generation/retrieval-cache replay, 1195/1195 HyRE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 144/1195 gold retrieved, retrieval exposure Hit@5 0.1205 / MRR@5 0.0602; 3 logged answer-format retries (`mbe_989`, `mbe_1131`, `mbe_288`), 4 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2025 tokens, max final-answer chars 8130, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags; visible Cloudflare 502/upstream idle retries recovered in-place under the pinned same provider/model; paired vs `rag_simple`: +4.02pp, b/c=121/73, p=0.000699; paired vs `llm_only`: +1.17pp, b/c=103/89, p=0.348; paired vs `rag_hyde`: +1.76pp, b/c=84/63, p=0.0987; paired vs `golden_plus_neighbors`: +1.34pp, b/c=101/85, p=0.271; paired vs `golden_passage`: +3.43pp, b/c=118/77, p=0.00406 |
| BarExamQA | `or-gemma4-26b` | `rag_rewrite` | 964/1195 = 80.7% | `logs/eval_rag_rewrite_or-gemma4-26b_20260517_124940_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_rewrite-nfull-k5_detail.jsonl` | signed with explicit retry/near-cap/transient caveat: dynamic rewrite retrieval, 1195/1195 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 raw-question fallbacks, 0 empty retrieval rows, 146/1195 gold retrieved, retrieval exposure Hit@5 0.1222 / MRR@5 0.0604 from `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_rewrite.md`; 4 logged answer-format retries (`mbe_501`, `mbe_763`, `mbe_989`, `mbe_486`), 3 rows at >=1900 total output tokens after retry, max output 2099 tokens, max final-answer chars 7618, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/long rows; visible Cloudflare upstream idle retries recovered in-place under the pinned same provider/model; paired vs `rag_simple`: +2.68pp, b/c=126/94, p=0.0364; paired vs `llm_only`: -0.17pp, b/c=102/104, p=0.944; paired vs `rag_hyde`: +0.42pp, b/c=88/83, p=0.760; paired vs `snap_hyre`: -1.34pp, b/c=75/91, p=0.244; paired vs `golden_plus_neighbors`: 0.00pp, b/c=98/98, p=1.000; paired vs `golden_passage`: +2.09pp, b/c=125/100, p=0.109 |
| CaseHOLD | `groq-llama70b` | `llm_only` | 2585/3600 = 71.8% | `logs/eval_llm_only_groq-llama70b_20260516_003945_casehold_local-snap-hyre-groq-llama70b-casehold-llm_only-nfull-k5_detail.jsonl` | signed with explicit retry caveat: no retrieval evidence by design, 3600/3600 Groq rows, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; 39 valid same-model answer-format retries, all `missing_marker`, retry max 5 tokens; max output 774 tokens, max final-answer chars 3845, avg 1.01 calls, no long rows or near-cap repairs |
| CaseHOLD | `groq-llama70b` | `rag_simple` | 2547/3600 = 70.8% | `logs/eval_rag_simple_groq-llama70b_20260516_020605_casehold_local-snap-hyre-groq-llama70b-casehold-rag_simple-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict raw retrieval-cache replay, 3600/3600 cache hits, 0 empty retrieval rows, 646/3600 gold retrieved, retrieval exposure Hit@5 0.1794 / MRR@5 0.1015 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_simple.md`; 23 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; max output 935 tokens, max final-answer chars 4850, no long rows or near-cap repairs; paired vs `llm_only`: -1.06pp, b/c=215/253, p=0.0871 |
| CaseHOLD | `groq-llama70b` | `golden_passage` | 3511/3600 = 97.5% | `logs/eval_golden_passage_groq-llama70b_20260516_032635_casehold_local-snap-hyre-groq-llama70b-casehold-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry caveat: oracle gold injected/retrieved on 3600/3600 rows, retrieved list length 1 on all rows, 0 empty retrieval rows, retrieval exposure Hit@1/Hit@5/MRR@5 1.0000 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_golden_passage.md`; 46 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/final-line prediction issues; max output 878 tokens, max final-answer chars 4845, no long rows or near-cap repairs; paired vs `rag_simple`: +26.78pp, b/c=968/4, p=1.86e-282; paired vs `llm_only`: +25.72pp, b/c=927/1, p=8.19e-277 |
| CaseHOLD | `groq-llama70b` | `golden_plus_neighbors` | 2859/3600 = 79.4% | `logs/eval_golden_plus_neighbors_groq-llama70b_20260516_045222_casehold_local-snap-hyre-groq-llama70b-casehold-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry caveat: strict golden-neighbor cache replay, 3600/3600 cache hits, retrieved list length 5 and neighbor list length 4 on every row, 3600/3600 gold retrieved, retrieval exposure Hit@1/Hit@5/MRR@5 1.0000 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_golden_plus_neighbors.md`; 19 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/final-line prediction issues; max output 943 tokens, max final-answer chars 4989, no long rows or near-cap repairs; trace response previews were clipped on 2794/3600 rows by `EVAL_TRACE_MAX_CHARS=800`, but full `final_answer` values were stored and exact; paired vs `rag_simple`: +8.67pp, b/c=459/147, p=2.70e-38; paired vs `llm_only`: +7.61pp, b/c=411/137, p=8.67e-33; paired vs `golden_passage`: -18.11pp, b/c=5/657, p=1.10e-187 |
| CaseHOLD | `groq-llama70b` | `rag_hyde` | 2532/3600 = 70.3% | `logs/merged/eval_rag_hyde_llama70b_mixed_20260516_casehold_nfull_k5_detail.jsonl` | signed with explicit mixed same-model provider caveat: Groq produced 2639 valid rows, then stopped on spend alert; `ch_test_2639` plus 960 tail rows were replayed through OpenRouter paid `meta-llama/llama-3.3-70b-instruct`, with the final 667-row segment pinned to `OPENROUTER_PROVIDER_ONLY=AkashML`; strict generated/retrieval-cache replay, 3600/3600 HyDE cache hits and retrieval-cache hits, 0 empty retrieval rows, 1844/3600 gold retrieved, retrieval exposure Hit@5 0.5122 / MRR@5 0.3983 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_hyde_mixed.md`; 21 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/near-cap outputs; max output 742 tokens; paired vs `rag_simple`: -0.42pp, b/c=247/262, p=0.535; paired vs `llm_only`: -1.47pp, b/c=211/264, p=0.0169; paired vs `golden_plus_neighbors`: -9.08pp, b/c=119/446, p=1.91e-45 |
| CaseHOLD | `groq-llama70b` | `snap_hyre` | 2538/3600 = 70.5% | `logs/merged/eval_snap_hyre_llama70b_mixed_20260516_casehold_nfull_k5_detail.jsonl` | signed with explicit retry/repair/mixed-provider caveat: OpenRouter paid same-model prefix supplied 581 rows while Groq spend alert was active; after user reset, repaired `ch_test_581` plus tail ran on Groq, for provider mix `or-llama70b-paid` 581 / `groq-llama70b` 3019. The invalid OpenRouter `ch_test_581` answer row was excluded; the generation cache row was formatting-only repaired to standalone `Answer: (A)`. Strict replay: 3600/3600 HyRE and retrieval-cache hits, 0 empty retrieval rows, 1619/3600 gold retrieved, retrieval exposure Hit@5 0.4497 / MRR@5 0.3286 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_snap_hyre_mixed.md`; 16 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/near-cap outputs; max output 826 tokens; paired vs `rag_simple`: -0.25pp, b/c=249/258, p=0.722; paired vs `llm_only`: -1.31pp, b/c=200/247, p=0.0295; paired vs `rag_hyde`: +0.17pp, b/c=225/219, p=0.812; paired vs `golden_plus_neighbors`: -8.92pp, b/c=118/439, p=1.98e-44 |
| CaseHOLD | `groq-llama70b` | `rag_rewrite` | 2542/3600 = 70.6% | `logs/eval_rag_rewrite_groq-llama70b_20260516_130926_casehold_local-snap-hyre-groq-llama70b-casehold-rag_rewrite-nfull-k5_detail.jsonl` | signed with explicit retry caveat: dynamic rewrite retrieval, 3600/3600 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 empty retrieval rows, 1623/3600 gold retrieved, retrieval exposure Hit@5 0.4508 / MRR@5 0.3319 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_rewrite.md`; 88 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/near-cap outputs; max output 899 tokens; paired vs `rag_simple`: -0.14pp, b/c=251/256, p=0.859; paired vs `snap_hyre`: +0.11pp, b/c=237/233, p=0.890; paired vs `rag_hyde`: +0.28pp, b/c=236/226, p=0.675; paired vs `llm_only`: -1.19pp, b/c=226/269, p=0.0589; paired vs `golden_plus_neighbors`: -8.81pp, b/c=134/451, p=4.62e-41 |
| HousingQA | `groq-llama70b` | `llm_only` | 3067/6853 = 44.8% | `logs/eval_llm_only_groq-llama70b_20260516_203552_housing_local-snap-hyre-groq-llama70b-housing-llm_only-nfull-k5_detail.jsonl` | clean: no retrieval evidence by design, provider/mode/dataset exactly `groq-llama70b`/`llm_only`/`housing`, retrieved list length 0 on all rows, exact final `Answer: Yes/No` lines on all rows, zero errors/missing predictions/parse failures/fallback keys/think tags/answer retries/long rows/near-cap outputs; max output 518 tokens, max final-answer chars 2730, avg 1.00 calls |
| HousingQA | `groq-llama70b` | `rag_simple` | 3241/6853 = 47.3% | `logs/eval_rag_simple_groq-llama70b_20260518_001738_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl` | clean: strict raw retrieval-cache replay, provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`housing`, 6853/6853 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 193/6853 gold retrieved, retrieval exposure Hit@5 0.0282 / MRR@5 0.0148; exact final `Answer: Yes/No` lines on all rows, zero errors/missing predictions/parse failures/fallback keys/think tags/answer retries/long rows/near-cap outputs; max output 866 tokens, max final-answer chars 4878; paired vs `llm_only`: +2.54pp, b/c=894/720, p=1.63e-05 |
| LegalBench-SCALR | `or-gemma4-26b` | `llm_only` | 417/571 = 73.0% | `logs/eval_llm_only_or-gemma4-26b_20260515_0056_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-llm_only-nfull-k5_merged_detail.jsonl` | clean: merged from rows 0-420, repaired row 421, and rows 422-570; zero errors, missing predictions, missing markers, fallback markers, long rows, and near-cap violations |
| LegalBench-SCALR | `or-ministral-8b` | `llm_only` | 384/571 = 67.3% | `logs/eval_llm_only_or-ministral-8b_20260514_195855_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` | clean: zero errors, missing predictions, missing markers, fallback markers, long rows, retry rows, and near-cap violations |
| LegalBench-SCALR | `or-ministral-8b` | `rag_simple` | 388/571 = 68.0% | `logs/eval_rag_simple_or-ministral-8b_20260515_093406_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 raw retrieval-cache hits, 0 empty retrieval rows, 283/571 gold retrieved, retrieval exposure Hit@5 0.4956 / MRR@5 0.3447, 16 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers/exact-final-line issues; 11 original answers reached >=2000 output tokens, max 2106 before 5-token final-line repair; paired vs `llm_only`: +0.70pp, b/c=47/43, p=0.752 |
| LegalBench-SCALR | `or-ministral-8b` | `golden_passage` | 532/571 = 93.2% | `logs/eval_golden_passage_or-ministral-8b_20260515_102620_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 gold injected, 571/571 evidence rows, 22 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers/exact-final-line issues; four original answers reached >=2000 output tokens, max 2050 before 5-token final-line repair; paired vs `rag_simple`: +25.22pp, b/c=145/1, p=3.30e-42; paired vs `llm_only`: +25.92pp, b/c=151/3, p=5.33e-41 |
| LegalBench-SCALR | `or-ministral-8b` | `golden_plus_neighbors` | 440/571 = 77.1% | `logs/eval_golden_plus_neighbors_or-ministral-8b_20260515_112849_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 golden-neighbor cache hits, 571/571 gold retrieved, 571/571 evidence rows, 5 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers/exact-final-line issues; three original answers reached >=2000 output tokens, max 2056 before 5-token final-line repair; paired vs `rag_simple`: +9.11pp, b/c=72/20, p=4.61e-08; paired vs `golden_passage`: -16.11pp, b/c=4/96, p=6.45e-24; paired vs `llm_only`: +9.81pp, b/c=78/22, p=1.59e-08 |
| LegalBench-SCALR | `or-ministral-8b` | `rag_hyde` | 406/571 = 71.1% | `logs/eval_rag_hyde_or-ministral-8b_20260515_130224_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 HyDE cache hits and retrieval-cache hits, 0 empty retrieval rows, 344/571 gold retrieved, 18 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers/exact-final-line issues; 14 original answers reached >=2000 output tokens, max 2104 before 5-token final-line repair; paired vs `rag_simple`: +3.15pp, b/c=43/25, p=0.0385; paired vs `llm_only`: +3.85pp, b/c=54/32, p=0.0230 |
| LegalBench-SCALR | `or-ministral-8b` | `snap_hyre` | 399/571 = 69.9% | `logs/eval_snap_hyre_or-ministral-8b_20260515_140203_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 HyRE cache hits and retrieval-cache hits, 0 empty retrieval rows, 354/571 gold retrieved, 9 logged answer-format retries, zero errors/missing predictions/parse failures/exact-final-line issues; targeted fallback-key audit found zero fallback keys/provider substitutions, while one incidental legal-text "fallback" appeared in generated CERCLA text; 7 original answers reached >=2000 output tokens, max 2082 before 5-token final-line repair; paired vs `rag_simple`: +1.93pp, b/c=45/34, p=0.260; paired vs `rag_hyde`: -1.23pp, b/c=29/36, p=0.457; paired vs `llm_only`: +2.63pp, b/c=46/31, p=0.110; paired vs `golden_plus_neighbors`: -7.18pp, b/c=20/61, p=5.66e-06 |
| LegalBench-SCALR | `or-ministral-8b` | `rag_rewrite` | 399/571 = 69.9% | `logs/eval_rag_rewrite_or-ministral-8b_20260515_145614_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_rewrite-nfull-k5_merged_detail.jsonl` | signed with explicit retry/repair caveat: dynamic rewrite retrieval, 0 empty retrieval rows, 371/571 gold retrieved, retrieval exposure Hit@5 0.6497 / MRR@5 0.5185 from `docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_rewrite.md`; 5 logged rewrite-format retries, 1 logged partial-JSON repair on `scalr_538`, 31 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues; paired vs `rag_simple`: +1.93pp, b/c=40/29, p=0.228; paired vs `snap_hyre`: tied, b/c=33/33, p=1.000; paired vs `rag_hyde`: -1.23pp, b/c=31/38, p=0.470; paired vs `llm_only`: +2.63pp, b/c=48/33, p=0.119; paired vs `golden_plus_neighbors`: -7.18pp, b/c=20/61, p=5.66e-06 |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_simple` | 419/571 = 73.4% | `logs/eval_rag_simple_or-gemma4-26b_20260514_204706_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | clean: 571/571 raw retrieval-cache hits, 0 empty evidence rows, 283/571 gold retrieved, 12 logged format-only retries, zero missing markers/fallbacks/near-cap violations |
| LegalBench-SCALR | `or-gemma4-26b` | `golden_plus_neighbors` | 464/571 = 81.3% | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260514_221537_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | clean: 571/571 golden-neighbor cache hits, 571/571 gold retrieved, 7 logged retries, zero missing markers/fallbacks/near-cap violations; paired vs `rag_simple`: 58 fixes, 13 regressions, exact McNemar p=6.27e-08 |
| LegalBench-SCALR | `or-gemma4-26b` | `golden_passage` | 559/571 = 97.9% | `logs/eval_golden_passage_or-gemma4-26b_20260514_235919_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | clean: 571/571 gold injected, 1 logged format-only retry, zero missing markers/fallbacks/near-cap violations; paired vs `golden_plus_neighbors`: gold-only 559 vs 464, plus-neighbor fixes 1 and regressions 96, exact McNemar p=1.24e-27 |
| LegalBench-SCALR | `or-gemma4-26b` | `snap_hyre` | 422/571 = 73.9% | `logs/eval_snap_hyre_or-gemma4-26b_20260515_020525_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 HyRE cache hits and retrieval-cache hits, 0 empty retrieval rows, 415/571 gold retrieved, 10 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers; five original answers reached >=2000 output tokens, max 2046/2048 before format repair; paired vs `rag_simple`: +0.53pp, b/c=27/24, p=0.780; paired vs `llm_only`: +0.88pp, b/c=26/21, p=0.560 |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_hyde` | 412/571 = 72.2% | `logs/eval_rag_hyde_or-gemma4-26b_20260515_042731_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | signed with explicit retry/truncation caveat: 571/571 HyDE cache hits and retrieval-cache hits, 0 empty retrieval rows, 404/571 gold retrieved, 8 logged answer-format retries, zero errors/missing predictions/parse failures/fallback markers; three original answers reached >=2000 output tokens, max 2047/2048 before format repair; paired vs `rag_simple`: -1.23pp, b/c=30/37, p=0.464; paired vs `snap_hyre`: -1.75pp, b/c=22/32, p=0.220 |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_rewrite` | 422/571 = 73.9% | `logs/eval_rag_rewrite_or-gemma4-26b_20260515_062250_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_rewrite-nfull-k5_detail.jsonl` | signed with explicit retry caveat: dynamic rewrite retrieval, 571/571 rewrite JSON parses, 0 rewrite retries/fallbacks, 0 empty retrieval rows, 385/571 gold retrieved, 9 logged valid answer-format retries, zero errors/missing predictions/parse failures/fallback markers; retrieval exposure Hit@5 0.6743, MRR@5 0.5212; paired vs `rag_simple`: +0.53pp, b/c=34/31, p=0.804; paired vs `snap_hyre`: tied, b/c=30/30, p=1.000; paired vs `rag_hyde`: +1.75pp, b/c=40/30, p=0.282 |

Rows are complete only after the detail log exists, row count matches the full
dataset, `scripts/analyze_detail_flags.py` passes, strict cache-hit checks pass
where applicable, and the result is added to `docs/signoff_log.md`.

## Current Notes

- The harness now streams detail rows to the final detail log path as rows
  complete, and records the violating row before a no-silent-fallback abort.
  This prevents late failures from losing row-level evidence.
- The OpenRouter final-answer retry now has a format-only path when the prior
  response already produced a parseable prediction. This preserves the same
  model and same evidence/question while preventing retry calls from reopening
  long reasoning loops. All three SCALR `llm_only` providers are now signed off
  clean under the fixed k=5/2048-token/no-silent-fallback contract.
- The Snap-HyRE generation-cache path now has an explicit same-model
  format-retry path for malformed generated answer/passage blocks. The SCALR
  `or-gemma4-26b` full `snap_hyre` cache used this once and records the retry
  reason in the cache; this is not a silent fallback or method substitution.
- The `rag_rewrite` path now validates query-rewriter JSON, logs
  `rewrite_parse_ok` and retry metadata, and does one same-model
  generation-format retry when `EVAL_GENERATION_FORMAT_RETRY=1`. Under
  `NO_SILENT_FALLBACK=1`, an unrepaired malformed rewrite raises before
  retrieval instead of silently falling back to the raw question.
- For discrete answer tasks, exact final answer lines are now the scoring source
  of truth. If the last non-empty line is exactly `Answer: (X)` or
  `Answer: Yes/No`, that value overrides earlier answer mentions; otherwise the
  same-model final-answer retry must produce the exact line before the row can
  pass the no-silent-fallback guard.
- The SCALR `groq-llama70b` `rag_simple` row is clean but downstream-negative
  versus `llm_only`: 72.9% vs 74.4% (-1.58pp, McNemar p=0.328). It used
  571/571 raw retrieval-cache hits, retrieved gold on 283/571 rows, and has raw
  retrieval exposure Hit@5 0.4956 / MRR@5 0.3447. There were no answer retries,
  exact-final-line issues, fallback keys, empty retrieval rows, or near-cap
  outputs.
- The SCALR `groq-llama70b` `golden_passage` oracle control is strongly
  positive downstream: 93.5%, +20.67pp over `rag_simple` and +19.09pp over
  `llm_only`, both with exact McNemar p < 1e-31. Cite with the retry caveat
  because 8 rows needed same-model final-line repair, but none approached the
  2048-token cap.
- The SCALR `groq-llama70b` `golden_plus_neighbors` row is positive versus
  strict raw RAG but much weaker than gold-only: 83.0%, +10.16pp over
  `rag_simple` (McNemar p=2.56e-10), and -10.51pp versus `golden_passage`
  (p=2.73e-17). Cite with the retry caveat because 2 rows needed same-model
  final-line repair. This confirms the gold-neighbor dilution pattern across
  all three SCALR providers.
- The SCALR `groq-llama70b` generated-query caches are now signed off for
  answer replay. Unlike the Gemma 26B and small-model rows, Groq `snap_hyre`
  is retrieval-positive over raw question retrieval but below Groq `rag_hyde`:
  Hit@5 0.5517 vs raw 0.4956 and HyDE 0.6147; MRR@10 0.4126 vs HyDE 0.5015.
- The SCALR `groq-llama70b` `rag_hyde` answer row is retrieval-positive but
  downstream-negative: retrieval Hit@5 improves over raw question retrieval
  from 0.4956 to 0.6147, but answer accuracy falls to 70.4% vs `rag_simple`
  72.9% (McNemar p=0.0925) and `llm_only` 74.4% (p=0.00140). The row is clean
  with no answer retries, fallback keys, exact-final-line issues, or near-cap
  outputs.
- The SCALR `groq-llama70b` `snap_hyre` answer row is also retrieval-positive
  over raw but downstream-negative: retrieval Hit@5 improves over raw question
  retrieval from 0.4956 to 0.5517, but answer accuracy is 71.3% vs
  `rag_simple` 72.9% (McNemar p=0.281) and `llm_only` 74.4% (p=0.0222). It is
  slightly above Groq `rag_hyde` downstream, 71.3% vs 70.4% (p=0.542), with a
  clean replay and no answer retries, fallback keys, exact-final-line issues,
  or near-cap outputs.
- The SCALR `groq-llama70b` `rag_rewrite` row is retrieval-positive over raw
  but still downstream-negative: dynamic rewrite retrieval Hit@5 is 0.5762 vs
  raw 0.4956, but answer accuracy is 71.6% vs `rag_simple` 72.9% (McNemar
  p=0.450) and `llm_only` 74.4% (p=0.0365). It is essentially tied with
  `snap_hyre` downstream, 71.6% vs 71.3% (p=0.890), and above `rag_hyde`,
  71.6% vs 70.4% (p=0.382). Cite with the retry caveat because 5 rows needed
  same-model final-line repair, with no near-cap repairs or fallback keys.
- The BarExamQA `groq-llama70b` `rag_hyde` row is the current best signed
  BarExamQA Llama 70B row at 80.2%. It is significantly above strict raw RAG
  (+5.61pp, McNemar p=3.73e-06), directionally above `llm_only` (+1.51pp,
  p=0.210), and directionally above `golden_passage` (+1.00pp, p=0.437). The
  `snap_hyre` row is also positive over strict raw RAG (+5.19pp, p=2.70e-05)
  and flat versus `llm_only` (+1.09pp, p=0.388), but is slightly below
  `rag_hyde` downstream despite stronger retrieval exposure (Hit@5 0.1105 vs
  0.1046). The `golden_passage` oracle row remains useful because it is also
  above strict raw RAG (+4.60pp, p=0.000246), but golden-plus-neighbors is
  weaker than gold-only by -1.34pp (p=0.198), repeating the caution that extra
  retrieved neighbors can dilute a gold oracle prompt even when all rows include
  gold.
- The BarExamQA `groq-llama70b` generated-query caches are now signed off for
  answer replay. Retrieval exposure is positive but still sparse in absolute
  terms: raw question retrieval Hit@5/MRR@5 was 0.0142/0.0068, `rag_hyde`
  reaches Hit@5 0.1046 / MRR@5 0.0515, and `snap_hyre` reaches Hit@5 0.1105 /
  MRR@5 0.0564. `snap_hyre` is slightly above `rag_hyde` at every audited k
  on BarExamQA, but the absolute gold-hit rate remains much lower than SCALR.
- The SCALR `or-ministral-8b` `rag_simple` answer row is answer-flat versus
  `llm_only` under strict raw retrieval-cache replay: 68.0% vs 67.3%
  (McNemar p=0.752). Cite with the retry caveat because 16 rows needed
  same-model exact-line repair and 11 original responses reached >=2000 output
  tokens before repair. The initial 08:29Z row is superseded by the 09:34:06
  detail log because stricter auditing found six source-safety failures.
- The SCALR `or-ministral-8b` `golden_passage` oracle control is strongly
  positive downstream: 93.2%, +25.22pp over strict `rag_simple` and +25.92pp
  over `llm_only`, both with exact McNemar p < 1e-40. This is a clean
  evidence-use ceiling signal for the small model, but cite with a retry caveat
  because 22 rows needed same-model final-line repair.
- The SCALR `or-ministral-8b` `golden_plus_neighbors` row is positive over
  strict `rag_simple` but much weaker than gold-only: 77.1%, +9.11pp over
  `rag_simple` (McNemar p=4.61e-08) and -16.11pp versus `golden_passage`
  (McNemar p=6.45e-24). Cite with the retry caveat because 5 rows needed
  same-model final-line repair and 3 original responses reached >=2000 output
  tokens before repair. This repeats the Gemma 26B pattern that adding retrieved
  neighbors to gold can dilute the oracle evidence-use ceiling on SCALR.
- The SCALR `or-ministral-8b` `rag_hyde` answer row is downstream-positive
  over strict raw RAG: 71.1%, +3.15pp over `rag_simple` (McNemar p=0.0385)
  and +3.85pp over `llm_only` (McNemar p=0.0230). Retrieval improves over raw
  question retrieval from Hit@5 0.4956 to 0.6025, but remains below
  `snap_hyre` retrieval on the small model. Cite with the retry caveat because
  18 rows needed same-model final-line repair and 14 original responses reached
  >=2000 output tokens before repair.
- The SCALR `or-ministral-8b` `snap_hyre` answer row is retrieval-positive but
  downstream weaker than `rag_hyde`: 69.9%, +1.93pp over `rag_simple`
  (McNemar p=0.260), +2.63pp over `llm_only` (p=0.110), and -1.23pp versus
  `rag_hyde` (p=0.457). It retrieves gold on 354/571 rows, above `rag_hyde`'s
  344/571 and raw RAG's 283/571. Cite with the retry caveat because 9 rows
  needed same-model final-line repair and 7 original responses reached >=2000
  output tokens before repair. A naive string scan finds one incidental legal
  use of the word "fallback" in the generated CERCLA query, but targeted
  fallback-key audit finds zero fallback keys or provider/method substitutions.
- The SCALR `or-ministral-8b` `rag_rewrite` answer row ties `snap_hyre` at
  69.9% and is retrieval-positive over raw question retrieval: Hit@5 0.6497
  vs raw 0.4956, with 371/571 gold retrieved. It is answer-flat versus
  `rag_simple` (+1.93pp, McNemar p=0.228) and below `rag_hyde` (-1.23pp,
  p=0.470). Cite with the explicit repair caveat: the first attempts exposed
  repeated no-silent rewrite JSON failures on long malformed outputs, so the
  harness now logs partial-JSON recovery when it can extract model-generated
  `primary` / `alternatives` strings without falling back to the raw question.
  The merged row used this once on `scalr_538`.
- The SCALR `or-gemma4-26b` `snap_hyre` answer row is retrieval-positive but
  answer-flat: it improves retrieval Hit@5 over raw question retrieval by
  +23.12pp, while downstream accuracy only moves from `rag_simple` 73.4% to
  73.9% (McNemar p=0.780). Cite with the explicit retry caveat because several
  original final-answer calls approached the 2048-token cap before the logged
  same-model final-line repair.
- The SCALR `or-gemma4-26b` `rag_hyde` answer row is also retrieval-positive
  but downstream negative/flat: retrieval Hit@5 improves over raw question
  retrieval by +21.19pp, while answer accuracy lands below `rag_simple` at
  72.2% vs 73.4% (McNemar p=0.464). Compared with `snap_hyre`, `rag_hyde` has
  higher retrieval MRR@10 but lower Hit@5 and lower downstream accuracy.
- The SCALR `or-gemma4-26b` `rag_rewrite` answer row ties `snap_hyre` at
  73.9% and is answer-flat versus `rag_simple` (73.4%, McNemar p=0.804).
  Rewrite retrieval is still positive over raw question retrieval
  (Hit@5 0.6743 vs raw 0.4956), but below `snap_hyre` and `rag_hyde` on
  Hit@5. The row has no rewrite parse failures or rewrite fallbacks; cite with
  the explicit final-answer retry caveat because 9 long-ish responses needed
  same-prediction final-line repair.
- HousingQA golden-neighbor cache construction is now unblocked, but answer
  rows using this cache should cite the explicit stored-gold-embedding backend
  rather than claiming it used arbitrary text re-embedding of the gold passage.
