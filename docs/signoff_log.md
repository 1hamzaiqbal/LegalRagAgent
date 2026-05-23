# Sign-off Log — verified results approved for paper/meeting citation

## Update 2026-05-12 ~Snap-HyRE comprehensive branch

Change reason: branch `snap_hyre_comprehensive` pivots the active narrative
from diagnostic adaptation to a fixed Snap-HyRE method evaluation. The May 11
diagnostic/controller docs and scripts were archived for traversibility, but
their validated rows remain source-gated here.

Last updated: 2026-05-21
Branch: `snap_hyre_comprehensive`

Current active planning docs:

- `docs/snap_hyre_comprehensive_plan_2026-05-12.md`
- `docs/literature_snap_hyre_2026-05-12.md`
- `docs/README.md`

HousingQA note, 2026-05-20: Housing retrieval rows in the active main matrix
must use the jurisdiction state filter. Older unfiltered Housing retrieval rows
remain provenance/ablation evidence only; do not cite them as main-matrix
Housing retrieval results unless the row/detail log records
`housing_state_filter=true` or `retrieval_where={"state": ...}`. `llm_only` and
`golden_passage` are unaffected because they do not perform corpus retrieval.

Archived May 11 package source docs now live under:

- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_prep_2026-05-11_diagnostic_adaptation.md`
- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_package_audit_2026-05-11.md`
- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_eval_expansion_status_2026-05-11.md`

Do not treat the archived diagnostic-controller framing as the active paper
story. Treat the rows below as historical, source-gated evidence that may be
reused in the new fixed-method Snap-HyRE tables only when the method, dataset,
model, `k`, detail log, and caveat still match.

## Update 2026-05-14 ~comprehensive launch/cache gate

Scope: fixed-method comprehensive launch prep at `RETRIEVAL_K=5` with
full-corpus deterministic retrieval caches and strict no-silent-fallback
guards.

Retrieval cache signoff:

| Dataset | Cache evidence | Health | Sign-off |
|---|---|---|---|
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_qfull_seed42_raw_question_k10.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_golden_neighbors_k10.jsonl` | 571 rows each; raw Hit@5 0.4956, Hit@10 0.5937; golden-neighbor injected-gold Hit/MRR 1.0000 | ✅ RETRIEVAL-CACHE-CLEAN |
| CaseHOLD | `caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl`; `caches/retrieval/full/casehold_qfull_seed42_golden_neighbors_k10.jsonl` | 3600 rows each; raw Hit@5 0.1794, Hit@10 0.2639; golden-neighbor injected-gold Hit/MRR 1.0000 | ✅ RETRIEVAL-CACHE-CLEAN |
| HousingQA | `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl`; `caches/retrieval/full/housing_qfull_seed42_golden_neighbors_k10.jsonl` | 6853 rows each; golden-neighbor final audit has duplicate_keys=0, missing_idx=0, empty_retrieval=0, rows_shorter_than_min_k=0, rows_without_gold=0, Hit@5 1.0000, MRR@10 1.0000 | ✅ RETRIEVAL-CACHE-CLEAN |
| HousingQA `groq-llama70b` `rag_hyde` | `caches/hyre/full/housing_qfull_seed42_groq-llama70b_rag_hyde.jsonl`; `caches/retrieval/full/housing_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | generation rows 6853/6853, unique labels 6853, zero errors, missing HyDE passages, truthy fallbacks, parse failures, answer-artifact passages, or think tags; provider route `{}` on all rows; passage chars min/p50/p95/max 304/461/557/737; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.1665, Recall@5 0.1124, MRR@5 0.1098, Hit@10 0.2196, MRR@10 0.1169; Housing full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=4096` for retrieval-cache construction | ✅ GENERATED-CACHE-CLEAN |
| HousingQA `groq-llama8b` `rag_hyde` | `caches/hyre/full/housing_qfull_seed42_groq-llama8b_rag_hyde.jsonl`; `caches/retrieval/full/housing_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl`; `caches/retrieval_doc/full/housing_qfull_seed42_groq-llama8b_rag_hyde_k10_doc_cache.jsonl` | generation rows 6853/6853, unique labels 6853, zero errors, missing HyDE passages, truthy fallbacks, parse failures, answer-artifact passages, think tags, retries, or near-cap generation rows; provider route `{}` on all rows; passage chars min/p50/p95/max 250/508/621/824; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.1246, Recall@5 0.0828, MRR@5 0.0775, Hit@10 0.1682, MRR@10 0.0833; Housing full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=4096` for retrieval-cache construction; hydrated document cache wrote 9068/9068 needed docs with missing=0 for strict replay without reopening the full Housing collection | ✅ GENERATED-CACHE-CLEAN |
| Legal-Link-EU | `caches/retrieval/full/legal_link_eu_qfull_seed42_raw_question_ce22000_k10.jsonl` | 1127 rows; duplicate_keys=0, missing_idx=0, empty_retrieval=0, rows_shorter_than_min_k=0, rows_without_gold=0; raw Hit@5 0.9059, Hit@10 0.9556, MRR@5 0.7621, MRR@10 0.7689; `CROSS_ENCODER_MAX_CHARS=22000` with CE doc truncation 0 and query truncation rows 0; source_doc and target_doc retrieval fields present | ✅ RETRIEVAL-CACHE-CLEAN |
| Legal-Link-EU `groq-llama8b` `rag_hyde` | `caches/generation/full/legal_link_eu_qfull_seed42_groq-llama8b_rag_hyde.jsonl`; `caches/retrieval/full/legal_link_eu_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl` | generation rows 1127/1127, unique labels 1127, zero errors, missing passages, fallbacks, parse failures, answer-artifact passages, or think tags; one logged same-model HyDE format retry resolved the single near-cap generation row; passage chars min/p50/p95/max 360/566/686/771; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.4756, Hit@10 0.5723, MRR@10 0.3819; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0 and query truncation 0 | ⚠️ GENERATED-CACHE-CLEAN/RETRY-CAVEAT |
| Legal-Link-EU `groq-llama8b` `snap_hyre` | `caches/generation/full/legal_link_eu_qfull_seed42_groq-llama8b_snap_hyre.jsonl`; `caches/retrieval/full/legal_link_eu_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl` | generation rows 1127/1127, unique labels 1127, zero errors, missing passages, fallbacks, parse failures, answer-artifact passages, or think tags; 32 logged same-model Snap-HyRE format retries, including one final-line repair and one overlong-passage repair; near-cap rows 7, all resolved by valid retry/cache text; max HyRE passage chars 702; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.3753, Hit@10 0.4552, MRR@5 0.2787, MRR@10 0.2890; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0 and query truncation 0, source_doc retrieved on 400/1127 rows and target_doc on 326/1127 | ⚠️ GENERATED-CACHE-CLEAN/RETRY-CAVEAT |
| Legal-Link-EU `or-gemma4-26b` `rag_hyde` | `caches/hyre/full/legal_link_eu_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`; `caches/retrieval/full/legal_link_eu_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | generation rows 1127/1127, unique labels 1127, all `OPENROUTER_PROVIDER_ONLY=Cloudflare`, zero errors, missing passages, fallbacks, parse failures, answer-artifact passages, think tags, retries, or near-cap generation rows; passage chars min/p50/p95/max 413/625/726/829; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.4898, Recall@5 0.1885, MRR@5 0.3637, Hit@10 0.5892, MRR@10 0.3771; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0 and query truncation 0, source_doc retrieved on 559/1127 rows and target_doc on 392/1127 | ✅ GENERATED-CACHE-CLEAN |
| Legal-Link-EU `or-gemma4-26b` `snap_hyre` | `caches/hyre/full/legal_link_eu_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`; `caches/retrieval/full/legal_link_eu_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` | generation rows 1127/1127, unique labels 1127, all `OPENROUTER_PROVIDER_ONLY=Cloudflare`, zero errors, missing passages, fallbacks, parse failures, missing snap letters, answer-artifact passages, think tags, retries, or near-cap generation rows; passage chars min/p50/p95/max 192/301/371/463; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, rows_without_gold=0, Hit@5 0.6788, Recall@5 0.2896, MRR@5 0.5473, Hit@10 0.7684, MRR@10 0.5588; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0 and query truncation 0, source_doc retrieved on 675/1127 rows and target_doc on 628/1127 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `or-gemma4-26b` `snap_hyre` | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` | generation rows 571/571, zero errors/missing passages/parse failures/answer-artifact passages, one logged same-model generation-format retry on `scalr_273`; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.7268, Hit@10 0.7828, MRR@10 0.6242 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `or-gemma4-26b` `rag_hyde` | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | generation rows 571/571, zero errors/missing passages/parse failures/answer-artifact passages/retries, compact passage lengths max 551 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.7075, Hit@10 0.7688, MRR@10 0.6383 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `or-ministral-8b` `rag_hyde` | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-ministral-8b_rag_hyde.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-ministral-8b_rag_hyde_k10.jsonl` | generation rows 571/571, zero errors/missing passages/fallbacks/parse failures/answer-artifact passages/retries, passage chars min/p50/p95/max 439/667/872/1043; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.6025, Hit@10 0.6865, MRR@10 0.4506 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `or-ministral-8b` `snap_hyre` | `caches/hyre/full/legalbench_scalr_qfull_seed42_or-ministral-8b_snap_hyre.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_or-ministral-8b_snap_hyre_k10.jsonl` | generation rows 571/571, zero errors/missing passages/fallbacks/parse failures/answer-artifact passages/retries, passage chars min/p50/p95/max 290/466/605/686; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.6200, Hit@10 0.7040, MRR@10 0.5110 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `groq-llama70b` `rag_hyde` | `caches/hyre/full/legalbench_scalr_qfull_seed42_groq-llama70b_rag_hyde.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | generation rows 571/571, zero errors/missing passages/fallbacks/parse failures/answer-artifact passages/retries, passage chars min/p50/p95/max 437/695/846/957; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.6147, Hit@10 0.6953, MRR@10 0.5015 | ✅ GENERATED-CACHE-CLEAN |
| LegalBench-SCALR `groq-llama70b` `snap_hyre` | `caches/hyre/full/legalbench_scalr_qfull_seed42_groq-llama70b_snap_hyre.jsonl`; `caches/retrieval/full/legalbench_scalr_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | generation rows 571/571, zero errors/missing passages/fallbacks/parse failures/answer-artifact passages/retries, passage chars min/p50/p95/max 244/461/582/714; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.5517, Hit@10 0.6462, MRR@10 0.4126 | ✅ GENERATED-CACHE-CLEAN |
| BarExamQA `groq-llama70b` `rag_hyde` | `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_rag_hyde.jsonl`; `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | generation rows 1195/1195, zero errors/missing passages/fallback keys/parse failures/answer-artifact passages, passage chars min/p50/p95/max 466/703/866/1169; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.1046, Hit@10 0.1757, MRR@10 0.0609 | ✅ GENERATED-CACHE-CLEAN |
| BarExamQA `groq-llama70b` `snap_hyre` | `caches/hyre/full/barexam_qfull_seed42_groq-llama70b_snap_hyre.jsonl`; `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | generation rows 1195/1195, zero errors/missing passages/fallback keys/parse failures/answer-artifact passages, passage chars min/p50/p95/max 281/447/562/666; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.1105, Hit@10 0.1849, MRR@10 0.0663 | ✅ GENERATED-CACHE-CLEAN |
| BarExamQA `groq-llama8b` `snap_hyre` | `caches/hyre/full/barexam_qfull_seed42_groq-llama8b_snap_hyre.jsonl`; `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl` | generation rows 1195/1195 with 216 logged same-model generation-format retries; zero errors, missing passages, fallback keys, parse failures, `hyde_contains_answer_artifact` rows, missing snap letters, or near-cap generation; passage chars min/p50/p95/max 96/339/484/842; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.0954, Hit@10 0.1481, MRR@10 0.0538 | ⚠️ GENERATED-CACHE-CLEAN/RETRY-CAVEAT |
| CaseHOLD `groq-llama70b` `rag_hyde` | `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_rag_hyde.jsonl`; `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | generation rows 3600/3600, zero errors/missing passages/fallbacks/parse failures/answer-artifact passages/think artifacts, one call per row, max output 193 tokens; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.5122, Hit@10 0.5914, MRR@10 0.4090 | ✅ GENERATED-CACHE-CLEAN |
| CaseHOLD `groq-llama70b` `snap_hyre` | `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_snap_hyre.jsonl`; `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | generation rows 3600/3600 after strict snap-final-line repair, zero errors/missing passages/missing snap letters/fallbacks/parse failures/answer-artifact passages/think artifacts, one call per row, max output 304 tokens; two malformed snap metadata rows (`ch_test_1108`, `ch_test_3118`) were regenerated with the same provider/model and merged before signoff; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, no_gold=0, Hit@5 0.4497, Hit@10 0.5289, MRR@10 0.3390 | ✅ GENERATED-CACHE-CLEAN |

HousingQA cache caveat: the golden-neighbor full cache was rebuilt with
`retrieval_backend=stored_gold_embedding` for all 6853 rows because the
canonical local text-embedding path OOM-killed on the 1.8M-document Housing
Chroma index. The neighbor query uses the persisted embedding for the gold
statute id(s), then keeps MiniLM cross-encoder reranking enabled. A
cross-encoder-only `CROSS_ENCODER_MAX_CHARS=4096` cap prevents OOM on very long
statute text; this cap is recorded in cache metadata and does not truncate final
LLM evidence.

MASLegalBench cache caveat: MASLegalBench has no official per-question qrels or
gold passage ids. The retrieval exposure below is therefore a source-document
proxy: a row counts as retrieved when at least one top-k passage comes from the
same penalty-notice source document as the question. Do not report these values
as gold-passage Hit@k.

| Dataset / provider / mode | Cache evidence | Health | Sign-off |
|---|---|---|---|
| MASLegalBench raw question | `caches/retrieval/full/mas_legal_bench_qfull_seed42_raw_question_k10.jsonl` | 303 rows, duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.7261 / MRR@5 0.6277; CE doc truncation 0, query truncation 0 | ✅ RETRIEVAL-CACHE-CLEAN-SOURCE-PROXY |
| MASLegalBench `groq-llama8b` `rag_hyde` | `caches/generation/full/mas_legal_bench_qfull_seed42_groq-llama8b_rag_hyde.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl` | generation rows 303/303 after one same-model repair of a malformed repeated row; zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.4818 / MRR@5 0.2838; CE doc truncation 0, query truncation 0 | ⚠️ GENERATED-CACHE-CLEAN-SOURCE-PROXY/REPAIR-CAVEAT |
| MASLegalBench `groq-llama8b` `snap_hyre` | `caches/generation/full/mas_legal_bench_qfull_seed42_groq-llama8b_snap_hyre.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl` | generation rows 303/303 with 89 logged same-model format retries; zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, missing snap letters, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.3993 / MRR@5 0.2175; CE doc truncation 0, query truncation 0 | ⚠️ GENERATED-CACHE-CLEAN-SOURCE-PROXY/RETRY-CAVEAT |
| MASLegalBench `or-gemma4-26b` `rag_hyde` | `caches/generation/full/mas_legal_bench_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl` | generation rows 303/303, all `OPENROUTER_PROVIDER_ONLY=Cloudflare`, zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.3894 / MRR@5 0.2221; CE doc truncation 0, query truncation 0 | ✅ GENERATED-CACHE-CLEAN-SOURCE-PROXY |
| MASLegalBench `or-gemma4-26b` `snap_hyre` | `caches/generation/full/mas_legal_bench_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl` | generation rows 303/303, all `OPENROUTER_PROVIDER_ONLY=Cloudflare`, zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, missing snap letters, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.3531 / MRR@5 0.2182; CE doc truncation 0, query truncation 0 | ✅ GENERATED-CACHE-CLEAN-SOURCE-PROXY |
| MASLegalBench `groq-llama70b` `rag_hyde` | `caches/generation/full/mas_legal_bench_qfull_seed42_groq-llama70b_rag_hyde.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl` | generation rows 303/303, zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.4587 / MRR@5 0.2551; CE doc truncation 0, query truncation 0 | ✅ GENERATED-CACHE-CLEAN-SOURCE-PROXY |
| MASLegalBench `groq-llama70b` `snap_hyre` | `caches/generation/full/mas_legal_bench_qfull_seed42_groq-llama70b_snap_hyre.jsonl`; `caches/retrieval/full/mas_legal_bench_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl` | generation rows 303/303, zero errors, missing passages, fallbacks, answer-artifact passages, parse failures, missing snap letters, near-cap generation, or passages >2500 chars; retrieval duplicate_keys=0, missing_idx=0, empty=0, short=0, official rows_without_gold=303; source-proxy same-source@5 0.4323 / MRR@5 0.2421; CE doc truncation 0, query truncation 0 | ✅ GENERATED-CACHE-CLEAN-SOURCE-PROXY |

Full answer rows:

| Dataset | Provider | Mode | Detail log | Accuracy | Health | Sign-off |
|---|---|---|---|---:|---|---|
| Legal-Link-EU | `or-gemma4-26b` | `llm_only` | `logs/eval_llm_only_or-gemma4-26b_20260519_021543_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-llm_only-nfull-k5_detail.jsonl` | 836/1127 = 74.2% | no retrieval evidence by design; OpenRouter routing explicitly pinned/logged as `OPENROUTER_PROVIDER_ONLY=Cloudflare`, with row-level provider route `{"openrouter_provider_only": "Cloudflare"}` on 1127/1127 rows; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 1127, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 1863 tokens, max final-answer chars 6785 at `complex_legallink_32005D0257_32000D0256_extends_validity__row0479`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`llm_only`/`legal_link_eu`, retrieved/evidence length 0 on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 1 valid same-model answer-format retry, no near-cap rows and no retry-near-cap rows; paired vs `groq-llama70b` `llm_only`: +4.88pp, b/c=178/123, p=0.00181; paired vs `groq-llama8b` `llm_only`: +26.18pp, b/c=369/74, p=3.53e-48 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `or-gemma4-26b` | `rag_simple` | `logs/eval_rag_simple_or-gemma4-26b_20260519_052910_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-rag_simple-nfull-k5_detail.jsonl` | 885/1127 = 78.5% | strict raw retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_raw_question_ce22000_k10.jsonl`: 1127/1127 cache hits, retrieved/evidence length 5 on all rows, 1021/1127 gold retrieved, Hit@5 0.9059 / MRR@5 0.7621, source_doc retrieved on 863/1127 rows and target_doc on 659/1127; OpenRouter route explicitly pinned/logged as `OPENROUTER_PROVIDER_ONLY=Cloudflare`, with row-level provider route `{"openrouter_provider_only": "Cloudflare"}` on 1127/1127 rows; `CROSS_ENCODER_MAX_CHARS=22000` persisted on all rows, CE doc truncation 0 and query truncation 0; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 1832 tokens, max final-answer chars 6541 at `complex_legallink_32004R2247_31997R1495_repeals__row0470`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_simple`/`legal_link_eu`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, no near-cap rows, and 6 valid same-model answer-format retries; paired vs `llm_only`: +4.35pp, b/c=176/127, p=0.00573, CI [1.33, 7.36] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `or-gemma4-26b` | `golden_passage` | `logs/eval_golden_passage_or-gemma4-26b_20260519_091425_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-golden_passage-nfull-k5_detail.jsonl` | 1082/1127 = 96.0% | oracle gold injected/retrieved on 1127/1127 rows, evidence-store length 1 on all rows, retrieved gold-id list length 5 on 1118 rows and 4 on 9 rows, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; OpenRouter route explicitly pinned/logged as `OPENROUTER_PROVIDER_ONLY=Cloudflare`, with row-level provider route `{"openrouter_provider_only": "Cloudflare"}` on 1127/1127 rows; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 1906 tokens, max final-answer chars 6602 at `complex_legallink_32000L0048_31986L0362_annex_2_completes__row0194`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`golden_passage`/`legal_link_eu`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, no near-cap rows, no retry-near-cap rows, and 6 valid same-model answer-format retries with max retry 565 tokens; paired vs `rag_simple`: +17.48pp, b/c=214/17, p=1.47e-44, CI [15.08, 19.96]; paired vs `llm_only`: +21.83pp, b/c=264/18, p=3.15e-57, CI [19.25, 24.49] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `or-gemma4-26b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260519_121534_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-golden_plus_neighbors-nfull-k5_detail.jsonl` | 1068/1127 = 94.8% | strict golden-neighbor retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_golden_neighbors_k10.jsonl`: 1127/1127 retrieval-cache hits, evidence-store length 5 on all rows, 1127/1127 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000, retrieved-id list length 9 on 1118 rows and 8 on 9 rows; OpenRouter route explicitly pinned/logged as `OPENROUTER_PROVIDER_ONLY=Cloudflare`, with row-level provider route `{"openrouter_provider_only": "Cloudflare"}` on 1127/1127 rows; wrapper stdout recorded one same-route transient OpenRouter 502/idle-timeout retry and continued without model/provider fallback; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 1904 tokens, max final-answer chars 6556 at `complex_legallink_32008R0950R(01)_32008R0950_corrects__row0715`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`golden_plus_neighbors`/`legal_link_eu`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, parse failures 0, 6 valid same-model answer-format retries, no single-call near-cap rows and no retry-near-cap rows; paired vs `rag_simple`: +16.24pp, b/c=203/20, p=2.60e-39, CI [13.84, 18.63]; paired vs `llm_only`: +20.59pp, b/c=260/28, p=2.77e-48, CI [17.92, 23.25]; paired vs `golden_passage`: -1.24pp, b/c=17/31, p=0.0595; wrapper ended with a post-OK shell parse artifact after the detail log and experiment summary were already written, and current `scripts/local/run_answer_cell.sh` passes `bash -n` | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-ROUTE-502/POSTRUN-SHELL-CAVEAT |
| Legal-Link-EU | `or-gemma4-26b` | `rag_hyde` | `logs/merged/eval_rag_hyde_or-gemma4-26b_20260519_legal_link_eu_merged_detail.jsonl` | 836/1127 = 74.2% | strict HyDE generation/retrieval-cache replay: 1127/1127 HyDE cache hits, 1127/1127 retrieval-cache hits, retrieved/evidence length 5 on all rows, 552/1127 gold retrieved, Hit@5 0.4898 / MRR@5 0.3637 from signed cache `caches/retrieval/full/legal_link_eu_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`; explicit same-model OpenRouter route recovery: first 713 clean rows pinned to `OPENROUTER_PROVIDER_ONLY=Cloudflare`, then Cloudflare degradation/throttling was stopped and rows 713:end were replayed with `OPENROUTER_PROVIDER_ONLY=Parasail` for the same `google/gemma-4-26b-a4b-it` model; `scripts/merge_detail_logs.py` produced 1127 unique rows, 0 empty retrieval rows, avg LLM calls 1.001; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max cumulative output 2596 tokens, max final-answer chars 6950 at `complex_legallink_32002D0242_32001D0327_extends_validity__row0219`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_hyde`/`legal_link_eu`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, parse failures 0, 1 valid same-model answer-format retry on `complex_legallink_31995R2625R(01)_31995R2625_corrects__row0138`, and 1 cumulative output row >=1900 tokens with intact final line; paired vs `llm_only`: 0.00pp, b/c=122/122, p=1.000; paired vs `rag_simple`: -4.35pp, b/c=110/159, p=0.00335, CI [-7.10, -1.33]; paired vs `golden_plus_neighbors`: -20.59pp, b/c=23/255, p=1.12e-50 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE-RECOVERY-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `rag_rewrite` | `logs/merged/eval_rag_rewrite_groq-llama8b_20260519_legal_link_eu_nfull_k5_merged_detail.jsonl` | 600/1127 = 53.2% | merged repair log: original partial `logs/eval_rag_rewrite_groq-llama8b_20260519_025741_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-rag_rewrite-nfull-k5_detail.jsonl` stopped on a Groq spend-threshold row, and repair tail `logs/eval_rag_rewrite_groq-llama8b_20260519_040100_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-rag_rewrite-nfull-k5_detail.jsonl` reran rows 343:end with same provider/model/method; `scripts/merge_detail_logs.py --on-duplicate last` produced 1127 unique rows, 0 empty retrieval rows, 429/1127 gold retrieved, retrieval exposure Hit@5 0.3807 / MRR@5 0.2934, avg LLM calls 2.083; dynamic rewrite retrieval, not cache replay: retrieved/evidence length 5 on all rows, rewrite JSON parse kind `json` on all rows, 0 rewrite fallbacks, 0 partial-JSON repairs, 0 row errors, provider route `{}` on all rows, source_doc retrieved on 302/1127 rows and target_doc on 244/1127; launch stdout recorded `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation count rows 0 and CE query truncation rows 0, but the merged detail row-level `cross_encoder_max_chars` field is blank because dynamic non-cache logging did not persist the env value before the 2026-05-19 patch; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2169 cumulative tokens, max final-answer chars 8534 at `complex_legallink_32005D1554_32001D0051_extends_validity__row0487`; custom no-silent audit: exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 93 valid same-model answer-format retries, 0 single-call near-cap rows and 0 retry-near-cap rows; paired vs `llm_only`: +5.24pp, b/c=226/167, p=0.00338; paired vs `rag_simple`: -4.17pp, b/c=188/235, p=0.0252; paired vs `rag_hyde`: -0.27pp, b/c=208/211, p=0.922; paired vs `snap_hyre`: +1.24pp, b/c=215/201, p=0.524; paired vs `golden_passage`: -18.46pp, b/c=91/299, p=5.63e-27; paired vs `golden_plus_neighbors`: -17.48pp, b/c=117/314, p=6.89e-22 | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR/RETRY/CE-MAX-LOGGING-CAVEAT |
| MASLegalBench | `groq-llama8b` | `llm_only` | `logs/eval_llm_only_groq-llama8b_20260518_193238_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-llm_only-nfull-k5_detail.jsonl` | 269/303 = 88.8% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 303, long rows 0, fallback markers 0, max output 397 tokens, max final-answer chars 2208; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`llm_only`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, 1 valid same-model answer-format retry, near-cap rows 0 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| MASLegalBench | `groq-llama8b` | `rag_simple` | `logs/eval_rag_simple_groq-llama8b_20260518_193556_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-rag_simple-nfull-k5_detail.jsonl` | 261/303 = 86.1% | strict raw retrieval-cache replay: 303/303 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from `caches/retrieval/full/mas_legal_bench_qfull_seed42_raw_question_k10.jsonl`: same-source@5 0.7261 / MRR@5 0.6277; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 2893; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_simple`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, 3 valid same-model answer-format retries, 3 rows at >=1900 output tokens with intact final lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| MASLegalBench | `groq-llama8b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260518_193947_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-rag_hyde-nfull-k5_detail.jsonl` | 267/303 = 88.1% | strict generated/retrieval-cache replay: 303/303 HyDE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.4818 / MRR@5 0.2838; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 2694; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_hyde`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, 6 valid same-model answer-format retries, 5 rows at >=1900 output tokens with intact final lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| MASLegalBench | `groq-llama8b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama8b_20260518_194401_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-snap_hyre-nfull-k5_detail.jsonl` | 267/303 = 88.1% | strict generated/retrieval-cache replay: 303/303 HyRE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.3993 / MRR@5 0.2175; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 3012; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`snap_hyre`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, 3 valid same-model answer-format retries, 3 rows at >=1900 output tokens with intact final lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| MASLegalBench | `groq-llama8b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama8b_20260518_195621_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-rag_rewrite-nfull-k5_detail.jsonl` | 262/303 = 86.5% | dynamic rewrite retrieval after tightening `skills/query_rewriter.md` to require bare JSON; 303/303 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 fallback keys, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from detail log: same-source@5 0.6007 / MRR@5 0.4234; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2140 tokens, max final-answer chars 2840; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_rewrite`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, 5 valid same-model answer-format retries, 3 rows at >=1900 output tokens with intact final lines; supersedes aborted pre-tightening partial log `logs/eval_rag_rewrite_groq-llama8b_20260518_195305_mas_legal_bench_local-snap-hyre-groq-llama8b-mas_legal_bench-rag_rewrite-nfull-k5_detail.jsonl` | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| MASLegalBench | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260518_200511_mas_legal_bench_local-snap-hyre-groq-llama70b-mas_legal_bench-llm_only-nfull-k5_detail.jsonl` | 280/303 = 92.4% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 303, long rows 0, fallback markers 0, max output 603 tokens, max final-answer chars 3318; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`llm_only`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 0, near-cap rows 0 | ✅ COMPREHENSIVE-CLEAN |
| MASLegalBench | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260518_201222_mas_legal_bench_local-snap-hyre-groq-llama70b-mas_legal_bench-rag_simple-nfull-k5_detail.jsonl` | 281/303 = 92.7% | strict raw retrieval-cache replay: 303/303 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from `caches/retrieval/full/mas_legal_bench_qfull_seed42_raw_question_k10.jsonl`: same-source@5 0.7261 / MRR@5 0.6277; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 641 tokens, max final-answer chars 3847; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 0, retrieved list length 5 on all rows, CE doc truncation 0, query truncation rows 0, near-cap rows 0 | ✅ COMPREHENSIVE-CLEAN |
| MASLegalBench | `groq-llama70b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260518_202024_mas_legal_bench_local-snap-hyre-groq-llama70b-mas_legal_bench-rag_hyde-nfull-k5_detail.jsonl` | 280/303 = 92.4% | strict generated/retrieval-cache replay: 303/303 HyDE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.4587 / MRR@5 0.2551; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 750 tokens, max final-answer chars 4068; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_hyde`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 0, retrieved list length 5 on all rows, HyDE cache hits 303/303, CE doc truncation 0, query truncation rows 0, near-cap rows 0 | ✅ COMPREHENSIVE-CLEAN |
| MASLegalBench | `groq-llama70b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama70b_20260518_202740_mas_legal_bench_local-snap-hyre-groq-llama70b-mas_legal_bench-snap_hyre-nfull-k5_detail.jsonl` | 279/303 = 92.1% | strict generated/retrieval-cache replay: 303/303 HyRE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.4323 / MRR@5 0.2421; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 782 tokens, max final-answer chars 4152; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`snap_hyre`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 0, retrieved list length 5 on all rows, HyRE cache hits 303/303, CE doc truncation 0, query truncation rows 0, near-cap rows 0; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CLEAN-SOURCE-PROXY |
| MASLegalBench | `groq-llama70b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama70b_20260518_203613_mas_legal_bench_local-snap-hyre-groq-llama70b-mas_legal_bench-rag_rewrite-nfull-k5_detail.jsonl` | 284/303 = 93.7% | dynamic rewrite retrieval, no deterministic retrieval cache; 303/303 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 fallback keys, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from detail log: same-source@5 0.4851 / MRR@5 0.2727; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 1067 tokens, max final-answer chars 5560; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_rewrite`/`mas_legal_bench`, exact final `Answer: (X)` lines on all rows, fallback keys 0, 1 valid same-model answer-format retry, retrieved list length 5 on all rows, near-cap rows 0; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SOURCE-PROXY |
| MASLegalBench | `or-gemma4-26b` | `llm_only` | `logs/eval_llm_only_or-gemma4-26b_20260518_200512_mas_legal_bench_local-snap-hyre-or-gemma4-26b-mas_legal_bench-llm_only-nfull-k5_detail.jsonl` | 290/303 = 95.7% | no retrieval evidence by design; explicit OpenRouter route: `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 303 rows for `google/gemma-4-26b-a4b-it`; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 303, long rows 0, fallback markers 0, max output 721 tokens, max final-answer chars 3388; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`llm_only`/`mas_legal_bench`, provider route exactly `openrouter_provider_only=Cloudflare`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 0, think tags 0, near-cap rows 0 | ✅ COMPREHENSIVE-CLEAN-SAME-MODEL-ROUTE |
| MASLegalBench | `or-gemma4-26b` | `rag_simple` | `logs/eval_rag_simple_or-gemma4-26b_20260518_204324_mas_legal_bench_local-snap-hyre-or-gemma4-26b-mas_legal_bench-rag_simple-nfull-k5_detail.jsonl` | 287/303 = 94.7% | explicit OpenRouter route: `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 303 rows for `google/gemma-4-26b-a4b-it`; strict raw retrieval-cache replay: 303/303 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from `caches/retrieval/full/mas_legal_bench_qfull_seed42_raw_question_k10.jsonl`: same-source@5 0.7261 / MRR@5 0.6277; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 1869 tokens, max final-answer chars 6643; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_simple`/`mas_legal_bench`, provider route exactly `openrouter_provider_only=Cloudflare`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 1, retrieved list length 5 on all rows, CE doc truncation 0, query truncation rows 0, near-cap rows 0; paired vs `llm_only`: -0.99pp, b/c=5/8, p=0.581; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE/SOURCE-PROXY |
| MASLegalBench | `or-gemma4-26b` | `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260518_213146_mas_legal_bench_local-snap-hyre-or-gemma4-26b-mas_legal_bench-rag_hyde-nfull-k5_detail.jsonl` | 289/303 = 95.4% | explicit OpenRouter route: `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 303 rows for `google/gemma-4-26b-a4b-it`; strict generated/retrieval-cache replay: 303/303 HyDE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.3894 / MRR@5 0.2221; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 1518 tokens, max final-answer chars 6757 at `maslb_46fb8d1d92853340`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_hyde`/`mas_legal_bench`, provider route exactly `openrouter_provider_only=Cloudflare`, exact final `Answer: (X)` lines on all rows, answer-format retries 0, retrieved list length 5 on all rows, `hyde_used_fallback=False` on all rows, CE doc truncation 0, query truncation rows 0, near-cap rows 0; paired vs `llm_only`: -0.33pp, b/c=6/7, p=1.000; paired vs `rag_simple`: +0.66pp, b/c=7/5, p=0.774; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CLEAN-SAME-MODEL-ROUTE/SOURCE-PROXY |
| MASLegalBench | `or-gemma4-26b` | `snap_hyre` | `logs/eval_snap_hyre_or-gemma4-26b_20260518_221830_mas_legal_bench_local-snap-hyre-or-gemma4-26b-mas_legal_bench-snap_hyre-nfull-k5_detail.jsonl` | 283/303 = 93.4% | explicit OpenRouter route: `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 303 rows for `google/gemma-4-26b-a4b-it`; strict generated/retrieval-cache replay: 303/303 HyRE cache hits, 303/303 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from signed cache: same-source@5 0.3531 / MRR@5 0.2182; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 1715 tokens, max final-answer chars 7802 at `maslb_1b6eb9d31902867c`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`snap_hyre`/`mas_legal_bench`, provider route exactly `openrouter_provider_only=Cloudflare`, exact final `Answer: (X)` lines on all rows, fallback keys 0, answer-format retries 1, retrieved list length 5 on all rows, HyRE cache hits 303/303, CE doc truncation 0, query truncation rows 0, near-cap rows 0; paired vs `rag_hyde`: -1.98pp, b/c=3/9, p=0.146; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE/SOURCE-PROXY |
| MASLegalBench | `or-gemma4-26b` | `rag_rewrite` | `logs/eval_rag_rewrite_or-gemma4-26b_20260518_230946_mas_legal_bench_local-snap-hyre-or-gemma4-26b-mas_legal_bench-rag_rewrite-nfull-k5_detail.jsonl` | 284/303 = 93.7% | explicit OpenRouter route: `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 303 rows for `google/gemma-4-26b-a4b-it`; dynamic rewrite retrieval, no deterministic retrieval cache; 303/303 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 rewrite fallbacks, retrieved list length 5 on all rows, 0 empty evidence rows; source-proxy retrieval from detail log: same-source@5 0.5149 / MRR@5 0.3202; `scripts/analyze_detail_flags.py`: rows 303, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 1747 tokens, max final-answer chars 7107 at `maslb_46fb8d1d92853340`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_rewrite`/`mas_legal_bench`, provider route exactly `openrouter_provider_only=Cloudflare`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, answer-format retries 0, CE doc truncation 0, query truncation rows 0, near-cap rows 0; paired vs `rag_simple`: -0.99pp, b/c=4/7, p=0.549; paired vs `snap_hyre`: +0.33pp, b/c=7/6, p=1.000; MAS retrieval numbers are same-source-document proxies because official per-question qrels are absent | ⚠️ COMPREHENSIVE-CLEAN-SAME-MODEL-ROUTE/SOURCE-PROXY |
| BarExamQA | `groq-llama8b` | `llm_only` | `logs/eval_llm_only_groq-llama8b_20260518_205159_barexam_local-snap-hyre-groq-llama8b-barexam-llm_only-nfull-k5_detail.jsonl` | 685/1195 = 57.3% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 1195, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 3431; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`llm_only`/`barexam`, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 9 valid same-model answer-format retries, 6 rows at >=1900 total output tokens with intact final lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `rag_simple` | `logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl` | 651/1195 = 54.5% | strict raw retrieval-cache replay: 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 17/1195 gold retrieved; retrieval exposure from raw cache: Hit@5 0.0142, MRR@5 0.0068; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 5372; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_simple`/`barexam`, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 13 valid same-model answer-format retries, 11 rows at >=1900 total output tokens with intact final lines; paired vs `llm_only`: -2.85pp, b/c=169/203, p=0.0870 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `golden_passage` | `logs/eval_golden_passage_groq-llama8b_20260518_213011_barexam_local-snap-hyre-groq-llama8b-barexam-golden_passage-nfull-k5_detail.jsonl` | 717/1195 = 60.0% | oracle gold injected/retrieved on 1195/1195 rows, retrieved list length 1 on all rows, 0 empty evidence rows, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 2053 tokens, max final-answer chars 4784; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_passage`/`barexam`, 1195/1195 gold retrieved, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 15 valid same-model answer-format retries, 14 rows at >=1900 total output tokens with intact final lines; paired vs `rag_simple`: +5.52pp, b/c=241/175, p=0.00141; paired vs `llm_only`: +2.68pp, b/c=217/185, p=0.122 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama8b_20260518_215012_barexam_local-snap-hyre-groq-llama8b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | 738/1195 = 61.8% | strict golden-neighbor retrieval-cache replay: 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1195/1195 gold retrieved; retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 2053 tokens, max final-answer chars 6070 at `mbe_213`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_plus_neighbors`/`barexam`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 14 valid same-model answer-format retries, 13 rows at >=1900 total output tokens with intact final lines, CE doc truncation 0, query truncation rows 0; paired vs `rag_simple`: +7.28pp, b/c=256/169, p=2.85e-05; paired vs `llm_only`: +4.44pp, b/c=249/196, p=0.0136; paired vs `golden_passage`: +1.76pp, b/c=165/144, p=0.255 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260518_225428_barexam_local-snap-hyre-groq-llama8b-barexam-rag_hyde-nfull-k5_detail.jsonl` | 670/1195 = 56.1% | strict generated/retrieval-cache replay: 1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 99/1195 gold retrieved; retrieval exposure from signed cache: Hit@5 0.0828, MRR@5 0.0452; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 2053 tokens, max final-answer chars 4243 at `mbe_94`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_hyde`/`barexam`, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 17 valid same-model answer-format retries, 15 rows at >=1900 total output tokens with intact final lines, CE doc truncation 74, query truncation rows 0; paired vs `rag_simple`: +1.59pp, b/c=196/177, p=0.351; paired vs `llm_only`: -1.26pp, b/c=200/215, p=0.492 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl` | 680/1195 = 56.9% | strict generated/retrieval-cache replay: 1195/1195 HyRE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 114/1195 gold retrieved; retrieval exposure from signed cache: Hit@5 0.0954, MRR@5 0.0469; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 2053 tokens, max final-answer chars 4114 at `mbe_1138`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`snap_hyre`/`barexam`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, fallback keys 0, parse failures 0, think tags 0, 15 valid same-model answer-format retries, 11 rows at >=1900 total output tokens with intact final lines, CE doc truncation 94, query truncation rows 0; paired vs `rag_simple`: +2.43pp, b/c=200/171, p=0.146; paired vs `rag_hyde`: +0.84pp, b/c=177/167, p=0.628; paired vs `llm_only`: -0.42pp, b/c=196/201, p=0.841; paired vs `golden_plus_neighbors`: -4.85pp, b/c=167/225, p=0.00393 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `groq-llama8b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama8b_20260518_233753_barexam_local-snap-hyre-groq-llama8b-barexam-rag_rewrite-nfull-k5_detail.jsonl` | 685/1195 = 57.3% | dynamic rewrite retrieval, no deterministic retrieval cache; 1195/1195 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 rewrite fallbacks, retrieved list length 5 on all rows, 0 empty evidence rows, 105/1195 gold retrieved; retrieval exposure from detail log: Hit@5 0.0879, MRR@5 0.0469; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 2.01, max output 2150 tokens, max final-answer chars 4389 at `mbe_758`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_rewrite`/`barexam`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 15 valid same-model answer-format retries, 14 rows at >=1900 total output tokens with intact final lines, CE doc truncation 43, query truncation rows 0; paired vs `rag_simple`: +2.85pp, b/c=218/184, p=0.0997; paired vs `snap_hyre`: +0.42pp, b/c=209/204, p=0.844 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-ministral-8b` | `llm_only` | `logs/eval_llm_only_or-ministral-8b_20260517_172732_barexam_local-snap-hyre-or-ministral-8b-barexam-llm_only-nfull-k5_detail.jsonl` | 679/1195 = 56.8% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 1195, long rows 0, fallback markers 0, avg LLM calls 1.06, max output 2070 tokens, max final-answer chars 9911 at `mbe_266`; custom no-silent audit: provider/mode/dataset exactly `or-ministral-8b`/`llm_only`/`barexam`, exact-final-line issues 0, fallback keys 0, think tags 0, 74 valid same-model answer-format retries, 9 rows at >=1900 total output tokens (`mbe_266`, `mbe_724`, `mbe_755`, `mbe_855`, `mbe_924`, `mbe_1081`, `mbe_1093`, `mbe_6`, `mbe_608`); all retried near-cap rows ended with short exact final `Answer:` lines and non-retried near-cap rows had intact final `Answer:` lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-ministral-8b` | `rag_simple` | `logs/eval_rag_simple_or-ministral-8b_20260517_195925_barexam_local-snap-hyre-or-ministral-8b-barexam-rag_simple-nfull-k5_detail.jsonl` | 680/1195 = 56.9% | strict raw retrieval-cache replay: 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 17/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_or-ministral-8b_rag_simple.md`: Hit@5 0.0142, MRR@5 0.0068; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2091 tokens, max final-answer chars 9682; custom no-silent audit: provider/mode/dataset exactly `or-ministral-8b`/`rag_simple`/`barexam`, exact-final-line issues 0, fallback keys 0, think tags 0, 27 valid same-model answer-format retries, 22 rows at >=1900 total output tokens, all near-cap rows were retry rows ending with short exact final `Answer:` lines; paired vs `llm_only`: +0.08pp, b/c=156/155, p=1.000 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-ministral-8b` | `golden_passage` | `logs/eval_golden_passage_or-ministral-8b_20260517_225625_barexam_local-snap-hyre-or-ministral-8b-barexam-golden_passage-nfull-k5_detail.jsonl` | 772/1195 = 64.6% | oracle gold injected/retrieved on 1195/1195 rows, retrieved list length 1 on all rows, 0 empty evidence rows; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2112 tokens, max final-answer chars 10173 at `mbe_233`; custom no-silent audit: provider/mode/dataset exactly `or-ministral-8b`/`golden_passage`/`barexam`, 1195/1195 gold retrieved, exact-final-line issues 0, fallback keys 0, think tags 0, 37 valid same-model answer-format retries, 34 rows at >=1900 total output tokens with intact final `Answer:` lines; one verbose non-retried row `mbe_233` reached 10173 chars but ended with exact `Answer: (C)` and did not trigger analyzer long-row failure; paired vs `rag_simple`: +7.70pp, b/c=205/113, p=2.78e-07; paired vs `llm_only`: +7.78pp, b/c=206/113, p=2.14e-07 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-ministral-8b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-ministral-8b_20260518_015518_barexam_local-snap-hyre-or-ministral-8b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | 755/1195 = 63.2% | strict golden-neighbor retrieval-cache replay: 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1195/1195 gold retrieved; retrieval cache audit from `caches/retrieval/full/barexam_qfull_seed42_golden_neighbors_k10.jsonl`: Hit@5 1.0000, Recall@5 1.0000, MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.03, max output 2087 tokens, max final-answer chars 9848 at `mbe_979`; custom no-silent audit: provider/mode/dataset exactly `or-ministral-8b`/`golden_plus_neighbors`/`barexam`, exact-final-line issues 0, fallback keys 0, think tags 0, 30 valid same-model answer-format retries, 28 rows at >=1900 total output tokens, zero runner near-cap failures under the 2048-token margin, zero long rows; paired vs `rag_simple`: +6.28pp, b/c=205/130, p=4.93e-05; paired vs `llm_only`: +6.36pp, b/c=199/123, p=2.70e-05; paired vs `golden_passage`: -1.42pp, b/c=111/128, p=0.301 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| LegalBench-SCALR | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260514_1414_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` | 425/571 = 74.4% | `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, near-cap rows 0 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260515_171203_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | 416/571 = 72.9% | strict raw retrieval-cache replay: 571/571 cache hits, 0 empty evidence rows, 283/571 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_simple.md`: Hit@5 0.4956, MRR@5 0.3447; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, answer retries 0, max output 760 tokens, near-cap violations 0; paired vs `llm_only`: -1.58pp, b/c=29/38, p=0.328 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `groq-llama70b` | `golden_passage` | `logs/eval_golden_passage_groq-llama70b_20260515_173003_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | 534/571 = 93.5% | oracle gold injected on 571/571 rows with 571/571 evidence rows; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 8 valid same-model final-answer retries, retry max 5 tokens, max output 603 tokens, near-cap violations 0; paired vs `rag_simple`: +20.67pp, b/c=119/1, p=1.82e-34; paired vs `llm_only`: +19.09pp, b/c=110/1, p=8.63e-32 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `groq-llama70b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama70b_20260515_174642_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | 474/571 = 83.0% | strict golden-neighbor retrieval-cache replay: 571/571 cache hits, 571/571 gold retrieved, 571/571 evidence rows; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 2 valid same-model final-answer retries, retry max 5 tokens, max output 750 tokens, near-cap violations 0; paired vs `rag_simple`: +10.16pp, b/c=73/15, p=2.56e-10; paired vs `golden_passage`: -10.51pp, b/c=1/61, p=2.73e-17; paired vs `llm_only`: +8.58pp, b/c=63/14, p=1.41e-08 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `groq-llama70b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260515_181837_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | 402/571 = 70.4% | strict generated/retrieval-cache replay: 571/571 HyDE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 351/571 gold retrieved; retrieval exposure from signed cache: Hit@5 0.6147, MRR@10 0.5015; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, answer retries 0, max output 724 tokens, near-cap violations 0; paired vs `rag_simple`: -2.45pp, b/c=23/37, p=0.0925; paired vs `llm_only`: -4.03pp, b/c=13/36, p=0.00140; paired vs `golden_plus_neighbors`: -12.61pp, b/c=8/80, p=4.60e-16 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `groq-llama70b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama70b_20260515_183326_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | 407/571 = 71.3% | strict generated/retrieval-cache replay: 571/571 HyRE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 315/571 gold retrieved; retrieval exposure from signed cache: Hit@5 0.5517, MRR@10 0.4126; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, answer retries 0, max output 789 tokens, near-cap violations 0; paired vs `rag_simple`: -1.58pp, b/c=23/32, p=0.281; paired vs `llm_only`: -3.15pp, b/c=19/37, p=0.0222; paired vs `rag_hyde`: +0.88pp, b/c=24/19, p=0.542; paired vs `golden_plus_neighbors`: -11.73pp, b/c=10/77, p=5.92e-14 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `groq-llama70b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama70b_20260515_184842_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-rag_rewrite-nfull-k5_detail.jsonl` | 409/571 = 71.6% | dynamic rewrite retrieval: 571/571 rewrite JSON parses, 0 rewrite retries, 0 partial-JSON repairs, 0 empty evidence rows, 329/571 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_scalr_groq-llama70b_rag_rewrite.md`: Hit@5 0.5762, MRR@5 0.4327; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 5 valid same-model final-answer retries, retry max 5 tokens, max output 859 tokens, near-cap violations 0; paired vs `rag_simple`: -1.23pp, b/c=28/35, p=0.450; paired vs `snap_hyre`: +0.35pp, b/c=27/25, p=0.890; paired vs `rag_hyde`: +1.23pp, b/c=27/20, p=0.382; paired vs `llm_only`: -2.80pp, b/c=18/34, p=0.0365; paired vs `golden_plus_neighbors`: -11.38pp, b/c=13/78, p=1.85e-12 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260515_191548_barexam_local-snap-hyre-groq-llama70b-barexam-llm_only-nfull-k5_detail.jsonl` | 940/1195 = 78.7% | `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, answer retries 0, max output 738 tokens, max final-answer chars 3681, near-cap violations 0 | ✅ COMPREHENSIVE-CLEAN |
| BarExamQA | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl` | 891/1195 = 74.6% | strict raw retrieval-cache replay: 1195/1195 cache hits, 0 empty evidence rows, 17/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_simple.md`: Hit@5 0.0142, MRR@5 0.0068; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 2 valid same-model final-answer retries (`mbe_272`, `mbe_202`), retry max 5 tokens, max output 1169 tokens, near-cap violations 0; paired vs `llm_only`: -4.10pp, b/c=66/115, p=0.000334 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama70b_20260515_202827_barexam_local-snap-hyre-groq-llama70b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | 930/1195 = 77.8% | strict golden-neighbor cache replay: 1195/1195 cache hits, 1195/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_groq-llama70b_golden_plus_neighbors.md`: Hit@5 1.0000, MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 2 valid same-model final-answer retries (`mbe_273`, `mbe_1098`), retry max 5 tokens, max output 1112 tokens, near-cap violations 0; paired vs `rag_simple`: +3.26pp, b/c=136/97, p=0.0126; paired vs `llm_only`: -0.84pp, b/c=97/107, p=0.529 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `golden_passage` | `logs/eval_golden_passage_groq-llama70b_20260515_210741_barexam_local-snap-hyre-groq-llama70b-barexam-golden_passage-nfull-k5_detail.jsonl` | 946/1195 = 79.2% | oracle gold injected/retrieved on 1195/1195 rows, 0 empty evidence rows; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 3 valid same-model final-answer retries (`mbe_2`, `mbe_197`, `mbe_1125`), retry max 5 tokens, max output 1029 tokens, near-cap violations 0; paired vs `rag_simple`: +4.60pp, b/c=137/82, p=0.000246; paired vs `llm_only`: +0.50pp, b/c=100/94, p=0.720; paired vs `golden_plus_neighbors`: +1.34pp, b/c=76/60, p=0.198 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260515_222654_barexam_local-snap-hyre-groq-llama70b-barexam-rag_hyde-nfull-k5_detail.jsonl` | 958/1195 = 80.2% | strict generated/retrieval-cache replay: 1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, 0 empty evidence rows, 125/1195 gold retrieved; retrieval exposure from signed cache: Hit@5 0.1046, MRR@10 0.0609; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 3 valid same-model final-answer retries (`mbe_90`, `mbe_95`, `mbe_131`), retry max 5 tokens, max output 1061 tokens, near-cap violations 0; paired vs `rag_simple`: +5.61pp, b/c=137/70, p=3.73e-06; paired vs `llm_only`: +1.51pp, b/c=101/83, p=0.210; paired vs `golden_passage`: +1.00pp, b/c=106/94, p=0.437 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl` | 953/1195 = 79.7% | strict generated/retrieval-cache replay: 1195/1195 HyRE cache hits, 1195/1195 retrieval-cache hits, 0 empty evidence rows, 132/1195 gold retrieved; retrieval exposure from signed cache: Hit@5 0.1105, MRR@10 0.0663; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 1 valid same-model final-answer retry (`mbe_2`), retry max 5 tokens, max output 1265 tokens, max final-answer chars 6534, near-cap violations 0; paired vs `rag_simple`: +5.19pp, b/c=138/76, p=2.70e-05; paired vs `llm_only`: +1.09pp, b/c=103/90, p=0.388; paired vs `rag_hyde`: -0.42pp, b/c=79/84, p=0.754; paired vs `golden_passage`: +0.59pp, b/c=110/103, p=0.681; paired vs `golden_plus_neighbors`: +1.92pp, b/c=114/91, p=0.124 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `groq-llama70b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama70b_20260515_234357_barexam_local-snap-hyre-groq-llama70b-barexam-rag_rewrite-nfull-k5_detail.jsonl` | 923/1195 = 77.2% | dynamic rewrite retrieval, no deterministic retrieval cache; 1195/1195 rewrite JSON parses, 0 rewrite-format retries, 1 logged partial-JSON repair, 0 empty retrieval rows, 146/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_groq-llama70b_rag_rewrite.md`: Hit@5 0.1222, MRR@5 0.0565; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, fallback keys 0, 11 valid same-model final-answer retries (`mbe_19`, `mbe_450`, `mbe_606`, `mbe_716`, `mbe_724`, `mbe_754`, `mbe_1081`, `mbe_1106`, `mbe_890`, `mbe_956`, `mbe_1141`), retry max 5 tokens, max output 1004 tokens, near-cap violations 0; paired vs `rag_simple`: +2.68pp, b/c=133/101, p=0.0425; paired vs `llm_only`: -1.42pp, b/c=102/119, p=0.282; paired vs `rag_hyde`: -2.93pp, b/c=84/119, p=0.0168; paired vs `snap_hyre`: -2.51pp, b/c=85/115, p=0.0400; paired vs `golden_passage`: -1.92pp, b/c=105/128, p=0.149 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/REPAIR-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `llm_only` | `logs/merged/eval_llm_only_or-gemma4-26b_cloudflare_tail_20260516_barexam_nfull_k5_detail.jsonl` | 966/1195 = 80.8% | signed with explicit same-model Cloudflare tail caveat: clean first 51 rows from the initial OpenRouter prefix were merged with a 1144-row `OPENROUTER_PROVIDER_ONLY=Cloudflare` tail on `google/gemma-4-26b-a4b-it`; the failed DekaLLM 401 row `mbe_60` is excluded and superseded after `NO_SILENT_FALLBACK` blocked it. `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider all `or-gemma4-26b`, exact-final-line issues 0, fallback keys 0, think tags 0, 3 valid same-model answer-format retries (`mbe_14`, `mbe_586`, `mbe_989`), 4 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2062 tokens, max final-answer chars 7959; one naive fallback-text hit on `mbe_608` is incidental legal explanation text, not provider/method fallback | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `rag_simple` | `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl` | 932/1195 = 78.0% | strict raw retrieval-cache replay: 1195/1195 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 17/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_simple.md`: Hit@5 0.0142, MRR@5 0.0068; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider all `or-gemma4-26b`, exact-final-line issues 0, fallback keys 0, think tags 0, 3 valid same-model answer-format retries (`mbe_576`, `mbe_989`, `mbe_1124`), 3 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2052 tokens, max final-answer chars 7739; paired vs `llm_only`: -2.85pp, b/c=78/112, p=0.0164 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `golden_passage` | `logs/eval_golden_passage_or-gemma4-26b_20260516_200935_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_passage-nfull-k5_detail.jsonl` | 939/1195 = 78.6% | oracle gold injected/retrieved on 1195/1195 rows, retrieved list length 1 on all rows; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`golden_passage`/`barexam`, 1195/1195 gold retrieved, exact-final-line issues 0, fallback keys 0, think tags 0, 4 valid same-model answer-format retries (`mbe_83`, `mbe_312`, `mbe_625`, `mbe_989`), 4 rows at >=1900 output tokens with intact final `Answer:` lines, max output 2023 tokens, max final-answer chars 8849; paired vs `rag_simple`: +0.59pp, b/c=102/95, p=0.669; paired vs `llm_only`: -2.26pp, b/c=78/105, p=0.0543 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260516_233647_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl` | 964/1195 = 80.7% | strict golden-neighbor retrieval-cache replay: retrieved list length 5 on all rows, 1195/1195 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2007 tokens, max final-answer chars 9033; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`golden_plus_neighbors`/`barexam`, exact-final-line issues 0, fallback keys 0, think tags 0, 5 valid same-model answer-format retries (`mbe_532`, `mbe_562`, `mbe_989`, `mbe_1131`, `mbe_563`), 3 rows at >=1900 output tokens with intact final `Answer:` lines (`mbe_713`, `mbe_989`, `mbe_563`); paired vs `rag_simple`: +2.68pp, b/c=116/84, p=0.0281; paired vs `llm_only`: -0.17pp, b/c=91/93, p=0.941; paired vs `golden_passage`: +2.09pp, b/c=91/66, p=0.0551 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl` | 959/1195 = 80.3% | strict HyDE generation/retrieval-cache replay: 1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 136/1195 gold retrieved, retrieval exposure Hit@5 0.1138 / MRR@5 0.0542; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2103 tokens, max final-answer chars 8665; custom no-silent audit: exact-final-line issues 0, fallback keys 0, think tags 0, 5 valid same-model answer-format retries (`mbe_141`, `mbe_291`, `mbe_576`, `mbe_899`, `mbe_989`), 5 rows at >=1900 output tokens with intact final `Answer:` lines; paired vs `rag_simple`: +2.26pp, b/c=113/86, p=0.0650; paired vs `llm_only`: -0.59pp, b/c=94/101, p=0.668; paired vs `golden_plus_neighbors`: -0.42pp, b/c=94/99, p=0.773; paired vs `golden_passage`: +1.67pp, b/c=113/93, p=0.185 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `snap_hyre` | `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl` | 980/1195 = 82.0% | strict HyRE generation/retrieval-cache replay: 1195/1195 HyRE cache hits, 1195/1195 retrieval-cache hits, retrieved list length 5 on all rows, 144/1195 gold retrieved, retrieval exposure Hit@5 0.1205 / MRR@5 0.0602; post-audit found one existing generated-passage artifact at `mbe_127` where the older parser kept self-correction/choice text in `hyde_passage`; the parser and artifact detector are now hardened, but this signed row should retain that caveat unless regenerated; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 2025 tokens, max final-answer chars 8130; custom no-silent audit: exact-final-line issues 0, fallback keys 0, think tags 0, 3 valid same-model answer-format retries (`mbe_989`, `mbe_1131`, `mbe_288`), 4 rows at >=1900 output tokens with intact final `Answer:` lines (`mbe_141`, `mbe_989`, `mbe_1131`, `mbe_288`); operator stream observed transient Cloudflare 502/upstream idle retries that recovered in-place under the pinned same model/provider and did not substitute models or methods; paired vs `rag_simple`: +4.02pp, b/c=121/73, p=0.000699; paired vs `llm_only`: +1.17pp, b/c=103/89, p=0.348; paired vs `rag_hyde`: +1.76pp, b/c=84/63, p=0.0987; paired vs `golden_plus_neighbors`: +1.34pp, b/c=101/85, p=0.271; paired vs `golden_passage`: +3.43pp, b/c=118/77, p=0.00406 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP/TRANSIENT/ONE-ARTIFACT-CAVEAT |
| BarExamQA | `or-gemma4-26b` | `rag_rewrite` | `logs/eval_rag_rewrite_or-gemma4-26b_20260517_124940_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_rewrite-nfull-k5_detail.jsonl` | 964/1195 = 80.7% | dynamic rewrite retrieval, no deterministic retrieval cache; 1195/1195 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 raw-question fallbacks, 0 empty retrieval rows, 146/1195 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_rewrite.md`: Hit@5 0.1222, MRR@5 0.0604; `scripts/analyze_detail_flags.py`: rows 1195, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 2099 tokens, max final-answer chars 7618; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_rewrite`/`barexam`, exact-final-line issues 0, fallback keys 0, think tags 0, rewrite parse failures 0, partial JSON repairs 0, 4 valid same-model answer-format retries (`mbe_501`, `mbe_763`, `mbe_989`, `mbe_486`), 3 rows at >=1900 total output tokens after retry (`mbe_501`, `mbe_763`, `mbe_989`); visible Cloudflare upstream idle retries recovered in-place under the pinned same model/provider; paired vs `rag_simple`: +2.68pp, b/c=126/94, p=0.0364; paired vs `llm_only`: -0.17pp, b/c=102/104, p=0.944; paired vs `rag_hyde`: +0.42pp, b/c=88/83, p=0.760; paired vs `snap_hyre`: -1.34pp, b/c=75/91, p=0.244; paired vs `golden_plus_neighbors`: 0.00pp, b/c=98/98, p=1.000; paired vs `golden_passage`: +2.09pp, b/c=125/100, p=0.109 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP/TRANSIENT-CAVEAT |
| CaseHOLD | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260516_003945_casehold_local-snap-hyre-groq-llama70b-casehold-llm_only-nfull-k5_detail.jsonl` | 2585/3600 = 71.8% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: providers all `groq-llama70b`, exact-final-line issues 0, fallback keys 0, 39 valid same-model final-answer retries, all `missing_marker`, retry max 5 tokens, max output 774 tokens, max final-answer chars 3845, near-cap violations 0, avg calls 1.01 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| CaseHOLD | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260516_020605_casehold_local-snap-hyre-groq-llama70b-casehold-rag_simple-nfull-k5_detail.jsonl` | 2547/3600 = 70.8% | strict raw retrieval-cache replay: 3600/3600 cache hits, retrieved list length 5 on 3600/3600 rows, 0 empty evidence rows, 646/3600 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_simple.md`: Hit@5 0.1794, MRR@5 0.1015; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`casehold`, exact-final-line issues 0, fallback keys 0, 23 valid same-model final-answer retries, max output 935 tokens, max final-answer chars 4850, near-cap violations 0, avg calls 1.01; paired vs `llm_only`: -1.06pp, b/c=215/253, p=0.0871 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| CaseHOLD | `groq-llama70b` | `golden_passage` | `logs/eval_golden_passage_groq-llama70b_20260516_032635_casehold_local-snap-hyre-groq-llama70b-casehold-golden_passage-nfull-k5_detail.jsonl` | 3511/3600 = 97.5% | oracle gold injected/retrieved on 3600/3600 rows, retrieved list length 1 on 3600/3600 rows, 0 empty evidence rows; retrieval exposure from `docs/generated/retrieval_qrels_casehold_groq-llama70b_golden_passage.md`: Hit@1 1.0000, Hit@5 1.0000, MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_passage`/`casehold`, final-line prediction issues 0, fallback keys 0, 46 valid same-model final-answer retries, max output 878 tokens, max final-answer chars 4845, near-cap violations 0, avg calls 1.01; paired vs `rag_simple`: +26.78pp, b/c=968/4, p=1.86e-282; paired vs `llm_only`: +25.72pp, b/c=927/1, p=8.19e-277 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| CaseHOLD | `groq-llama70b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama70b_20260516_045222_casehold_local-snap-hyre-groq-llama70b-casehold-golden_plus_neighbors-nfull-k5_detail.jsonl` | 2859/3600 = 79.4% | strict golden-neighbor retrieval-cache replay: 3600/3600 cache hits, retrieved list length 5 and neighbor list length 4 on 3600/3600 rows, 0 empty evidence rows, 3600/3600 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_casehold_groq-llama70b_golden_plus_neighbors.md`: Hit@1 1.0000, Hit@5 1.0000, MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_plus_neighbors`/`casehold`, final-line prediction issues 0, fallback keys 0, 19 valid same-model final-answer retries, max output 943 tokens, max final-answer chars 4989, near-cap violations 0, avg calls 1.01; trace response previews clipped on 2794/3600 rows by `EVAL_TRACE_MAX_CHARS=800`, but full final answers were stored separately and exact; paired vs `rag_simple`: +8.67pp, b/c=459/147, p=2.70e-38; paired vs `llm_only`: +7.61pp, b/c=411/137, p=8.67e-33; paired vs `golden_passage`: -18.11pp, b/c=5/657, p=1.10e-187 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| CaseHOLD | `groq-llama70b` | `rag_hyde` | `logs/merged/eval_rag_hyde_llama70b_mixed_20260516_casehold_nfull_k5_detail.jsonl` | 2532/3600 = 70.3% | signed with explicit mixed same-model provider caveat: Groq produced 2639 valid rows, then stopped on spend alert; `ch_test_2639` plus 960 tail rows were replayed through OpenRouter paid `meta-llama/llama-3.3-70b-instruct`, with the final 667-row segment pinned to `OPENROUTER_PROVIDER_ONLY=AkashML`; strict generated/retrieval-cache replay, 3600/3600 HyDE cache hits and retrieval-cache hits, 0 empty evidence rows, 1844/3600 gold retrieved, retrieval exposure Hit@5 0.5122 / MRR@5 0.3983 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_hyde_mixed.md`; 21 logged answer-format retries, zero errors/missing predictions/parse failures/fallback keys/exact-final-line issues/think tags/near-cap outputs; max output 742 tokens; paired vs `rag_simple`: -0.42pp, b/c=247/262, p=0.535; paired vs `llm_only`: -1.47pp, b/c=211/264, p=0.0169; paired vs `golden_plus_neighbors`: -9.08pp, b/c=119/446, p=1.91e-45 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/MIXED-PROVIDER-CAVEAT |
| CaseHOLD | `groq-llama70b` | `snap_hyre` | `logs/merged/eval_snap_hyre_llama70b_mixed_20260516_casehold_nfull_k5_detail.jsonl` | 2538/3600 = 70.5% | signed with explicit mixed same-model provider and repair caveat: OpenRouter paid same-model prefix supplied 581 rows while Groq spend alert was active; after user reset, repaired `ch_test_581` plus rows `ch_test_582`-`ch_test_3599` ran on Groq; invalid OpenRouter answer row `ch_test_581` is excluded from the merged log. Generation cache row `ch_test_581` had a formatting-only repair from inline snap answer to standalone `Answer: (A)` before replay. Strict generated/retrieval-cache replay: 3600/3600 HyRE cache hits and retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1619/3600 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_casehold_groq-llama70b_snap_hyre_mixed.md`: Hit@5 0.4497 / MRR@5 0.3286; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider mix `or-llama70b-paid` 581 / `groq-llama70b` 3019, exact-final-line issues 0, fallback keys 0, think tags 0, 16 valid same-model answer-format retries, max output 826 tokens, max final-answer chars 4651, near-cap violations 0; paired vs `rag_simple`: -0.25pp, b/c=249/258, p=0.722; paired vs `llm_only`: -1.31pp, b/c=200/247, p=0.0295; paired vs `rag_hyde`: +0.17pp, b/c=225/219, p=0.812; paired vs `golden_plus_neighbors`: -8.92pp, b/c=118/439, p=1.98e-44 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/REPAIR/MIXED-PROVIDER-CAVEAT |
| CaseHOLD | `groq-llama70b` | `rag_rewrite` | `logs/eval_rag_rewrite_groq-llama70b_20260516_130926_casehold_local-snap-hyre-groq-llama70b-casehold-rag_rewrite-nfull-k5_detail.jsonl` | 2542/3600 = 70.6% | signed with explicit retry caveat: dynamic rewrite retrieval, 3600/3600 rewrite JSON parses, 0 rewrite-format retries, 0 partial-JSON repairs, 0 empty retrieval rows, 1623/3600 gold retrieved, retrieval exposure Hit@5 0.4508 / MRR@5 0.3319 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_rewrite.md`; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider all `groq-llama70b`, exact-final-line issues 0, fallback keys 0, think tags 0, 88 valid same-model answer-format retries, max output 899 tokens, max final-answer chars 4618, near-cap violations 0; paired vs `rag_simple`: -0.14pp, b/c=251/256, p=0.859; paired vs `snap_hyre`: +0.11pp, b/c=237/233, p=0.890; paired vs `rag_hyde`: +0.28pp, b/c=236/226, p=0.675; paired vs `llm_only`: -1.19pp, b/c=226/269, p=0.0589; paired vs `golden_plus_neighbors`: -8.81pp, b/c=134/451, p=4.62e-41 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| CaseHOLD | `or-gemma4-26b` | `llm_only` | `logs/eval_llm_only_or-gemma4-26b_20260517_200915_casehold_local-snap-hyre-or-gemma4-26b-casehold-llm_only-nfull-k5_detail.jsonl` | 2614/3600 = 72.6% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 3600, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`llm_only`/`casehold`, retrieved list length 0 and evidence-store length 0 on all rows, exact final `Answer: (X)` lines on all rows, fallback keys 0, think tags 0, 24 valid same-model answer-format retries, max output 2279 tokens, max final-answer chars 8587, 19 rows at >=1900 total output tokens with intact final answer lines, avg calls 1.01; paired vs `groq-llama70b` `llm_only`: +0.81pp, b/c=356/327, p=0.284 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| HousingQA | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260516_203552_housing_local-snap-hyre-groq-llama70b-housing-llm_only-nfull-k5_detail.jsonl` | 3067/6853 = 44.8% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`llm_only`/`housing`, retrieved list length 0 on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, answer retries 0, max output 518 tokens, max final-answer chars 2730, near-cap violations 0, avg calls 1.00 | ✅ COMPREHENSIVE-CLEAN |
| HousingQA | `groq-llama8b` | `llm_only` | `logs/eval_llm_only_groq-llama8b_20260519_082209_housing_local-snap-hyre-groq-llama8b-housing-llm_only-nfull-k5_detail.jsonl` | 3795/6853 = 55.4% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 6853, long rows 0, fallback markers 0, avg LLM calls 1.02, max cumulative output 2051 tokens, max final-answer chars 2486 at `3281`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`llm_only`/`housing`, provider route `{}` on all rows, retrieved/evidence length 0 on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, 163 valid same-model answer-format retries, all retry repairs 3 output tokens, 0 single-call near-cap rows, 0 retry-near-cap rows, 15 cumulative output rows >=1900 tokens all with two calls and intact final lines; paired vs `groq-llama70b` `llm_only`: +10.62pp, b/c=1455/727, p=1.23e-55, CI [9.30, 11.95] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| HousingQA | `groq-llama8b` | `rag_simple` | `logs/merged/eval_rag_simple_groq-llama8b_20260519_housing_nfull_k5_merged_detail.jsonl` | 3925/6853 = 57.3% | merged repair log: original prefix `logs/eval_rag_simple_groq-llama8b_20260519_102546_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` stopped after 117 rows with exit 137, repair chunk `logs/eval_rag_simple_groq-llama8b_20260519_105232_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` supplied rows 117-171 before being intentionally stopped to serialize Housing Chroma hydration, and final tail `logs/eval_rag_simple_groq-llama8b_20260519_114559_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` completed rows 172-end with the same provider/model/method and raw retrieval cache; `scripts/merge_detail_logs.py` produced 6853 unique rows, 0 empty retrieval rows, 193/6853 gold retrieved, avg LLM calls 1.011; strict raw retrieval-cache replay: 6853/6853 cache hits, retrieved/evidence length 5 on all rows; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl`: Hit@5 0.0282, MRR@5 0.0148; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max cumulative output 2051 tokens, max final-answer chars 3616 at `8366`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_simple`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, 76 valid same-model answer-format retries with max retry 3 tokens, no single-call near-cap rows and no retry-near-cap rows; paired vs `groq-llama8b` `llm_only`: +1.90pp, b/c=1477/1347, p=0.0152, CI [0.36, 3.46]; paired vs `groq-llama70b` `rag_simple`: +9.98pp, b/c=1688/1004, p=5.46e-40, CI [8.51, 11.45] | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR/RETRY-CAVEAT |
| HousingQA | `groq-llama8b` | `golden_passage` | `logs/eval_golden_passage_groq-llama8b_20260519_140614_housing_local-snap-hyre-groq-llama8b-housing-golden_passage-nfull-k5_detail.jsonl` | 4408/6853 = 64.3% | oracle gold injected/retrieved on 6853/6853 rows, evidence-store length 1 on all rows, 0 empty evidence rows, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.04, max cumulative output 2051 tokens, max final-answer chars 4357 at `5157`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_passage`/`housing`, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, 249 valid same-model answer-format retries, 0 single-call near-cap rows and 0 retry-near-cap rows; 9 cumulative output rows at >=1900 tokens all had logged final-answer repairs, including 6 `select_final_line` repairs after no parseable first-call prediction and 3 `format_existing_prediction` repairs; saved call-trace response snippets were abbreviated by `EVAL_TRACE_MAX_CHARS=1200` on 994 rows, with full response length retained in `response_chars` and no model-facing prompt truncation; paired vs `llm_only`: +8.94pp, b/c=1341/728, p=7.36e-42, CI [7.68, 10.24]; paired vs `rag_simple`: +7.05pp, b/c=1506/1023, p=7.00e-22, CI [5.60, 8.48]; paired vs `groq-llama70b` `golden_passage`: -2.96pp, b/c=727/930, p=6.75e-07, CI [-4.14, -1.81] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP/TRACE-ABBREVIATION-CAVEAT |
| HousingQA | `groq-llama8b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama8b_20260519_141204_housing_local-snap-hyre-groq-llama8b-housing-golden_plus_neighbors-nfull-k5_detail.jsonl` | 4372/6853 = 63.8% | strict golden-neighbor retrieval/doc-cache replay: 6853/6853 rows, evidence-store length 5 on all rows, 0 empty evidence rows, 6853/6853 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max cumulative output 2052 tokens, max final-answer chars 9867 at `1527`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_plus_neighbors`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, 128 valid same-model answer-format retries, 72 cumulative output rows at >=1900 tokens with intact final lines, including 1 single-call near-cap row and 71 retry-near-cap rows; saved call-trace response snippets were abbreviated by `EVAL_TRACE_MAX_CHARS=1200` on 2962 rows, with full response length retained in `response_chars` and no model-facing prompt truncation; paired vs `llm_only`: +8.42pp, b/c=1386/809, p=3.98e-35, CI [7.09, 9.75]; paired vs `rag_simple`: +6.52pp, b/c=1496/1049, p=7.76e-19, CI [5.08, 7.97]; paired vs `golden_passage`: -0.53pp, b/c=777/813, p=0.380; paired vs `groq-llama70b` `golden_plus_neighbors`: -2.22pp, b/c=776/928, p=0.000252, CI [-3.37, -1.07] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP/TRACE-ABBREVIATION-CAVEAT |
| HousingQA | `groq-llama8b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260519_204401_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl` | 3998/6853 = 58.3% | strict HyDE/retrieval/doc-cache replay: 6853/6853 HyDE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 854/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl`: Hit@5 0.1246, Recall@5 0.0828, MRR@5 0.0775, Hit@10 0.1682, MRR@10 0.0833; Housing full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=4096` for retrieval-cache construction and strict doc-cache replay; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max cumulative output 2052 tokens, max final-answer chars 4663 at `1124`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_hyde`/`housing`, retrieved/evidence length 5 on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, parse-false markers 0, 128 valid same-model answer-format retries, 92 cumulative output rows at >=1900 tokens with intact final lines; paired vs `llm_only`: +2.96pp, b/c=1361/1158, p=5.65e-05, CI [1.53, 4.38]; paired vs `rag_simple`: +1.07pp, b/c=1216/1143, p=0.138, CI [-0.29, 2.47]; paired vs `golden_passage`: -5.98pp, b/c=968/1378, p=2.55e-17, CI [-7.34, -4.71]; paired vs `groq-llama70b` `rag_hyde`: -0.53pp, b/c=1138/1174, p=0.467, CI [-1.94, 0.90] | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| HousingQA | `groq-llama8b` | `snap_hyre` | `logs/merged/eval_snap_hyre_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl` | 3937/6853 = 57.4% | merged repair log after no-silent guard blocked the original row `hqa_South Carolina_4486`: original prefix `logs/eval_snap_hyre_groq-llama8b_20260520_012038_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl` supplied rows 0-5333, and repair tail `logs/eval_snap_hyre_groq-llama8b_20260520_031420_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-repairtail-nfull-k5_detail.jsonl` supplied row 5334 through end after the same-model final-answer retry was patched to format the prior response rather than re-solve; `scripts/merge_detail_logs.py --on-duplicate last` produced 6853 unique rows, 0 empty retrieval rows, 512/6853 gold retrieved, avg LLM calls 1.015, avg tokens 2698.9; strict HyRE/retrieval/doc-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, evidence-store length 5 on all rows; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`: Hit@5 0.0747, Recall@5 0.0497, MRR@5 0.0432, Hit@10 0.1067, MRR@10 0.0474; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max cumulative output 2051 tokens, max final-answer chars 5178 at `2979`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`snap_hyre`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, truthy fallback rows 0, think tags 0, 103 valid same-model answer-format retries, 76 cumulative output rows at >=1900 tokens with intact final lines, cache misses 0, HyRE cache misses 0, doc-cache misses 0; Housing full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=4096` for retrieval-cache construction and strict doc-cache replay, with cross-encoder doc truncation on 3674 rows and no query truncation; paired vs `llm_only`: +2.07pp, b/c=1404/1262, p=0.00631, CI [0.61, 3.59]; paired vs `rag_simple`: +0.18pp, b/c=1117/1105, p=0.815, CI [-1.17, 1.53]; paired vs `rag_hyde`: -0.89pp, b/c=1072/1133, p=0.201, CI [-2.20, 0.45]; paired vs `golden_passage`: -6.87pp, b/c=1025/1496, p=6.12e-21, CI [-8.29, -5.43]; paired vs `golden_plus_neighbors`: -6.35pp, b/c=1021/1456, p=2.28e-18, CI [-7.75, -4.95] | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR/RETRY/NEAR-CAP-CAVEAT |
| HousingQA | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260518_001738_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl` | 3241/6853 = 47.3% | strict raw retrieval-cache replay: 6853/6853 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 193/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl`: Hit@5 0.0282, Recall@5 0.0192, MRR@5 0.0148; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 866 tokens, max final-answer chars 4878 at `3488`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`housing`, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, answer retries 0, near-cap outputs 0, avg calls 1.00; paired vs `llm_only`: +2.54pp, b/c=894/720, p=1.63e-05 | ✅ COMPREHENSIVE-CLEAN |
| HousingQA | `groq-llama70b` | `golden_passage` | `logs/eval_golden_passage_groq-llama70b_20260518_061249_housing_local-snap-hyre-groq-llama70b-housing-golden_passage-nfull-k5_detail.jsonl` | 4611/6853 = 67.3% | oracle gold injected/retrieved on 6853/6853 rows, evidence-store length 1 on all rows, 0 empty evidence rows, 6853/6853 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 720 tokens, max final-answer chars 3649 at `6596`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_passage`/`housing`, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, 18 valid same-model answer-format retries, near-cap outputs 0, avg calls 1.00; paired vs `rag_simple`: +19.99pp, b/c=1817/447, p=5.25e-195; paired vs `llm_only`: +22.53pp, b/c=1929/385, p=3.09e-246; paired vs `golden_plus_neighbors`: +1.27pp, b/c=540/453, p=0.00632 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| HousingQA | `groq-llama70b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama70b_20260518_033525_housing_local-snap-hyre-groq-llama70b-housing-golden_plus_neighbors-nfull-k5_detail.jsonl` | 4524/6853 = 66.0% | strict golden-neighbor retrieval-cache replay: 6853/6853 cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 6853/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_golden_neighbors_k10.jsonl`: Hit@1 1.0000, Recall@5 0.9893, Hit@5 1.0000, MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 625 tokens, max final-answer chars 3331 at `452`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_plus_neighbors`/`housing`, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, one valid same-model answer-format retry (`4192`), near-cap outputs 0, avg calls 1.00; paired vs `rag_simple`: +18.72pp, b/c=1702/419, p=1.17e-182; paired vs `llm_only`: +21.26pp, b/c=1829/372, p=2.91e-230 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| HousingQA | `groq-llama70b` | `rag_hyde` | `logs/merged/eval_rag_hyde_groq-llama70b_20260519_housing_nfull_k5_merged_detail.jsonl` | 4034/6853 = 58.9% | merged repair log: original prefix `logs/eval_rag_hyde_groq-llama70b_20260519_081625_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` stopped at 4410 rows with exit 137, repair chunk `logs/eval_rag_hyde_groq-llama70b_20260519_102827_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` stopped after 810 tail rows with exit 137 during a concurrent Housing Chroma-hydration replay, and final repair chunk `logs/eval_rag_hyde_groq-llama70b_20260519_105416_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` completed rows 5220:end with the same provider/model/method and the same HyDE/retrieval caches; `scripts/merge_detail_logs.py` produced 6853 unique rows, 0 empty retrieval rows, 1141/6853 gold retrieved, avg LLM calls 1.000; strict generated/retrieval-cache replay: 6853/6853 HyDE cache hits and 6853/6853 retrieval-cache hits, retrieved list length 5 on all rows; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl`: Hit@5 0.1665, Recall@5 0.1124, MRR@5 0.1098, Hit@10 0.2196, MRR@10 0.1169; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 766 tokens, max final-answer chars 4254 at `5689`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_hyde`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, truthy fallback rows 0, think tags 0, answer-format retries 0, near-cap rows 0; paired vs `llm_only`: +14.11pp, b/c=1725/758, p=6.47e-86; paired vs `rag_simple`: +11.57pp, b/c=1408/615, p=3.09e-71; paired vs `snap_hyre`: +6.45pp, b/c=1022/580, p=1.42e-28; paired vs `golden_passage`: -8.42pp, b/c=653/1230, p=7.87e-41; paired vs `golden_plus_neighbors`: -7.15pp, b/c=632/1122, p=6.80e-32 | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR-CHUNK-CAVEAT |
| HousingQA | `groq-llama70b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama70b_20260519_042924_housing_local-snap-hyre-groq-llama70b-housing-snap_hyre-nfull-k5_detail.jsonl` | 3592/6853 = 52.4% | strict HyRE generation/retrieval-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 611/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`: Hit@5 0.0892, Recall@5 0.0579, MRR@5 0.0530, Hit@10 0.1379, MRR@10 0.0594; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, max output 731 tokens, max final-answer chars 3797 at `7113`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`snap_hyre`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, answer retries 0, near-cap outputs 0, retrieved list length 5 on all rows, avg calls 1.00; paired vs `llm_only`: +7.66pp, b/c=1342/817, p=8.96e-30; paired vs `rag_simple`: +5.12pp, b/c=1068/717, p=9.51e-17; paired vs `golden_passage`: -14.87pp, b/c=554/1573, p=1.98e-112; paired vs `golden_plus_neighbors`: -13.60pp, b/c=531/1463, p=2.64e-100 | ✅ COMPREHENSIVE-CLEAN |
| Legal-Link-EU | `groq-llama70b` | `llm_only` | `logs/eval_llm_only_groq-llama70b_20260518_232903_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-llm_only-nfull-k5_detail.jsonl` | 781/1127 = 69.3% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 1127, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 917 tokens, max final-answer chars 4771 at `complex_legallink_32013R1420_32002R2306_repeals__row0923`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`llm_only`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 14 valid same-model answer-format retries, retry-near-cap rows 0, near-cap rows 0 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260519_000257_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-rag_simple-nfull-k5_detail.jsonl` | 849/1127 = 75.3% | strict raw retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_raw_question_ce22000_k10.jsonl`: 1127/1127 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1021/1127 gold retrieved; retrieval exposure Hit@5 0.9059 / MRR@5 0.7621, source_doc retrieved on 863/1127 rows, target_doc retrieved on 659/1127 rows; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max output 2053 tokens, max final-answer chars 4775 at `complex_legallink_32011D0740_32006D0799_extends_validity__row0823`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 19 valid same-model answer-format retries, 1 near-cap row with intact exact final line; paired vs `llm_only`: +6.03pp, b/c=187/119, p=0.0001207 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `golden_passage` | `logs/eval_golden_passage_groq-llama70b_20260519_004215_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-golden_passage-nfull-k5_detail.jsonl` | 1004/1127 = 89.1% | oracle gold injected/retrieved on 1127/1127 rows, evidence-store length 1 on all rows, 0 empty evidence rows, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.03, max output 1260 tokens, max final-answer chars 3861 at `complex_legallink_32017D0496_32011D0172_extends_application__row0993`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_passage`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 34 valid same-model answer-format retries, no near-cap rows; paired vs `rag_simple`: +13.75pp, b/c=191/36, p=1.07e-26; paired vs `llm_only`: +19.79pp, b/c=268/45, p=8.62e-40 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama70b_20260519_011422_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-golden_plus_neighbors-nfull-k5_detail.jsonl` | 993/1127 = 88.1% | strict golden-neighbor retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_golden_neighbors_k10.jsonl`: 1127/1127 cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 1127/1127 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 1010 tokens, max final-answer chars 5320 at `complex_legallink_32008R0950R(01)_32008R0950_corrects__row0714`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`golden_plus_neighbors`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 15 valid same-model answer-format retries, no near-cap rows; paired vs `rag_simple`: +12.78pp, b/c=191/47, p=8.87e-22; paired vs `llm_only`: +18.81pp, b/c=263/51, p=1.43e-35; paired vs `golden_passage`: -0.98pp, b/c=39/50, p=0.289 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260519_015353_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-rag_hyde-nfull-k5_detail.jsonl` | 817/1127 = 72.5% | strict HyDE generation/retrieval-cache replay: 1127/1127 HyDE cache hits, 1127/1127 retrieval-cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 616/1127 gold retrieved; retrieval cache audit from `caches/retrieval/full/legal_link_eu_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl`: Hit@5 0.5466, MRR@5 0.4336, Hit@10 0.6539, MRR@10 0.4475; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0, source_doc retrieved on 504/1127 rows and target_doc on 368/1127; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.03, max output 1009 tokens, max final-answer chars 4959 at `complex_legallink_32007D0457_32002D0739_extends_validity__row0595`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_hyde`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 29 valid same-model answer-format retries, no near-cap rows; paired vs `llm_only`: +3.19pp, b/c=155/119, p=0.0343; paired vs `rag_simple`: -2.84pp, b/c=116/148, p=0.0562; paired vs `golden_passage`: -16.59pp, b/c=36/223, p=4.07e-34; paired vs `golden_plus_neighbors`: -15.62pp, b/c=48/224, p=2.36e-28 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama70b_20260519_022935_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-snap_hyre-nfull-k5_detail.jsonl` | 813/1127 = 72.1% | strict HyRE generation/retrieval-cache replay: 1127/1127 HyRE cache hits, 1127/1127 retrieval-cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 817/1127 gold retrieved; retrieval cache audit from `caches/retrieval/full/legal_link_eu_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`: Hit@5 0.7249, MRR@5 0.5810, Hit@10 0.8181, MRR@10 0.5938; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0, source_doc retrieved on 665/1127 rows and target_doc on 544/1127; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.03, max output 1010 tokens, max final-answer chars 4979 at `complex_legallink_32011D0895_32002D0546_extends_validity__row0826`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`snap_hyre`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 29 valid same-model answer-format retries, no near-cap rows; paired vs `llm_only`: +2.84pp, b/c=169/137, p=0.0762; paired vs `rag_simple`: -3.19pp, b/c=104/140, p=0.0248; paired vs `rag_hyde`: -0.35pp, b/c=112/116, p=0.843; paired vs `golden_passage`: -16.95pp, b/c=36/227, p=4.59e-35; paired vs `golden_plus_neighbors`: -15.97pp, b/c=40/220, p=2.71e-31 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| Legal-Link-EU | `groq-llama70b` | `rag_rewrite` | `logs/merged/eval_rag_rewrite_groq-llama70b_20260519_legal_link_eu_nfull_k5_merged_detail.jsonl` | 811/1127 = 72.0% | merged repair log: original partial `logs/eval_rag_rewrite_groq-llama70b_20260519_030809_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-rag_rewrite-nfull-k5_detail.jsonl` stopped on a Groq spend-threshold row, and repair tail `logs/eval_rag_rewrite_groq-llama70b_20260519_032432_legal_link_eu_local-snap-hyre-groq-llama70b-legal_link_eu-rag_rewrite-nfull-k5_detail.jsonl` reran rows 211:end with same provider/model/method; `scripts/merge_detail_logs.py --on-duplicate last` produced 1127 unique rows, 0 empty retrieval rows, 742/1127 gold retrieved, avg LLM calls 2.051; dynamic rewrite retrieval, not cache replay: retrieved/evidence length 5 on all rows, rewrite JSON parse kind `json` on all rows, 0 rewrite fallbacks, 0 partial-JSON repairs, 0 row errors, provider route `{}` on all rows, source_doc retrieved on 575/1127 rows and target_doc on 467/1127; launch stdout recorded `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation count rows 0 and CE query truncation rows 0, but the merged detail row-level `cross_encoder_max_chars` field is blank because dynamic non-cache logging did not persist the env value before the 2026-05-19 patch; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, max output 1070 tokens, max final-answer chars 4821 at `complex_legallink_31972L0194_31964L0221_extends_application__row0008`; paired vs `llm_only`: +2.66pp, b/c=152/122, p=0.0796; paired vs `rag_simple`: -3.37pp, b/c=110/148, p=0.0211; paired vs `rag_hyde`: -0.53pp, b/c=124/130, p=0.754; paired vs `snap_hyre`: -0.18pp, b/c=123/125, p=0.949; paired vs `golden_passage`: -17.13pp, b/c=38/231, p=6.63e-35; paired vs `golden_plus_neighbors`: -16.15pp, b/c=50/232, p=3.44e-29 | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR/CE-MAX-LOGGING-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `llm_only` | `logs/eval_llm_only_groq-llama8b_20260519_002019_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-llm_only-nfull-k5_detail.jsonl` | 541/1127 = 48.0% | no retrieval evidence by design; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 1127, long rows 0, fallback markers 0, avg LLM calls 1.03, max output 2054 tokens, max final-answer chars 3177 at `complex_legallink_32011R0045_32010R0950_repeals__row0827`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`llm_only`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 32 valid same-model answer-format retries, 28 near-cap rows with intact exact final lines | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `rag_simple` | `logs/eval_rag_simple_groq-llama8b_20260519_003837_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-rag_simple-nfull-k5_detail.jsonl` | 647/1127 = 57.4% | strict raw retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_raw_question_ce22000_k10.jsonl`: 1127/1127 cache hits, retrieved list length 5 on all rows, 0 empty evidence rows, 1021/1127 gold retrieved; retrieval exposure Hit@5 0.9059 / MRR@5 0.7621 from the full raw retrieval cache, source_doc retrieved on 863/1127 rows, target_doc retrieved on 659/1127 rows; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.05, max output 2059 tokens, max final-answer chars 5376 at `complex_legallink_32011R1061_52014XC1121(01)_rendered_obsolete_by__row0855`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_simple`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 59 valid same-model answer-format retries, 58 near-cap rows with intact exact final lines; paired vs `llm_only`: +9.41pp, b/c=270/164, p=4.10e-07 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `golden_passage` | `logs/eval_golden_passage_groq-llama8b_20260519_010454_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-golden_passage-nfull-k5_detail.jsonl` | 808/1127 = 71.7% | oracle gold injected/retrieved on 1127/1127 rows, evidence-store length 1 on all rows, 0 empty evidence rows, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.05, max output 2086 tokens, max final-answer chars 9076 at `complex_legallink_32016D0207_32013D0233_extends_application__row0969`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_passage`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 54 valid same-model answer-format retries, 34 near-cap rows with intact exact final lines; paired vs `rag_simple`: +14.29pp, b/c=277/116, p=2.66e-16; paired vs `llm_only`: +23.69pp, b/c=349/82, p=2.77e-40 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_groq-llama8b_20260519_012359_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-golden_plus_neighbors-nfull-k5_detail.jsonl` | 797/1127 = 70.7% | strict golden-neighbor retrieval-cache replay from `caches/retrieval/full/legal_link_eu_qfull_seed42_golden_neighbors_k10.jsonl`: 1127/1127 cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 1127/1127 gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max output 2053 tokens, max final-answer chars 5661 at `complex_legallink_32022D1943_32019D1720_extends_validity__row1097`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`golden_plus_neighbors`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 21 valid same-model answer-format retries, 20 near-cap rows with intact exact final lines; paired vs `rag_simple`: +13.31pp, b/c=263/113, p=6.75e-15; paired vs `llm_only`: +22.72pp, b/c=358/102, p=2.11e-34; paired vs `golden_passage`: -0.98pp, b/c=137/148, p=0.554 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `snap_hyre` | `logs/merged/eval_snap_hyre_groq-llama8b_20260519_legal_link_eu_nfull_k5_merged_detail.jsonl` | 586/1127 = 52.0% | merged after explicit Groq TPM tail rerun: first 578 valid rows from `logs/eval_snap_hyre_groq-llama8b_20260519_014228_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-snap_hyre-nfull-k5_detail.jsonl` plus rows 578-1126 from paced tail `logs/eval_snap_hyre_groq-llama8b_20260519_020721_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-snap_hyre-nfull-k5_detail.jsonl`; the failed TPM row is excluded. Strict HyRE/retrieval-cache replay: 1127/1127 HyRE cache hits, 1127/1127 retrieval-cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 423/1127 gold retrieved; signed generation/retrieval cache exposure Hit@5 0.3753 / MRR@5 0.2787, Hit@10 0.4552 / MRR@10 0.2890; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0, source_doc retrieved on 335/1127 rows and target_doc on 241/1127; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.09, max output 2056 tokens, max final-answer chars 6219 at `complex_legallink_32018D0333_32014D0119_extends_application__row1037`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`snap_hyre`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 103 valid same-model answer-format retries, 102 near-cap rows with intact exact final lines; paired vs `llm_only`: +3.99pp, b/c=239/194, p=0.0344; paired vs `rag_simple`: -5.41pp, b/c=183/244, p=0.00364; paired vs `golden_passage`: -19.70pp, b/c=112/334, p=1.04e-26; paired vs `golden_plus_neighbors`: -18.72pp, b/c=125/336, p=2.23e-23 | ⚠️ COMPREHENSIVE-CITE-WITH-TPM-RERUN/RETRY/NEAR-CAP-CAVEAT |
| Legal-Link-EU | `groq-llama8b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260519_022403_legal_link_eu_local-snap-hyre-groq-llama8b-legal_link_eu-rag_hyde-nfull-k5_detail.jsonl` | 603/1127 = 53.5% | strict HyDE generation/retrieval-cache replay: 1127/1127 HyDE cache hits, 1127/1127 retrieval-cache hits, evidence-store length 5 on all rows, 0 empty evidence rows, 536/1127 gold retrieved; retrieval cache audit from `caches/retrieval/full/legal_link_eu_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl`: Hit@5 0.4756, MRR@5 0.3690, Hit@10 0.5723, MRR@10 0.3819; Legal-Link full cache contract preserved with `CROSS_ENCODER_MAX_CHARS=22000`, CE doc truncation 0, CE query truncation 0, source_doc retrieved on 434/1127 rows and target_doc on 285/1127; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.08, max output 2131 tokens, max final-answer chars 4994 at `complex_legallink_32005R1795_31998R2366_article_12_BI_completes__row0517`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_hyde`/`legal_link_eu`, provider route `{}` on all rows, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 94 valid same-model answer-format retries, 93 rows at >=1900 total output tokens with intact exact final lines and retry-near-cap rows 0; paired vs `llm_only`: +5.50pp, b/c=233/171, p=0.00237; paired vs `rag_simple`: -3.90pp, b/c=173/217, p=0.0293; paired vs `snap_hyre`: +1.51pp, b/c=193/176, p=0.405; paired vs `golden_passage`: -18.19pp, b/c=104/309, p=1.06e-24; paired vs `golden_plus_neighbors`: -17.21pp, b/c=127/321, p=1.90e-20 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/NEAR-CAP-CAVEAT |
| LegalBench-SCALR | `or-gemma4-26b` | `llm_only` | `logs/eval_llm_only_or-gemma4-26b_20260515_0056_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-llm_only-nfull-k5_merged_detail.jsonl` | 417/571 = 73.0% | merged clean from rows 0-420, repaired row 421, and rows 422-570; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: missing markers 0, near-cap violations 0 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `or-ministral-8b` | `llm_only` | `logs/eval_llm_only_or-ministral-8b_20260514_195855_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` | 384/571 = 67.3% | `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: missing markers 0, retry rows 0, near-cap violations 0 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `or-ministral-8b` | `rag_simple` | `logs/eval_rag_simple_or-ministral-8b_20260515_093406_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | 388/571 = 68.0% | strict raw retrieval-cache replay: 571/571 cache hits, 0 empty evidence rows, 283/571 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_simple.md`: Hit@5 0.4956, MRR@5 0.3447; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, 16 valid same-model final-answer retries, retry max 5 tokens, 11 original responses at >=2000 output tokens before repair, near-cap violations 0; paired vs `llm_only`: +0.70pp, b/c=47/43, p=0.752 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-ministral-8b` | `golden_passage` | `logs/eval_golden_passage_or-ministral-8b_20260515_102620_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | 532/571 = 93.2% | oracle gold injected on 571/571 rows with 571/571 evidence rows; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, 22 valid same-model final-answer retries, retry max 5 tokens, four original responses at >=2000 output tokens before repair, near-cap violations 0; paired vs `rag_simple`: +25.22pp, b/c=145/1, p=3.30e-42; paired vs `llm_only`: +25.92pp, b/c=151/3, p=5.33e-41 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-ministral-8b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-ministral-8b_20260515_112849_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | 440/571 = 77.1% | strict golden-neighbor retrieval-cache replay: 571/571 cache hits, 571/571 gold retrieved, 571/571 evidence rows; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, 5 valid same-model final-answer retries, retry max 5 tokens, three original responses at >=2000 output tokens before repair, near-cap violations 0; paired vs `rag_simple`: +9.11pp, b/c=72/20, p=4.61e-08; paired vs `golden_passage`: -16.11pp, b/c=4/96, p=6.45e-24; paired vs `llm_only`: +9.81pp, b/c=78/22, p=1.59e-08 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-ministral-8b` | `rag_hyde` | `logs/eval_rag_hyde_or-ministral-8b_20260515_130224_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | 406/571 = 71.1% | strict generated/retrieval-cache replay: 571/571 HyDE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 344/571 gold retrieved; retrieval exposure from signed cache: Hit@5 0.6025, MRR@10 0.4506; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, 18 valid same-model answer-format retries, retry max 5 tokens, 14 original responses at >=2000 output tokens before repair, near-cap violations 0; paired vs `rag_simple`: +3.15pp, b/c=43/25, p=0.0385; paired vs `llm_only`: +3.85pp, b/c=54/32, p=0.0230 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-ministral-8b` | `snap_hyre` | `logs/eval_snap_hyre_or-ministral-8b_20260515_140203_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | 399/571 = 69.9% | strict generated/retrieval-cache replay: 571/571 HyRE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 354/571 gold retrieved; retrieval exposure from signed cache: Hit@5 0.6200, MRR@10 0.5110; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, 9 valid same-model answer-format retries, retry max 5 tokens, 7 original responses at >=2000 output tokens before repair, near-cap violations 0, targeted fallback-key audit 0; one naive fallback-string hit was incidental legal text in a generated CERCLA query, not a provider/method fallback; paired vs `rag_simple`: +1.93pp, b/c=45/34, p=0.260; paired vs `rag_hyde`: -1.23pp, b/c=29/36, p=0.457; paired vs `llm_only`: +2.63pp, b/c=46/31, p=0.110; paired vs `golden_plus_neighbors`: -7.18pp, b/c=20/61, p=5.66e-06 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-ministral-8b` | `rag_rewrite` | `logs/eval_rag_rewrite_or-ministral-8b_20260515_145614_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_rewrite-nfull-k5_merged_detail.jsonl` | 399/571 = 69.9% | dynamic rewrite retrieval, no deterministic retrieval cache; merged from clean non-overlapping slices after superseding no-silent failures on `scalr_110`, `scalr_431`, and `scalr_538`; 571/571 rows, 0 empty retrieval rows, 371/571 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_scalr_or-ministral-8b_rag_rewrite.md`: Hit@5 0.6497, MRR@5 0.5185; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: exact-final-line issues 0, rewrite parse false 0, fallback keys 0, 5 valid same-model rewrite-format retries, 1 logged partial-JSON repair on `scalr_538`, 31 valid answer-format retries, 27 original responses at >=2000 output tokens before repair; paired vs `rag_simple`: +1.93pp, b/c=40/29, p=0.228; paired vs `snap_hyre`: tied, b/c=33/33, p=1.000; paired vs `rag_hyde`: -1.23pp, b/c=31/38, p=0.470; paired vs `llm_only`: +2.63pp, b/c=48/33, p=0.119; paired vs `golden_plus_neighbors`: -7.18pp, b/c=20/61, p=5.66e-06 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/REPAIR-CAVEAT |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_simple` | `logs/eval_rag_simple_or-gemma4-26b_20260514_204706_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl` | 419/571 = 73.4% | strict raw retrieval-cache replay: 571/571 cache hits, 0 empty evidence rows, 283/571 gold retrieved; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: missing markers 0, 12 format-only retries, retry max 5 tokens, near-cap violations 0 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `or-gemma4-26b` | `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260514_221537_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-golden_plus_neighbors-nfull-k5_detail.jsonl` | 464/571 = 81.3% | strict golden-neighbor retrieval-cache replay: 571/571 cache hits, 571/571 gold retrieved; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: missing markers 0, 7 retries, retry max 5 tokens, near-cap violations 0; paired vs `rag_simple`: 58 fixes, 13 regressions, exact McNemar p=6.27e-08 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `or-gemma4-26b` | `golden_passage` | `logs/eval_golden_passage_or-gemma4-26b_20260514_235919_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-golden_passage-nfull-k5_detail.jsonl` | 559/571 = 97.9% | oracle gold injected on 571/571 rows; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; custom no-silent audit: missing markers 0, 1 format-only retry, retry max 5 tokens, near-cap violations 0; paired vs `golden_plus_neighbors`: plus-neighbor fixes 1, regressions 96, exact McNemar p=1.24e-27 | ✅ COMPREHENSIVE-CLEAN |
| LegalBench-SCALR | `or-gemma4-26b` | `snap_hyre` | `logs/eval_snap_hyre_or-gemma4-26b_20260515_020525_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-snap_hyre-nfull-k5_detail.jsonl` | 422/571 = 73.9% | strict generated/retrieval-cache replay: 571/571 HyRE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 415/571 gold retrieved; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; retry caveat: 10 logged same-model answer-format retries, 9 preserving existing predictions and 1 retry after an empty first response, with five original responses at >=2000 output tokens and max 2046/2048 before final-line repair; paired vs `rag_simple`: +0.53pp, b/c=27/24, p=0.780; paired vs `llm_only`: +0.88pp, b/c=26/21, p=0.560; paired vs `golden_plus_neighbors`: -7.36pp, b/c=9/51, p=3.09e-08 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260515_042731_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_hyde-nfull-k5_detail.jsonl` | 412/571 = 72.2% | strict generated/retrieval-cache replay: 571/571 HyDE cache hits, 571/571 retrieval-cache hits, 0 empty evidence rows, 404/571 gold retrieved; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; retry caveat: 8 logged same-model answer-format retries, all preserving existing predictions, with three original responses at >=2000 output tokens and max 2047/2048 before final-line repair; paired vs `rag_simple`: -1.23pp, b/c=30/37, p=0.464; paired vs `snap_hyre`: -1.75pp, b/c=22/32, p=0.220; paired vs `llm_only`: -0.88pp, b/c=25/30, p=0.590 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |
| LegalBench-SCALR | `or-gemma4-26b` | `rag_rewrite` | `logs/eval_rag_rewrite_or-gemma4-26b_20260515_062250_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-rag_rewrite-nfull-k5_detail.jsonl` | 422/571 = 73.9% | dynamic rewrite retrieval, no deterministic retrieval cache; 571/571 rewrite JSON parses, 0 rewrite retries/fallbacks, 0 empty evidence rows, 385/571 gold retrieved; retrieval exposure from `docs/generated/retrieval_qrels_scalr_or-gemma4-26b_rag_rewrite.md`: Hit@5 0.6743, MRR@5 0.5212; `scripts/analyze_detail_flags.py`: rows 571, errors 0, missing predictions 0, parse failures 0, long rows 0, fallback markers 0; retry caveat: 9 logged same-model answer-format retries, all valid format-existing-prediction repairs with max retry 5 tokens and max pre-repair answer 9105 chars; paired vs `rag_simple`: +0.53pp, b/c=34/31, p=0.804; paired vs `snap_hyre`: 0.00pp, b/c=30/30, p=1.000; paired vs `rag_hyde`: +1.75pp, b/c=40/30, p=0.282; paired vs `llm_only`: +0.88pp, b/c=42/37, p=0.653 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY-CAVEAT |

Blocked comprehensive rows under the fixed 2048-token launch contract:

No current blocked SCALR `llm_only` comprehensive row. The initial
`or-ministral-8b` blocker was superseded by the clean full rerun listed above.

Superseded comprehensive row: the initial `or-ministral-8b` SCALR `rag_simple`
run at
`logs/eval_rag_simple_or-ministral-8b_20260515_082923_legalbench_scalr_local-snap-hyre-or-ministral-8b-legalbench_scalr-rag_simple-nfull-k5_detail.jsonl`
is rejected for citation. A stricter exact-final-line audit found six
source-safety failures: three stored predictions disagreed with the exact final
line and three rows lacked an exact final line after verbose/truncated output.
Use the 2026-05-15T09:34:06 strict rerun listed in the full-answer table.

Superseded blocker: the initial `or-gemma4-26b` SCALR `llm_only` partial at
`logs/eval_llm_only_or-gemma4-26b_20260514_141502_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl`
is no longer the cite gate. It was replaced by the merged clean full detail log
listed above after the retry path was tightened to format existing predictions
instead of asking the same model to answer the whole question again.

## Update 2026-05-13 ~local WSL validation gate

Change reason: local WSL setup was validated for the Snap-HyRE comprehensive
path before broader full-corpus answer sweeps. These rows are N=50 validation
evidence, not final full-corpus claims.

Environment/source artifacts:

- Commit: `4e6236d` plus local runner/reporting hardening in this working tree.
- Collections: `legal_passages` patched to 856,835 docs, `housing_statutes` 1,837,403 docs,
  `casehold_holdings` rebuilt cleanly to 51,296 docs, and
  `legalbench_scalr_holdings` 1,733 docs.
- Retrieval stack: `Alibaba-NLP/gte-large-en-v1.5`, 1024 dimensions,
  `EMBEDDING_MAX_SEQ_LENGTH=512`; `cross-encoder/ms-marco-MiniLM-L-6-v2`
  cached locally and enabled for cache builds.
- Package artifacts:
  `docs/generated/retrieval_cache_matrix.md`,
  `docs/generated/retrieval_cache_matrix_gemma4-26b_generated.md`, and
  `docs/generated/snap_hyre_package/package_status.md`.

Provider smoke:

| Provider | Status | Evidence | Sign-off |
|---|---|---|---|
| `or-gemma3n-e4b` | `rag_simple` and `snap_hyre` N=2 smoke passed on LegalBench-SCALR | `logs/eval_rag_simple_or-gemma3n-e4b_20260512_2349_legalbench_scalr_local-api-smoke-or-gemma3n-e4b-rag_simple-n2-k3_detail.jsonl`; `logs/eval_snap_hyre_or-gemma3n-e4b_20260512_2350_legalbench_scalr_local-api-smoke-or-gemma3n-e4b-snap_hyre-n2-k3_detail.jsonl` | ✅ SMOKE-CLEAN |
| `or-gemma4-26b` | `rag_simple` and `snap_hyre` N=2 smoke passed on LegalBench-SCALR | `logs/eval_rag_simple_or-gemma4-26b_20260512_2350_legalbench_scalr_local-api-smoke-or-gemma4-26b-rag_simple-n2-k3_detail.jsonl`; `logs/eval_snap_hyre_or-gemma4-26b_20260512_2352_legalbench_scalr_local-api-smoke-or-gemma4-26b-snap_hyre-n2-k3_detail.jsonl` | ✅ SMOKE-CLEAN |
| `groq-llama70b` | blocked by Groq 401 invalid API key before detail logs | `scripts/local/run_api_smoke.sh` stdout from 2026-05-13T04:52Z | ⛔ BLOCKED |
| `or-llama70b` | fallback preflight reachable but answer calls hit upstream OpenRouter 429 rate limits | `logs/eval_rag_simple_or-llama70b_20260512_2357_legalbench_scalr_local-api-smoke-or-llama70b-rag_simple-n2-k3_detail.jsonl` is failed/do-not-use | ⛔ BLOCKED |

Retrieval-cache/qrel gate:

| Dataset | Cache evidence | Qrel alignment | Sign-off |
|---|---|---:|---|
| BarExamQA | `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`; `caches/retrieval/full/barexam_qfull_seed42_golden_neighbors_k10.jsonl` | 1149/1149 unique gold ids found = 100.00% after appending validation/test passages to `legal_passages` | ✅ RETRIEVAL-CACHE-CLEAN |
| HousingQA | `caches/retrieval/full/housing_raw_question_k10.jsonl` | 990/990 unique gold ids found = 100.00% | ✅ RETRIEVAL-CACHE-CLEAN |
| CaseHOLD | `caches/retrieval/full/casehold_raw_question_k10.jsonl`; collection rebuilt from scratch before cache | 3595/3595 unique gold ids found = 100.00% | ✅ RETRIEVAL-CACHE-CLEAN |
| LegalBench-SCALR | `caches/retrieval/full/legalbench_scalr_raw_question_k10.jsonl` | 571/571 unique gold ids found = 100.00% | ✅ RETRIEVAL-CACHE-CLEAN |

LegalBench-SCALR × `or-gemma4-26b` N=50 validation answer ladder at `k=5`:

| Mode | Detail log | Accuracy | Health | Sign-off |
|---|---|---:|---|---|
| `llm_only` | `logs/eval_llm_only_or-gemma4-26b_20260513_0138_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-llm_only-n50-k5_detail.jsonl` | 38/50 = 76.0% | `analyze_detail_flags.py`: 0 errors, 0 missing predictions, 0 parse failures, 0 long rows | ✅ VALIDATION-CLEAN |
| `rag_simple` | `logs/eval_rag_simple_or-gemma4-26b_20260513_0149_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-rag_simple-n50-k5_detail.jsonl` | 38/50 = 76.0% | clean; cached retrieval; empty retrieval 0 | ✅ VALIDATION-CLEAN |
| `rag_rewrite` | `logs/eval_rag_rewrite_or-gemma4-26b_20260513_0204_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-rag_rewrite-n50-k5_detail.jsonl` | 37/50 = 74.0% | clean; 2.00 answer-run calls | ✅ VALIDATION-CLEAN |
| `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260513_0215_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-rag_hyde-n50-k5_detail.jsonl` | 39/50 = 78.0% | clean; generation/retrieval replay cache used; answer log records final calls only | ✅ VALIDATION-CLEAN |
| `snap_hyre` | `logs/eval_snap_hyre_or-gemma4-26b_20260513_0225_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-snap_hyre-n50-k5_detail.jsonl` | 40/50 = 80.0% | clean; generation/retrieval replay cache used; answer log records final calls only | ✅ VALIDATION-CLEAN |
| `golden_passage` | `logs/eval_golden_passage_or-gemma4-26b_20260513_0237_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-golden_passage-n50-k5_detail.jsonl` | 37/50 = 74.0% | clean detail log, but SCALR rows lack `gold_passage` text and the mode fell back to direct answering | ⚠️ NON-ORACLE/FALLBACK |
| `golden_plus_neighbors` | `logs/eval_golden_plus_neighbors_or-gemma4-26b_20260513_0300_legalbench_scalr_local-snap-hyre-gemma4-26b-legalbench_scalr-golden_plus_neighbors-n50-k5_detail.jsonl` | 39/50 = 78.0% | summary-guard tagged `_FAILED-EMPTY-RETRIEVAL`; no golden-neighbor cache for SCALR | ⛔ REJECT AS ORACLE CONTROL |

Generated-query retrieval cache on the same SCALR N=50 slice:

| Method | Cache | Hit@5 | Hit@10 | MRR@10 | Sign-off |
|---|---|---:|---:|---:|---|
| `rag_hyde` | `caches/retrieval/full/legalbench_scalr_gemma4-26b_rag_hyde_k10.jsonl` | 74.0% | 84.0% | 0.6565 | ✅ RETRIEVAL-CACHE-CLEAN |
| `snap_hyre` | `caches/retrieval/full/legalbench_scalr_gemma4-26b_snap_hyre_k10.jsonl` | 68.0% | 82.0% | 0.6017 | ✅ RETRIEVAL-CACHE-CLEAN |

Validation interpretation:

- The local WSL stack is ready for deliberate one-cell-at-a-time expansion with
  MiniLM reranking enabled.
- Keep `RETRIEVAL_K=5` as the provisional answer-sweep value until generated
  caches and downstream rows exist across more datasets; raw retrieval exposure
  generally rises through `k=10`, but the current validation run used the
  predeclared `k=5` gate.
- The earlier N=50 SCALR `golden_passage` / `golden_plus_neighbors` rows remain
  rejected as oracle controls because they were run before gold-reference
  hydration was fixed.
- Do not launch `groq-llama70b` rows until the Groq key is replaced or another
  non-rate-limited Llama 70B provider is configured.

Cache/alignment hardening follow-up on 2026-05-13:

- `utils/augment_barexam_collection.py` appended the 170,511 missing
  validation/test BarExam passages to `legal_passages` with the same
  `Alibaba-NLP/gte-large-en-v1.5` encoder and `max_seq_length=512`; final
  count is 856,835 docs.
- Qrel alignment is now 100% on all four comprehensive corpora:
  BarExamQA 1149/1149, HousingQA 990/990, CaseHOLD 3595/3595, and
  LegalBench-SCALR 571/571 unique gold ids found.
- Scoped retrieval cache names now include question scope, e.g.
  `barexam_qfull_seed42_raw_question_k10.jsonl` and
  `legalbench_scalr_q5_seed42_gemma4-26b_snap_hyre_k10.jsonl`, so N=50 caches
  cannot be mistaken for full-corpus caches.
- `scripts/smoke_retrieval_cache_hydration.py` verified strict replay on all
  rows of the two BarExam full caches and on q5 raw/golden caches for all four
  datasets.
- SCALR q5 `or-gemma4-26b` generated-cache and answer-ladder smoke completed
  with strict cache requirements: all seven ladder modes were 5/5 with zero
  errors, missing predictions, parse failures, and empty retrieval rows for
  retrieval modes. This is a cache-path smoke only, not a result claim.

## Update 2026-05-13 ~strict API grid and q100 top-k preflight

Change reason: provider configuration was hardened to prevent silent fallback,
the active three-model API grid was revalidated, q20 oracle controls were
rerun after golden-reference hydration, and q100 retrieval exposure matrices
were built before launching long answer sweeps.

Provider grid and strictness:

| Provider label | Upstream model | Evidence | Sign-off |
|---|---|---|---|
| `or-ministral-8b` | `mistralai/ministral-8b-2512` | `logs/eval_llm_only_or-ministral-8b_20260513_1524_legalbench_scalr_local-api-smoke-or-ministral-8b-llm_only-n1-k3_detail.jsonl` | ✅ SMOKE-CLEAN |
| `or-gemma4-26b` | `google/gemma-4-26b-a4b-it` | `logs/eval_llm_only_or-gemma4-26b_20260513_1525_legalbench_scalr_local-api-smoke-or-gemma4-26b-llm_only-n1-k3_detail.jsonl` | ✅ SMOKE-CLEAN |
| `groq-llama70b` | `llama-3.3-70b-versatile` | `logs/eval_llm_only_groq-llama70b_20260513_1525_legalbench_scalr_local-api-smoke-groq-llama70b-llm_only-n1-k3_detail.jsonl` | ✅ SMOKE-CLEAN |

Implementation gate:

- `NO_SILENT_FALLBACK=1` is required by the local cache/answer runners.
- Unknown `LLM_PROVIDER` values fail closed, and OpenRouter requests are sent
  with provider fallback disabled.
- OpenRouter local runners now pace calls with
  `LLM_CALL_MIN_INTERVAL_SEC=2.0` and
  `LLM_CALL_RATE_LIMIT_COOLDOWN_SEC=8.0` by default. This is an operational
  guard only; it does not alter prompts, retrieval, token caps, or model labels.
- Historical `or-gemma3n-e4b` smoke rows are no longer part of the active grid.
  Historical local-vLLM small-Gemma work used `google/gemma-4-E4B-it`, not
  Gemma 3n.

LegalBench-SCALR N=20 golden-reference control rerun at `k=5`:

| Provider | `golden_passage` | `golden_plus_neighbors` | Health | Sign-off |
|---|---:|---:|---|---|
| `or-ministral-8b` | 17/20 = 85.0% | 18/20 = 90.0% | zero errors/fallbacks/parse failures; oracle evidence present | ✅ CONTROL-CLEAN |
| `or-gemma4-26b` | 20/20 = 100.0% | 18/20 = 90.0% | zero errors/fallbacks/parse failures; oracle evidence present | ✅ CONTROL-CLEAN |
| `groq-llama70b` | 20/20 = 100.0% | 18/20 = 90.0% | zero errors/fallbacks/parse failures; oracle evidence present | ✅ CONTROL-CLEAN |

Interpretation: golden-plus-neighbors is not universally better than
gold-only on this q20 SCALR control; it helped the small model by +1 row and
hurt both stronger models by -2 rows. Use it as a diagnostic control, not as a
settled default.

Q100 retrieval exposure, raw question cache:

| Source | Macro Hit@5 | Macro Hit@10 | Macro MRR@5 | Macro MRR@10 | Sign-off |
|---|---:|---:|---:|---:|---|
| `docs/generated/retrieval_cache_matrix_q100_raw.md` | 0.1775 | 0.2150 | 0.1135 | 0.1184 | ✅ CACHE-CLEAN |

Q100 retrieval exposure, `or-gemma4-26b` generated caches:

| Method | Macro Hit@5 | Macro Hit@10 | Macro MRR@5 | Macro MRR@10 | Source | Sign-off |
|---|---:|---:|---:|---:|---|---|
| `rag_hyde` | 0.4050 | 0.4575 | 0.3078 | 0.3152 | `docs/generated/retrieval_cache_matrix_or-gemma4-26b_generated.md` | ✅ CACHE-CLEAN |
| `snap_hyre` | 0.3700 | 0.4125 | 0.2956 | 0.3010 | `docs/generated/retrieval_cache_matrix_or-gemma4-26b_generated.md` | ✅ CACHE-CLEAN |

Per-dataset caveat: generated queries are not uniformly better by method. On
HousingQA q100, `rag_hyde` Hit@10 is 0.2100 while `snap_hyre` Hit@10 is
0.0500; on LegalBench-SCALR q100, `snap_hyre` Hit@10 is 0.7800 and
`rag_hyde` Hit@10 is 0.7600. Do not promote a universal method claim from
retrieval exposure alone.

Downstream answer status:

| Dataset | Provider | Mode | k | Accuracy | Detail log | Sign-off |
|---|---|---|---:|---:|---|---|
| BarExamQA q100 | `or-gemma4-26b` | `llm_only` | 5 | 86/100 = 86.0% | `logs/eval_llm_only_or-gemma4-26b_20260513_1707_barexam_local-snap-hyre-or-gemma4-26b-barexam-llm_only-n100-k5_detail.jsonl` | ✅ BASELINE-CLEAN |

The attempted broad q100 answer sweep was stopped immediately after this
baseline because the baseline alone took 1278 seconds at 12.8 seconds/query.
No q100 downstream top-k winner has been signed off yet. The next clean gate is
a deliberate `or-gemma4-26b` answer sweep comparing `k=5` vs `k=10` with
strict replay caches, scoped narrowly enough that it is not confused with the
full comprehensive run.

## Update 2026-05-14 ~top-k prelaunch gate and truncation guard

Change reason: avoid over-tuning top-k before the comprehensive grid while
still answering whether k=10 looks worth promoting over the predeclared k=5
default.

Source artifacts:

- `docs/top_k_prelaunch_probe_2026-05-14.md`
- `docs/generated/retrieval_cache_matrix_or-gemma4-26b_q100_k1_to_k10.md`
- `docs/generated/retrieval_cache_matrix_or-gemma4-26b_q100_k1_to_k10.csv`

Current provider callability smoke, LegalBench-SCALR q1 `llm_only`:

| Provider label | Upstream model | Detail log | Health | Sign-off |
|---|---|---|---|---|
| `or-ministral-8b` | `mistralai/ministral-8b-2512` | `logs/eval_llm_only_or-ministral-8b_20260514_1340_legalbench_scalr_local-api-smoke-or-ministral-8b-llm_only-n1-k3_detail.jsonl` | 1/1, zero errors, missing predictions, parse failures, long rows | ✅ SMOKE-CLEAN |
| `or-gemma4-26b` | `google/gemma-4-26b-a4b-it` | `logs/eval_llm_only_or-gemma4-26b_20260514_1342_legalbench_scalr_local-api-smoke-or-gemma4-26b-llm_only-n1-k3_detail.jsonl` | 1/1, zero errors, missing predictions, parse failures, long rows | ✅ SMOKE-CLEAN |
| `groq-llama70b` | `llama-3.3-70b-versatile` | `logs/eval_llm_only_groq-llama70b_20260514_1342_legalbench_scalr_local-api-smoke-groq-llama70b-llm_only-n1-k3_detail.jsonl` | 1/1, zero errors, missing predictions, parse failures, long rows | ✅ SMOKE-CLEAN |

Fine-grained q100 retrieval exposure, macro over `rag_simple`, `rag_hyde`, and
`snap_hyre` caches for `or-gemma4-26b`:

| k | Macro Hit@k | Macro MRR@k | Sign-off |
|---:|---:|---:|---|
| 5 | 0.3175 | 0.2389 | ✅ CACHE-CLEAN |
| 6 | 0.3300 | 0.2410 | ✅ CACHE-CLEAN |
| 7 | 0.3367 | 0.2420 | ✅ CACHE-CLEAN |
| 8 | 0.3508 | 0.2438 | ✅ CACHE-CLEAN |
| 9 | 0.3542 | 0.2441 | ✅ CACHE-CLEAN |
| 10 | 0.3617 | 0.2449 | ✅ CACHE-CLEAN |

Interpretation: k=10 improves retrieval exposure, but most of the post-k5
gain is recall rather than earlier-rank evidence. Use the k=1..10 retrieval
curve for analysis/figures; do not treat it as a reason to delay the answer
grid.

Default-model BarExamQA q100 downstream k=5 vs k=10:

| Provider | Mode | k | Accuracy | Detail log | Health | Sign-off |
|---|---|---:|---:|---|---|---|
| `or-gemma4-26b` | `rag_simple` | 5 | 83/100 = 83.0% | `logs/merged/barexam_or-gemma4-26b_rag_simple_q100_k5_20260514_detail.jsonl` | zero errors, missing predictions, parse failures, empty retrieval rows, long rows | ✅ TOPK-GATE-CLEAN |
| `or-gemma4-26b` | `rag_simple` | 10 | 81/100 = 81.0% | `logs/merged/barexam_or-gemma4-26b_rag_simple_q100_k10_20260514_detail.jsonl` | zero errors, missing predictions, parse failures, empty retrieval rows, long rows | ✅ TOPK-GATE-CLEAN |
| `or-gemma4-26b` | `rag_hyde` | 5 | 87/100 = 87.0% | `logs/merged/barexam_or-gemma4-26b_rag_hyde_q100_k5_20260514_detail.jsonl` | zero errors, missing predictions, parse failures, empty retrieval rows, long rows; one same-model answer-format retry | ✅ TOPK-GATE-CLEAN |
| `or-gemma4-26b` | `rag_hyde` | 10 | 84/100 = 84.0% | `logs/merged/barexam_or-gemma4-26b_rag_hyde_q100_k10_20260514_detail.jsonl` | zero errors, missing predictions, parse failures, empty retrieval rows, long rows | ✅ TOPK-GATE-CLEAN |

Launch decision: keep `RETRIEVAL_K=5` as the shared comprehensive answer
setting. Do not run a broader per-model/per-k downstream search before the main
grid unless a later result specifically motivates it.

Implementation guardrail:

- `scripts/local/run_answer_cell.sh` now defaults answer cells to
  `LLM_MAX_COMPLETION_TOKENS=2048`.
- Explicit environment overrides still win.
- The runner fails closed if `LLM_MAX_COMPLETION_TOKENS` is below
  `EVAL_MIN_COMPLETION_TOKENS` (default 2048), preventing stale `.env` values
  such as 768 from silently creating truncation-prone answer runs.
- Local and HPC answer runners default `EVAL_FINAL_FORMAT_RETRY=1`; format
  retries use the same model and same evidence and are logged per row.
- Local and HPC answer runners require `NO_SILENT_FALLBACK` to be truthy and
  fail before launch if it is disabled.

First full comprehensive row from the 2026-05-14 launch pass:

| Dataset | Provider | Mode | k | Accuracy | Detail log | Health | Sign-off |
|---|---|---|---:|---:|---|---|---|
| LegalBench-SCALR full | `groq-llama70b` | `llm_only` | 5 | 425/571 = 74.4% | `logs/eval_llm_only_groq-llama70b_20260514_1414_legalbench_scalr_local-snap-hyre-groq-llama70b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` | zero errors, missing predictions, parse failures, long rows, fallback markers, and near-cap outputs; `analyze_detail_flags.py` PASS | ✅ COMPREHENSIVE-CLEAN |

Blocked under the current fixed launch contract:

| Dataset | Provider | Mode | Blocker | Evidence | Sign-off |
|---|---|---|---|---|---|
| LegalBench-SCALR full | `or-ministral-8b` | `llm_only` | row `qa_unknown_scalr_7` reached 2046 output tokens under `LLM_MAX_COMPLETION_TOKENS=2048`, tripping the no-silent-fallback near-cap guard before a complete row could be promoted | local run started 2026-05-14T19:02Z with tag `local-snap-hyre-or-ministral-8b-legalbench_scalr-llm_only-nfull-k5` | ⛔ BLOCKED-PENDING-CAP-OR-PROMPT-DECISION |
| LegalBench-SCALR full | `or-gemma4-26b` | `llm_only` | row `qa_unknown_scalr_10` reached 2043 output tokens under `LLM_MAX_COMPLETION_TOKENS=2048` and missed the required final `Answer:` marker | `logs/eval_llm_only_or-gemma4-26b_20260514_141502_legalbench_scalr_local-snap-hyre-or-gemma4-26b-legalbench_scalr-llm_only-nfull-k5_detail.jsonl` partial 11-row strict-guard log | ⛔ BLOCKED-PENDING-CAP-OR-PROMPT-DECISION |

## Update 2026-05-11 ~meeting package

Change reason: consolidated the May 11 diagnostic-adaptation meeting package,
validated the repaired CaseHOLD direct option-table held-out run, and generated
source-gated figures for the inherited ablation/controller story.

Last updated: 2026-05-11
Branch: `codex/final-report-snap-hyde`

### Delta since 2026-05-10 diagnostic controller package

1. ✅ **Meeting package ready for the diagnostic-adaptation frame**:
   `docs/archive/diagnostic_adaptation_2026-05-12/meeting_prep_2026-05-11_diagnostic_adaptation.md` consolidates the
   four legal benchmarks, inherited calibration/held-out ablation tables,
   bottleneck summary, controller narrative, and generated figure pack. Use it
   as the first meeting entrypoint for the May 11 discussion.
2. ✅ **CaseHOLD direct option-table route is no longer implementation-blocked**:
   SLURM job `67744` completed with exit `0:0`. The repaired
   `adaptive_snap_hyre_option_table` route runs on the held-out CaseHOLD rows
   200-249 and passes `analyze_detail_flags.py` plus
   `audit_adaptive_hyre_logs.py`.
3. ⚠️ **CaseHOLD direct option-table is a clean negative design point**:
   `adaptive_snap_hyre_option_table` is 35/50 = 70.0% with 2.00 calls. It is a
   small lift over `rag_simple` on the same rows (34/50 = 68.0%; +2pp,
   b/c=2/1, p=1.0000), but below `rag_rewrite` (38/50 = 76.0%; -6pp,
   b/c=1/4, p=0.3750) and `adaptive_snap_hyre_diverse` (39/50 = 78.0%; -8pp,
   b/c=2/6, p=0.2891). Cite this as evidence that answer-option conversion
   remains a distinct bottleneck, not as a positive route.
4. ✅ **Figure pack generated from source-gated summaries**:
   `scripts/build_meeting_package_figures.py` emits figures 12-16 under
   `docs/presentation/figures/`. Figures 12-14 use the diagnostic controller
   portfolio and held-out query/controller JSON files; figures 15-16 are
   scripted diagrams derived from the meeting-prep source claims and linked
   result docs.
5. ⚠️ **Snap-only ladder controls are locally auditable across all four legal benchmarks**:
   BarExam `snap_only_in_final` job `67773` is 171/200 = 85.5% with 2.00 calls,
   errors 0, and one missing prediction. HousingQA `snap_only_in_final` job
   `67775` is 110/200 = 55.0% with 2.00 calls, errors 0, and one missing
   prediction. CaseHOLD `snap_only_in_final` job `67867` is 145/200 = 72.5%
   with 2.00 calls, errors 0, no missing predictions, and no long-answer rows;
   it supersedes health-caveated job `67777`. LegalBench-SCALR
   `snap_only_in_final` job `67779` is 145/200 = 72.5% with 2.00 calls, errors
   0, and no missing predictions. All four detail logs were copied from the
   cluster and summarized with `scripts/analyze_detail_flags.py`; empty
   retrieval payloads are expected for `snap_only_in_final`.
6. ✅ **Retrieval-bearing blocker is repaired enough for evaluated jobs**:
   `rag_utils.py` reinitializes the GTE remote-code `position_ids` buffer.
   Direct embedding smoke `67820` produced finite unit-norm query embeddings,
   and `rag_hyde` smoke `67821` completed 5/5. The N=200 HyRE-only and fixed
   Snap-HyRE rows below have since landed under the normal gates; SCALR
   HyRE-only uses capped rerun `67864` with a postprocess-wrapper caveat.
7. ✅ **BarExam HyRE-only landed as a modest positive retrieval control**:
   `rag_hyde` job `67825` completed with exit `0:0` at 164/200 = 82.0%,
   average calls 2.00, errors 0, missing predictions 0, and empty retrieval 0.
   It improves over baseline retrieval (80.0%) but trails snap-only reasoning
   (85.5%) and the stronger fixed Snap-HyRE v2 route (86.0%), so cite it as
   evidence for routing between rewrite/Snap-HyRE rather than generic HyRE-only.
8. ⚠️ **HousingQA HyRE-only landed as a negative control**:
   `rag_hyde` job `67826` completed with exit `0:0` at 100/200 = 50.0%,
   average calls 2.00, errors 0, missing predictions 0, and empty retrieval 0.
   Cite as evidence against generic HyRE retrieval for HousingQA; the stronger
   route remains state-filter retrieval plus verifier.
9. ⚠️ **CaseHOLD HyRE-only landed as a weak/negative retrieval control**:
   `rag_hyde` job `67827` completed with exit `0:0` at 143/200 = 71.5%,
   average calls 2.00, errors 0, one missing prediction, and empty retrieval 0.
   It trails the current N=200 `rag_simple` baseline (73.0%), snap-only
   control (72.5%), and diverse HyRE-family row (73.5%), so generic HyRE-only
   does not resolve the CaseHOLD answer-option conversion bottleneck.
10. ❌ **SCALR HyRE-only uncapped completed but is not a clean report row**:
   `rag_hyde` job `67828` completed with exit `0:0` at 142/200 = 71.0%, but
   `analyze_detail_flags.py` flags one runaway final answer with 267,458 chars
   and 70,593 output tokens. Do not cite this as a clean method result.
11. ⚠️ **SCALR capped HyRE-only landed as wrapper-caveated evidence**:
   `rag_hyde` job `67864` completed the eval loop at 148/200 = 74.0%, average
   calls 2.00, errors 0, missing predictions 1, empty retrieval 0, and no
   long-answer rows. The SLURM job state is `FAILED` because the wrapper tried
   to run missing `scripts/postprocess_adaptive_hyre_sweep.py` after writing the
   detail log. Cite as detail-log clean but wrapper-caveated; it matches the
   SCALR baseline and trails fixed Snap-HyRE/controller rows.
12. ⚠️ **HousingQA fixed Snap-HyRE landed as a negative control**:
   `rag_snap_hyde_2call` job `67830` completed with exit `0:0` at 103/200 =
   51.5%, average calls 2.00, errors 0, missing predictions 0, empty retrieval
   0, and no long-answer rows. It is below snap-only, state-filter retrieval,
   snap-HyRE state retrieval, and the verifier route.
13. ⚠️ **CaseHOLD fixed Snap-HyRE landed as weak/negative**:
   `rag_snap_hyde_2call` job `67831` completed with exit `0:0` at 144/200 =
   72.0%, average calls 2.00, errors 0, missing predictions 0, empty retrieval
   0, and no long-answer rows. It trails baseline retrieval, clean snap-only,
   and diverse HyRE-family rows.
14. ✅ **BarExam fixed Snap-HyRE landed**:
   `rag_snap_hyde_2call` job `67829` completed with exit `0:0` at 169/200 =
   84.5%, average calls 2.00, errors 0, one missing prediction, empty retrieval
   0, and no long-answer rows. It beats baseline retrieval (80.0%) and
   HyRE-only (82.0%), but trails snap-only (85.5%) and adaptive Snap-HyRE v2
   (86.0%).
15. ✅ **Groq Llama 70B held-out sanity mostly landed**:
   clean rows: BarExam `rag_simple` 38/50 = 76.0%, BarExam
   `adaptive_snap_hyre_v2` 36/50 = 72.0%, HousingQA `rag_state_filter` 22/50 =
   44.0%, HousingQA verifier 30/50 = 60.0%, CaseHOLD `rag_simple` 33/50 =
   66.0%, SCALR `rag_simple` 41/50 = 82.0%, and SCALR frontier 44/50 = 88.0%.
   CaseHOLD diverse HyRE is rejected as clean model-coverage evidence because
   it has errors 2, empty retrieval 2, and missing predictions 2.
16. ❌ **Full-SCALR probe `67863` is not a promoted result**:
   job `67863` wrote the full-SCALR `rag_simple` detail log at 424/571 =
   74.3%, average calls 1.00, errors 0, missing predictions 0, and empty
   retrieval 0, but `analyze_detail_flags.py` flags three long-answer rows
   with max 233,166 final-answer chars / 73,151 output tokens. Do not cite as a
   clean full-corpus baseline. The paired frontier half then produced a
   232,797-character answer at row 296 and was cancelled before writing a clean
   detail log. Capped replacement `67897` is running with
   `LLM_MAX_COMPLETION_TOKENS=4096`; do not cite it unless both modes finish and
   pass validation.
17. ✅ **CaseHOLD capped snap-only replacement `67867` is clean**:
   `67866` is rejected because it was cancelled at 71/200 after row 12 produced
   a 157,678-character answer and `pred=None`. After patching OpenRouter caps
   through `extra_body.max_tokens`, replacement `67867` completed with exit
   `0:0` at 145/200 = 72.5%, average calls 2.00, errors 0, missing predictions
   0, and no long-answer rows. Use `67867`, not health-caveated `67777`, for
   the snap-only ladder.
18. ✅ **Capped SCALR baseline half of `67897` is clean, paired frontier health-gated**:
   the completed `rag_simple` mode was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1218_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 419/571 = 73.4%, average
   calls 1.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 4,405, and no long-answer rows. Cite as a verified baseline-half log
   only; the paired frontier half is health-gated below.
19. ✅ **CaseHOLD N=500 baseline mode of `67913` is clean, paired modes pending**:
   the completed `rag_simple` mode was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1334_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 359/500 = 71.8%, average
   calls 1.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 2,725, and no long-answer rows. Cite only as a verified
   baseline-mode log until `rag_rewrite` and `adaptive_snap_hyre_diverse`
   finish and validate.
20. ❌ **Capped SCALR frontier half of `67897` completed but is health-gated**:
   the paired `adaptive_snap_hyre_frontier` detail log was copied locally from
   `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl`
   and reaches 417/571 = 73.0%, average calls 2.00, errors 0, parse failures 0,
   and empty retrieval 0. It is not a clean report row: one row has no
   predicted answer, `scripts/audit_adaptive_hyre_logs.py` fails with
   `missing_prediction=1`, and `scripts/analyze_detail_flags.py` flags one
   long-answer row with max final-answer chars 20,480 / max output tokens
   8,454. Do not promote `67897` as a clean paired full-SCALR result.
21. ✅ **SCALR N=571 query-rewrite retry `67915` completed cleanly**:
   `rag_rewrite` was copied locally from
   `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1542_legalbench_scalr_meeting-n500-canonical-r2-or-gemma4-26b-legalbench_scalr-n571-k5-rag_rewrite_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 423/571 = 74.1%, average
   calls 2.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 4,005, and no long-answer rows. This is the clean N>=500 SCALR
   rewrite control after invalid CUDA/ECC attempt `67914`.
22. ⚠️ **BarExam N=500 baseline mode of `67911` completed with one missing prediction**:
   `rag_simple` was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1538_barexam_meeting-n500-canonical-or-gemma4-26b-barexam-n500-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 400/500 = 80.0%, average
   calls 1.00, errors 0, empty retrieval 0, max output tokens 2,260, and no
   long-answer rows, but has one missing prediction. Cite as a verified
   baseline-mode log with that caveat until `rag_rewrite` and
   `adaptive_snap_hyre_v2` finish and validate.
23. ⚠️ **HousingQA N=500 state-filter baseline mode of `67912` completed with one missing prediction**:
   `rag_state_filter` was copied locally from
   `logs/eval_rag_state_filter_or-gemma4-26b_20260511_1552_housing_meeting-n500-canonical-or-gemma4-26b-housing-n500-k5-rag_state_filter_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 270/500 = 54.0%, average
   calls 1.00, errors 0, empty retrieval 0, max output tokens 967, and no
   long-answer rows, but has one missing prediction. Cite as a verified
   baseline-mode log with that caveat until `rag_rewrite` and
   `adaptive_snap_hyre_housing_verifier` finish and validate.
24. ✅ **CaseHOLD N=500 query-rewrite mode of `67913` completed cleanly**:
   `rag_rewrite` was copied locally from
   `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1557_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_rewrite_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 354/500 = 70.8%, average
   calls 2.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 2,213, and no long-answer rows. This is a clean N>=500 CaseHOLD
   rewrite control; diverse HyRE remains pending.
25. ❌ **Remaining N=500 scale-up jobs timed out before completing all modes**:
   SLURM jobs `67911`, `67912`, and `67913` hit the 4-hour time limit and the
   stdout logs were copied locally as
   `logs/slurm_67911_n500_barexam_timeout.out`,
   `logs/slurm_67912_n500_housing_timeout.out`, and
   `logs/slurm_67913_n500_casehold_timeout.out`. Do not promote partial modes:
   BarExam `rag_rewrite` stopped at `92/500` and adaptive v2 was not reached;
   HousingQA `rag_rewrite` stopped at `116/500` and the verifier was not
   reached; CaseHOLD diverse HyRE stopped at `35/500`.

### New source paths

- Meeting package: `docs/archive/diagnostic_adaptation_2026-05-12/meeting_prep_2026-05-11_diagnostic_adaptation.md`.
- Package audit: `docs/archive/diagnostic_adaptation_2026-05-12/meeting_package_audit_2026-05-11.md`.
- Snap-only summary: `docs/snap_only_controls_2026-05-11.json`.
- CaseHOLD direct option-table result:
  `docs/archive/diagnostic_adaptation_2026-05-12/casehold_option_table_direct_heldout_2026-05-11.md`.
- Figure generator: `scripts/build_meeting_package_figures.py`.
- Figure outputs:
  `docs/presentation/figures/12_diagnostic_adaptation_calibration_ablation.png`,
  `docs/presentation/figures/13_diagnostic_adaptation_heldout_ablation.png`,
  `docs/presentation/figures/14_diagnostic_controller_macro_lift.png`,
  `docs/presentation/figures/15_bottleneck_diagnostic_route_map.png`,
  `docs/presentation/figures/16_method_ladder_flowchart.png`.
- Snap-only detail logs:
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0346_barexam_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0259_housing_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0943_casehold_meeting-capped-snap-casehold-v2-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0411_legalbench_scalr_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`.
- BarExam HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0526_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- HousingQA HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0443_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- CaseHOLD HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0511_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- SCALR HyRE-only rejected detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0559_legalbench_scalr_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- SCALR HyRE-only capped detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0734_detail.jsonl`.
- Full-SCALR `rag_simple` health-gated detail log:
  `logs/eval_rag_simple_or-gemma4-26b_20260511_0731_legalbench_scalr_meeting-full-scalr-sanity-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`.
- Full-SCALR cancelled stdout:
  `logs/slurm_67863_full_scalr_cancelled.out`.
- HousingQA fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0559_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- CaseHOLD fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0602_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- BarExam fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0626_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- Groq held-out sanity detail logs:
  `logs/eval_rag_simple_groq-llama70b_20260511_0604_barexam_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_v2_groq-llama70b_20260511_0605_barexam_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_v2_detail.jsonl`,
  `logs/eval_rag_state_filter_groq-llama70b_20260511_0622_housing_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_state_filter_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_housing_verifier_groq-llama70b_20260511_0624_housing_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_housing_verifier_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260511_0610_casehold_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_diverse_groq-llama70b_20260511_0617_casehold_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_diverse_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260511_0622_legalbench_scalr_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_frontier_groq-llama70b_20260511_0626_legalbench_scalr_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_frontier_detail.jsonl`.
- N>=500 scale-up baseline-mode detail logs:
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1218_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`,
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1538_barexam_meeting-n500-canonical-or-gemma4-26b-barexam-n500-k5-rag_simple_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260511_1552_housing_meeting-n500-canonical-or-gemma4-26b-housing-n500-k5-rag_state_filter_detail.jsonl`,
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1334_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_simple_detail.jsonl`.
- N>=500 clean rewrite detail logs:
  `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1542_legalbench_scalr_meeting-n500-canonical-r2-or-gemma4-26b-legalbench_scalr-n571-k5-rag_rewrite_detail.jsonl`,
  `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1557_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_rewrite_detail.jsonl`.
- N>=500 rejected/health-gated adaptive detail logs:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl`.
- N>=500 timeout stdout logs:
  `logs/slurm_67911_n500_barexam_timeout.out`,
  `logs/slurm_67912_n500_housing_timeout.out`,
  `logs/slurm_67913_n500_casehold_timeout.out`.

## Update 2026-05-01 ~meeting prep

Change reason: pulled completed cluster jobs `58282` and `58283`, fixed the
Housing state-filter metadata casing bug, and consolidated the meeting story in
`docs/meeting_state_2026-05-01.md`.

Last updated: 2026-05-01
Branch: `codex/final-report-snap-hyde`

### Delta since 2026-04-30 overlay

1. ✅ **CaseHOLD repaired pair landed**: after rebuilding `casehold_holdings`,
   `rag_simple` is 69.5% and `rag_snap_hyde_2call` is 72.0%, +2.5pp,
   b/c=16/11, McNemar p=0.4421, 95% bootstrap CI [-2.5, +7.5] pp. Gold
   retrieval is now meaningful for this pair: 16.0% -> 47.0%. Cite as
   "better gold retrieval does not yet translate into reliable answer lift,"
   not as a positive method result.
2. ⚠️ **CaseHOLD repaired top-k and HyDE follow-ups landed**: k=1 is 64.5%,
   k=5 is 69.5%, k=10 is 68.0%, and `rag_hyde` is 72.0%. The k=1 -> k=5
   depth delta is +5.0pp with McNemar p=0.0525; k=5 -> k=10 is flat/negative
   (-1.5pp, p=0.6072). Cite as a diagnostic option-conversion bottleneck,
   not as a reliable answer lift.
3. ❌ **Housing state-filter job `58282` is invalid as a method result**:
   both k=5 and k=10 runs were tagged `_FAILED-EMPTY-RETRIEVAL`, with 200/200
   rows having no retrieved evidence. The logged accuracies, 53.5% and 55.0%,
   are parametric behavior and must not be cited as state-filtered retrieval.
4. ✅ **Housing state-filter blocker unblocked for N=200 diagnostics**:
   `_housing_state_where(...)` now lowercases question states to match the
   lowercase statute metadata in `datasets/housing_qa/statutes.csv`. The fixed
   cluster run `58799` landed a clean k=5 row: 123/200 (61.5%), 0/200 empty
   retrieval, 81/200 gold retrieved. It beats generic top-5 by +8.0pp
   (b/c=36/20, p=0.0440) and is directionally above generic top-10 by +3.5pp
   (p=0.4350). The chunked k=10 rerun `58937` landed cleanly at 125/200
   (62.5%), 0/200 empty retrieval, 98/200 gold retrieved. It beats generic
   top-5 by +9.0pp (b/c=34/16, p=0.0153), is directionally above generic top-10
   by +4.5pp (p=0.3057), and is only +1.0pp above state-filter k=5 (p=0.8145).
   Cite as metadata-filtering signal, not as deeper-is-always-better.
5. 🧭 **Meeting framing**: top-k sensitivity should be called a cheap
   retrieval-policy stress test or first-pass bottleneck signal. It directly
   probes retrieval-depth/candidate-set sensitivity; query formulation,
   evidence use, metadata filtering, and option anchoring require the broader
   diagnostic matrix.

### New source paths

- Meeting synthesis: `docs/meeting_state_2026-05-01.md`.
- CaseHOLD repaired rerun: `docs/casehold_repaired_rerun_2026-05-01.md`.
- Housing state-filter followup: `docs/housing_state_filter_followup_2026-05-01.md`.
- Pulled detail logs:
  `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`,
  `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260501_1440_detail.jsonl`,
  `logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1649_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1720_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260501_1406_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260501_k10_merged_detail.jsonl`.

## Update 2026-04-30 ~15:30 CDT

Change reason: second adversarial report pass found that the class report had
outpaced this signoff layer. This update is a source-gated overlay for the
2026-04-28/30 bottleneck-taxonomy pivot; older sections are preserved below for
traceability.

Last updated: 2026-04-30 ~15:30 CDT
Branch: `codex/evidence-ledger-router`, HEAD after report refresh: `b59cb62`

### Current quick reference

1. ✅ **MuSiQue Llama 70B `snap_hyde_2call` is the current N=200 method
   vehicle**: `rag_simple` 27.5% -> `snap_hyde_2call` 37.0%, +9.5pp,
   b/c=33/14, McNemar p=0.007943, 95% bootstrap CI [+3.0, +16.0] pp.
   Cite as paired N=200, not full-corpus.
2. ✅ **MuSiQue top-k collapse is the cleanest retrieval-depth diagnostic**:
   `rag_simple` top-5 27.5% -> top-1 13.0%, -14.5pp, b/c=3/32,
   p=4.177e-07, CI [-20.0, -9.5] pp.
3. ✅ **BarExam top-k is depth-flat on the N=200 diagnostic slice**:
   `rag_simple` top-5 82.5% -> top-1 83.0%, +0.5pp, b/c=18/17, p=1.0.
   Keep separate from the full-corpus BarExam method result.
4. ✅ **LegalBench-SCALR is candidate-depth limited then saturated**:
   top-5 77.0% -> top-1 59.5%, -17.5pp, b/c=3/38, p=1.048e-08;
   top-10 ties top-5 at 77.0%, b/c=8/8, p=1.0.
5. ⚠️ **HousingQA is a directional statutory depth signal**:
   top-1 50.5% -> top-10 58.0%, +7.5pp, b/c=38/23, p=0.0722.
   Cite with state-metadata and low gold-hit caveats.
6. ⚠️ **CaseHOLD is answer-flat under current logs, not retrieval-recall
   evidence**: top-5 72.0% -> top-1 70.5%, -1.5pp, b/c=10/13, p=0.6776;
   two-call 69.5%, -2.5pp, b/c=14/19, p=0.4869. Old gold-hit is 0/200 due to
   instrumentation; wait for repaired Chroma rerun before retrieval claims.
7. ✅ **MuSiQue `golden_passage` control confirms context utility is
   task-dependent**: `golden_passage` reaches 56.5% EM, beating `rag_simple`
   by +29.0pp (b/c=64/6, p=2.44e-13) and `snap_hyde_2call` by +19.5pp
   (b/c=47/8, p=8.07e-08). This is a privileged-context diagnostic, not a
   deployable method.

### Current source paths

- Bottleneck matrix and paired deltas: `docs/evidence_matrix_2026-04-30.md`.
- Housing metrics and metadata caveats:
  `docs/housing_speculative_metrics_2026-04-30.md` and
  `docs/housing_metadata_depth_audit_2026-04-30.md`.
- CaseHOLD instrumentation caveat and repair:
  `docs/casehold_flatness_audit_2026-04-30.md` and
  `docs/casehold_gold_mapping_repair_2026-04-30.md`.
- SCALR depth audit: `docs/scalr_depth_disagreement_2026-04-30.md`.
- MuSiQue golden control: `docs/musique_golden_passage_2026-04-30.md`.
- Full-corpus BarExam method matrix remains Section A below.

## Update 2026-04-27 ~12:30 CDT

Change reason: added the 2026-04-27 ~12:30 CDT McNemar results for Llama planning methods and BarExam cross-domain mhd. That McNemar section gives paired statistics but no separate audit IDs, so new audit cells cite the 12:30 McNemar source rather than inventing IDs.

Last updated: 2026-04-27 ~12:30 CDT
Branch: hpc-setup, HEAD: a50f67a

This log lists results that have:
1. Landed cleanly (no preflight failure, no harness crash)
2. Passed per-entry confound audit (codex sampled records: no MAJOR truncation, leakage, fallback, empty-retrieval, or format issues)
3. Been reviewed by architect (Claude Opus) for paper-defensibility
4. Have a direct path to detail log + commit SHA + audit doc

**Sign-off levels:**
- ✅ **APPROVED** — cite freely, paper-grade
- ⚠️ **APPROVED-WITH-CAVEAT** — cite with the documented caveat
- ⏸ **PENDING** — landed but awaiting audit
- ❌ **REJECTED** — known confound, do not cite

## Quick reference: top 5 cite-able findings for the paper (Tier 2 MuSiQue = N=200 paired; full-corpus replicate pending)

1. ✅ BarExam snap+HyDE is the Tier 3 legal-MC winner: Gemma 4 26B-A4B 78.08% → 81.17% (+3.09pp) and Gemma 4 E4B 58.49% → 62.18% (+3.69pp). **Architecture note**: ~76-83% of `rag_snap_hyde` final preds match `snap_letter` (BY DESIGN architecture — the mode combines snap reasoning + HyDE retrieval; snap reasoning dominates because Gemma 4 has strong legal MC priors). HyDE provides marginal lift and sometimes conflicting evidence; when pred==snap, EM=88.7%, while pred≠snap is 45.7%. Frame this as mechanism understanding.
2. ✅ Llama 70b MuSiQue `multi_hyde_diverse` is the superseded pre-pivot Tier 2 N=200 paired multi-hop headline: 27.5% → 35.5%, +8pp, McNemar p=0.0195; *pending full-corpus replicate*. The current 2026-04-30/05-01 meeting vehicle is `snap_hyde_2call` plus the bottleneck taxonomy.
3. ⚠️ Llama 70b MuSiQue `iterative_planning_table` is cite-able as N=200 paired TRENDING-SIG, not fully significant: 27.5% → 36.0%, +8.5pp, p=0.0533; *pending full-corpus replicate*.
4. ✅ Gemma 3 27B MuSiQue mhd is a cite-able N=200 paired negative cross-family check: 28.5% → 31.0%, +2.5pp, p=0.5901 NULL; *full-corpus replicate would solidify*.
5. ⚠️ Llama 70b MuSiQue `subagent_rag` N=200 paired -12pp p=0.0007 SIG NEGATIVE. **Implementation caveat**: 200/200 records triggered gap-routing (100% rate; over-aggressive); 59/200=29.5% finals are "Unknown/Not found" vs 12.5–15% in other methods. With our gap-routing implementation, `subagent_rag` systematically over-abstains on multi-hop and produces a real -12pp finding; reframing the prompt could likely close part of this gap. Do not generalize beyond this implementation.

## Audit lineage (2026-04-27 ~14:30 CDT, comprehensive per-log Haiku audit)

Per-log audit reports under `docs/audits/`:
- `2026-04-27_barexam_26b_audit.md` — 8 logs × N=1195. Initial Haiku review raised a snap-dominance concern; architect-verified as BY DESIGN architecture (the mode combines snap+HyDE, snap dominates because Gemma 4 has strong legal priors, and HyDE acts as marginal evidence). 7/8 CLEAN, 1 architecture-clarified.
- `2026-04-27_barexam_e4b_audit.md` — 8 logs × N=1195. All ✅ CLEAN.
- `2026-04-27_llama70b_musique_audit.md` — 8 logs × N=200. All data CLEAN; subagent_rag flagged for implementation quirk (100% gap-routing trigger → over-abstention) — caveat documented in Top 5 #5.
- `2026-04-27_other_tier2_audit.md` — 6 logs × N=200. All ✅ CLEAN.

---

## Section A — Tier 3 / Full corpus

### A.1 BarExam Gemma 4 26B-A4B method matrix at N=1195

| Mode | EM | Audit | Sign-off | Caveat |
|---|---:|---|---|---|
| `rag_simple` | 78.08% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | 2/15 sampled records had null pred + empty retrieval; 933/1195 = 78.08% holds |
| `rag_snap_hyde` | 81.17% | MINOR | ✅ APPROVED | low BarExam exact-gold retrieval (generic to dataset) |
| `snap_only_in_final` | 80.59% | CLEAN | ✅ APPROVED | — |
| `rag_hyde` | 78.91% | MINOR | ✅ APPROVED | low BarExam exact-gold retrieval |
| `subagent_rag` | 78.16% | MINOR | ✅ APPROVED | 8 records empty retrieval in full scan; sample clean |
| `subagent_hybrid` | 74.23% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | rescore note: raw stored 74.14%, audit re-scored to 74.23% |
| `llm_only` | 79.75% | CLEAN | ✅ APPROVED | — |
| `golden_passage` | 78.66% | CLEAN | ✅ APPROVED | — |

**Source-of-truth**: `docs/audit_log.md` (post-fix re-scored from detail logs; experiments.jsonl rows are pre-fix and stale).
**Detail logs**: `logs/eval_*_cluster-vllm_2026042{5,6}_*_detail.jsonl` (see `docs/compiled_results.md` Section 1.2).
**Result commits**: `8bbf0e7` (audit), `ed15eb7` (extractor).

### A.2 BarExam Gemma 4 E4B method matrix at N=1195

| Mode | EM | Audit | Sign-off | Caveat |
|---|---:|---|---|---|
| `rag_simple` | 58.49% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | low exact-gold retrieval; no sampled parser issue |
| `rag_hyde` | 60.59% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `rag_snap_hyde` | 62.18% | MINOR | ✅ APPROVED | one raw null parsed prediction in full scan; sample clean |
| `snap_hyde_report` | 60.75% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `snap_only_in_final` | 57.82% | CLEAN | ✅ APPROVED | — |
| `subagent_hybrid` | 58.83% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `subagent_hyde` | 60.17% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `subagent_rag` | 60.92% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |

**Detail logs**: `logs/eval_*_cluster-vllm_20260426_*_detail.jsonl` (E4B); see `docs/compiled_results.md` Section 1.3.

### A.3 BarExam cross-size headline (PAPER STORY)

**`rag_snap_hyde` lifts BarExam EM at both Gemma 4 sizes:**
- Gemma 4 26B-A4B: +3.09pp (78.08% → 81.17%), b/c=124/87, McNemar p=0.0130,
  95% bootstrap CI [+0.67, +5.44] pp.
- Gemma 4 E4B: +3.68pp (58.49% → 62.18%), b/c=172/128, McNemar p=0.0129,
  95% bootstrap CI [+0.92, +6.53] pp.

**Sign-off**: ✅ APPROVED (Tier 3, cross-size confirmed; both sizes have post-fix detail-log/audit support, with the caveats listed above).

---

## Section B — Tier 2 / N=200 paired McNemar

### B.1 Llama 70b MuSiQue method matrix (PAPER HEADLINE + TRENDING)

| Mode | EM | Δ | McNemar p | Audit | Sign-off |
|---|---:|---:|---:|---|---|
| `rag_simple` | 27.5% | — | — | N=200 paired; CLEAN | ✅ APPROVED (baseline) |
| **`iterative_planning_table`** | **36.0%** | **+8.5pp** | **0.0533** | N=200 paired; McNemar 12:30 | **✅ APPROVED — TRENDING-SIG (*pending full-corpus replicate*)** |
| **`multi_hyde_diverse`** | **35.5%** | **+8pp** | **0.0195** | N=200 paired; CLEAN | **✅ APPROVED — superseded pre-pivot headline (*pending full-corpus replicate*)** |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | N=200 paired; CLEAN | ✅ APPROVED (mechanism decomposition) |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | N=200 paired; CLEAN | ✅ APPROVED (cross-domain neg evidence) |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | N=200 paired; CLEAN | ✅ APPROVED (multi-round neutral at large) |
| `advisor_planning_table` | 23.0% | -4.5pp | 0.222 | N=200 paired; McNemar 12:30 | ✅ APPROVED — NS but informative negative |
| **`subagent_rag`** | **15.5%** | **-12.0pp** | **0.0007** | N=200 paired; CLEAN | **✅ APPROVED — sig negative** |

**Detail logs**: `logs/eval_*_groq-llama70b_20260427_{0952,1010,1019,1036,1044,1112,1208,1216}_detail.jsonl`.
**Source-of-truth**: `docs/mcnemar_2026-04-27.md`.

### B.2 Mechanism decomposition (Llama 70b N=200 paired)

**mhd's +8pp lift decomposes into:**
- HyDE-style answer-bearing passages: ~6.5pp (mhd minus multi_query)
- Query diversity alone: +1.5pp NS (multi_query alone)

**Sign-off**: ✅ APPROVED (Tier 2 N=200 paired; HyDE-style is dominant ~80% contributor; *full-corpus replicate would solidify*).

### B.3 Cross-family negative finding (N=200 paired)

**mhd × Gemma 3 27B N=200 paired = 31.0%, +2.5pp, p=0.5901 NULL**

**Sign-off**: ✅ APPROVED (negative finding) — Tier 2 N=200 paired NULL on Gemma 3 27B; the cross-family lift on dense models is NOT universal; *full-corpus replicate would solidify*.

### B.4 BarExam cross-domain (paired N=200)

| Method / model | Comparator | Result | McNemar p | Sign-off |
|---|---|---:|---:|---|
| `multi_hyde_diverse` × Gemma 4 26B-A4B | N=200 paired first-200 `rag_simple` = 84.5% | 82.0%, -2.5pp | 0.499 | ⏸ SOURCE-PENDING — do not cite as landed |

**Source status**: source-pending in `docs/mcnemar_2026-04-27.md`; keep provisional until the SLURM 55107 detail log lands locally.

---

## Section C — Tier 1 / direction-only (NOT paper-grade alone)

### C.1 Friend/foe attribution-bias probe

| Model | N | Outcome changes | Audit | Sign-off |
|---|---:|---:|---|---|
| Gemma 3 27B | 30 | 4/30 = 13.3% | CLEAN | ⚠️ APPROVED-WITH-CAVEAT (N=30 directional only) |
| Llama 70b | 30 | 6/30 = 20.0% | CLEAN | ⚠️ APPROVED-WITH-CAVEAT (N=30 directional only) |

**Sign-off**: ⚠️ APPROVED-WITH-CAVEAT — cite as "real mechanism detected but limited effect size at N=30". For paper claim, scale to N=100+.

### C.2 iter_hyde × small-model negative direction

| Model | iter_hyde EM | rag_simple comparator | Δ | Sign-off |
|---|---:|---:|---:|---|
| Gemma 3 27B (N=30) | 6.7% | 22% (N=100) | -15pp | ⚠️ DIRECTION-ONLY |
| Llama 4 Scout (N=30) | 16.7% | 30% (N=100) | -13pp | ⚠️ DIRECTION-ONLY |
| Qwen3 30B MoE (N=30) | 6.7% | 24% (N=100) | -17pp | ⚠️ DIRECTION-ONLY |

**Sign-off**: ⚠️ DIRECTION-ONLY (N=30 small samples; direction is consistent but cite as "trend not test").

### C.3 Llama 70b iter_hyde Tier 2 (lift to APPROVED)

iter_hyde × Llama 70b N=200 = -3pp p=0.47 NS (audit CLEAN).

**Sign-off**: ✅ APPROVED — multi-round HyDE doesn't help large dense (statistically null).

---

## Section D — In flight (will sign off when landed + audited)

| Run | Status | Spot-check verdict | Expected sign-off |
|---|---|---|---|
| SLURM 55107 BarExam mhd+iter_hyde × Gemma 4 26B-A4B N=200 | still running; operator snapshot says mhd 82% done and iter_hyde at q106+/200 (78.3% partial PASS rate) | LEGIT by operator spot-check, but source detail log not present locally | Expected ✅ APPROVED only after landing + source log |
| `qwen_full` mhd-pair × Qwen3 30B MoE × N=2400 MuSiQue | RUNNING ~q1058/2400 (rag_simple = 26.1%, slow but progressing) | LEGIT by operator spot-check, but source log not present locally | Tier 2.5 partial only until full run + audit land |

### Section D — KILLED jobs (cannot be relied on as Tier 2/3 results)

| Run | Status at kill | Reason killed | Citation guidance |
|---|---|---|---|
| `gemma4_full` mhd-pair × Gemma 4 26B-A4B × N=2400 MuSiQue (or-gemma4-26b API) | KILLED 2026-04-27 14:00 CDT at q431/2400 (rag_simple partial = 30.9%) | Hung 73+ min on q432 due to OR-served Gemma 4 26B runaway-loop generation (one 91k-char looped answer at q431 took 601s; subsequent query never returned) | ⚠️ Tier 2.5 partial — citeable ONLY as "Gemma 4 26B-A4B `rag_simple` MuSiQue N=431 = 30.9% (partial, OR-Gemma serving cut short by runaway loops)". Do NOT cite as Tier 3. |
| `iterative_planning_table` × Gemma 27B N=200 (or-gemma27b) | KILLED 2026-04-27 14:00 CDT at q29/200 | Same OR-Gemma issue — one query took 2405s = 40 min. Projected ETA was 10+ hours. | ❌ DO NOT CITE — N=29 is below Tier 0 threshold and contains a 40-min outlier. |

### Section D' — OR-served Gemma serving issue (methodology finding)

**Discovered 2026-04-27 ~13:00 CDT:** OpenRouter-served Gemma models (Gemma 4 26B-A4B and Gemma 3 27B) exhibit pathological **runaway-loop generations** on iterative or multi-step prompts. Symptoms:
- Single queries occasionally take 600s, 1200s, 2400s instead of normal 5-30s
- Answer text contains repetitive looping (e.g., "Lou Boudreau (no), it is Lou Boudreau (no)..." for 91k chars)
- ~2% of `rag_simple` (single-call) MuSiQue queries echoed `[your answer here]` placeholder and looped
- Effect compounds in iterative/multi-call modes: `iter_planning_table × Gemma 27B` was projected to take 10+ hours for N=200

**Mitigation**: Use cluster vLLM (Gemma 4 26B-A4B served locally via vLLM nightly + transformers 5.5.0) instead of OR API for Gemma. SLURM 55107 confirms cluster vLLM is clean — same model, no leakage, normal latencies.

**Implication for meeting**: Whenever cited Gemma results were collected via OR, prefer cluster vLLM equivalents. The `gemma4_full` partial result (q431=30.9% rag_simple Gemma 4 26B MuSiQue) should be treated as a noisy lower-bound, not a Tier 2/3 cite-able number.

---

## Section E — Sign-off process

1. Run lands cleanly → enters PENDING
2. Codex per-entry audit (sample 5-10 records) → CLEAN / MINOR / MAJOR
3. Architect reviews audit + cross-checks sources → ✅ APPROVED / ⚠️ APPROVED-WITH-CAVEAT / ❌ REJECTED
4. Entry added here with date/time + commit SHA + paths
5. Compiled_results.md is the detailed reference; this log is the cite-or-not gate

**Architect**: Claude Opus 4.7 (1M context), this session.
**Audit principal**: codex CLI 0.126.0-alpha.4 with `~/.codex/config.toml` defaults.

## Section F — Historical N≥200 runs retroactively audited

(Audited 2026-04-27 ~12:00 CDT, 3-record spot-check per row)

| Tag | Mode | Provider | N | EM | T? | E? | Th? | ER? | Sign-off |
|---|---|---|---|---|---|---|---|---|---|
| `captain-llama70b-musique-mhd-n200` | `multi_hyde_diverse` | `groq-llama70b` | 200 | 35.5% | N | N | N | N | ✅ APPROVED |
| `mhd-pair-gemma27b-n200-power` | `multi_hyde_diverse` | `or-gemma27b` | 200 | 31.0% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-snap-hyde-n200` | `rag_snap_hyde` | `groq-llama70b` | 200 | 24.0% | N | N | N | N | ✅ APPROVED |
| `26b-seed99-repeat` | `rag_snap_hyde` | `custom` | 1195 | 75.4% | N | N | N | N | ✅ APPROVED |
| `e4b-n200-postfix-v2` | `rag_snap_hyde` | `custom` | 200 | 67.5% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `rag-multi-query-scout-n200` | `rag_multi_query` | `groq-scout` | 200 | 30.5% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-multi-query-n200` | `rag_multi_query` | `groq-llama70b` | 200 | 29.0% | N | N | N | N | ✅ APPROVED |
| `rag-multi-query-gemma27b-n200` | `rag_multi_query` | `or-gemma27b` | 200 | 28.5% | N | N | N | N | ✅ APPROVED |
| `rag-simple-scout-n200` | `rag_simple` | `groq-scout` | 200 | 30.0% | N | N | N | N | ✅ APPROVED |
| `mhd-pair-gemma27b-n200-power` | `rag_simple` | `or-gemma27b` | 200 | 28.5% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-musique-rag-simple-n200` | `rag_simple` | `groq-llama70b` | 200 | 27.5% | N | N | N | N | ✅ APPROVED |
| `e4b-seed99-repeat` | `rag_simple` | `custom` | 1195 | 55.7% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `e4b-n200-prompt-fix` | `rag_simple` | `custom` | 200 | 61.5% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `e4b-n200-postfix-v2` | `rag_simple` | `custom` | 200 | 61.0% | N | N | N | N | ✅ APPROVED |
| `e4b-n200-postfix-v2` | `rag_hyde` | `custom` | 200 | 61.5% | N | N | N | N | ✅ APPROVED |

T? = Truncation, E? = Empty pred, Th? = <think> leak, ER? = Empty retrieval

## Section G — Historical runs INVALIDATED (do not cite)
- Pre-fix BarExam (timestamps before 2026-04-22): `26b-seed99-repeat` (2026-04-21T21:15:16Z, `rag_simple`); `26b-baseline-ceiling` (2026-04-21T21:58:57Z, `golden_passage`); `31b-full-matrix` (2026-04-21T22:09:55Z, `rag_hyde`); `26b-subagent-1` (2026-04-21T22:26:13Z, `subagent_rag`); `26b-subagent-2` (2026-04-21T22:30:08Z, `subagent_hybrid`); `e2b-full-matrix-redo` (2026-04-21T22:58:01Z, `rag_hyde`); `26b-seed99-repeat` (2026-04-21T23:33:23Z, `rag_hyde`); `26b-full-matrix` (2026-04-21T23:39:52Z, `snap_only_in_final`)
- Empty-retrieval contaminated: `api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL` (2026-04-27T03:42:40Z, `advisor_planning_table`)
- Smoke/test runs: `api-smoke` (2026-04-26T04:44:05Z, `llm_only`); `api-musique-smoke2` (2026-04-26T04:54:25Z, `llm_only`); `api-musique-ptable-smoke` (2026-04-26T22:20:07Z, `planning_table`); `api-smoke-groq-qwen` (2026-04-27T00:23:27Z, `llm_only`); `api-smoke-groq-llama70b` (2026-04-27T00:23:27Z, `llm_only`)

---

## Section F — Historical runs (retroactively audited 2026-04-27)

Scope: top paper-relevant historical rows from `logs/experiments.jsonl`, excluding rows already covered in Sections A/B/C. For rows with detail logs, codex checked first 2 + middle 1 + last 2 records for truncation, empty predictions, `<think>` leakage, snap-letter echo, fallbacks, and empty retrieval; obvious full-log quality counters were also checked. Missing detail log means `PENDING`.

### F.1 BarExam Gemma 4 26B-A4B historical (post-fix era)

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `20260421_2149 / 26b-subagent-2` | `snap_hyde_report` | 1195 | 76.57% | detail log missing (`logs/eval_snap_hyde_report_cluster-vllm_20260421_2149_detail.jsonl`) | ⏸ PENDING |
| `20260421_2150 / 26b-subagent-1` | `subagent_hyde` | 1195 | 76.57% | detail log missing (`logs/eval_subagent_hyde_cluster-vllm_20260421_2150_detail.jsonl`) | ⏸ PENDING |
| `20260421_2234 / 26b-seed99-repeat` | `rag_snap_hyde` | 1195 | 75.40% | 5-row spot clean; no empty pred, no snap-stage echo, no fallback, no empty retrieval; superseded by Section A current 26B matrix | ⚠️ APPROVED-WITH-CAVEAT |

### F.2 BarExam Gemma 4 E4B historical

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `20260421_2000 / e4b-seed99-repeat` | `rag_simple` | 1195 | 55.73% | detail log missing (`logs/eval_rag_simple_cluster-vllm_20260421_2000_detail.jsonl`) | ⏸ PENDING |
| `20260421_2239 / e4b-n200-postfix-v2` | `rag_simple` | 200 | 61.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `20260421_2312 / e4b-n200-postfix-v2` | `rag_hyde` | 200 | 61.50% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `20260421_2331 / p1a-full-rerun` | `snap_only_in_final` | 1195 | 54.81% | detail log missing (`logs/eval_snap_only_in_final_cluster-vllm_20260421_2331_detail.jsonl`) | ⏸ PENDING |
| `20260422_0007 / e4b-n200-postfix-v2` | `rag_snap_hyde` | 200 | 67.50% | detail log missing (`logs/eval_rag_snap_hyde_cluster-vllm_20260422_0007_detail.jsonl`) | ⏸ PENDING |

### F.3 MuSiQue historical (Llama 70b, Gemma 27B, Scout, Qwen; N≥100 only)

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `api-musique-rag-simple-llama-n100` | `rag_simple` / Llama 70b | 100 | 21.00% | 5-row spot clean; `audit_log.md` paired-advisor check re-scored 21/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-advisor-llama-n100` | `advisor_planning_table` / Llama 70b | 100 | 23.00% | 5-row spot clean; `audit_log.md` says CLEAN but not statistically significant vs rag_simple | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-mhd-llama-n100` | `multi_hyde_diverse` / Llama 70b | 100 | 33.00% | 5-row spot clean; `audit_log.md` cross-family N=100 audit confirmed 33/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-rag-simple-gemma27b-n100` | `rag_simple` / Gemma 3 27B | 100 | 22.00% | 5-row spot clean; `audit_log.md` confirmed 22/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-mhd-gemma27b-n100` | `multi_hyde_diverse` / Gemma 3 27B | 100 | 30.00% | 5-row spot clean; `audit_log.md` confirmed 30/100, p=0.134 trend vs rag_simple | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-scout-n100` | `rag_simple` / Scout | 100 | 30.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-scout-n100` | `multi_hyde_diverse` / Scout | 100 | 29.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-snap-hyde-llama-musique-n100` | `rag_snap_hyde` / Llama 70b | 100 | 21.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 echo/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-llama-musique-n100` | `rag_multi_query` / Llama 70b | 100 | 25.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-scout-musique-n100` | `rag_multi_query` / Scout | 100 | 25.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-scout-n200` | `rag_multi_query` / Scout | 200 | 30.50% | sample clean, but full log has 1 placeholder-echo prediction (`[your answer here]`) counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-qwen-n100` | `rag_simple` / Qwen3 30B MoE | 100 | 24.00% | sample clean, but full log has 1 blank final answer / empty prediction counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-qwen-n100` | `multi_hyde_diverse` / Qwen3 30B MoE | 100 | 28.00% | sample clean, but full log has 1 generate-empty error and 2 empty predictions counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-simple-scout-n200` | `rag_simple` / Scout | 200 | 30.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `rag-multi-query-gemma27b-n200` | `rag_multi_query` / Gemma 3 27B | 200 | 28.50% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |

### F.4 BarExam other models (Qwen3 30B MoE, Llama 70b, Scout, Gemma 27B)

No BarExam Llama 70b N≥200 row was found in the Apr. 20-26 historical slice; the clean N=100 cross-family API rows are signed below as support-only results.

| Tag | Model | Mode | N | EM | Audit | Sign-off |
|---|---|---|---:|---:|---|---|
| `api-cross-scout-n100` | Llama 4 Scout 17B | `llm_only` | 100 | 67.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-llama70b-n100` | Llama 3.3 70B | `llm_only` | 100 | 81.00% | 5-row spot clean; `audit_log.md` cross-family check says CLEAN | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-qwen3-32b-n100` | Qwen3 32B dense | `llm_only` | 100 | 68.00% | `audit_log.md` found 13/100 truncated mid-`<think>` with `predicted_answer=None`; sample reproduced 1 empty pred | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-gemma3-27b` | Gemma 3 27B | `llm_only` | 100 | 68.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-qwen3-30b-moe-n100` | Qwen3 30B MoE | `llm_only` | 100 | 70.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |

## Section G — Historical runs INVALIDATED (do not cite)

### G.1 Pre-fix BarExam rows (formatter/retrieval-query bug window)

Current BarExam citations must use the post-fix source-of-truth values in `docs/audit_log.md` / Sections A and F. These `logs/experiments.jsonl` rows are retained only as historical references because they landed before the `3d5ff05` retrieval-query fix or in the immediate pre-2026-04-22 bug window:

- `20260420_2349_rag_snap_hyde_cluster-vllm_leak-fix-validation` (`leak-fix-validation`, N=30)
- `20260421_0055_rag_simple_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0203_rag_hyde_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0359_rag_snap_hyde_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0405_rag_simple_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0458_rag_hyde_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0526_snap_only_in_final_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0632_rag_snap_hyde_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0746_snap_only_in_final_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0802_rag_simple_cluster-vllm_e2b-full-matrix` (`e2b-full-matrix`, N=1195)
- `20260421_0812_rag_simple_cluster-vllm_p1a-full-rerun` (`p1a-full-rerun`, N=1195)
- `20260421_0857_rag_simple_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1112_rag_hyde_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1203_rag_simple_cluster-vllm_31b-full-matrix` (`31b-full-matrix`, N=1195)
- `20260421_1402_rag_snap_hyde_cluster-vllm_p1b-full-rerun` (`p1b-full-rerun`, N=1195)
- `20260421_1449_rag_hyde_cluster-vllm_p1a-full-rerun` (`p1a-full-rerun`, N=1195)
- `20260421_1501_llm_only_cluster-vllm_26b-baseline-ceiling` (`26b-baseline-ceiling`, N=1195)
- `20260421_1515_rag_snap_hyde_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1615_rag_simple_cluster-vllm_26b-seed99-repeat` (`26b-seed99-repeat`, N=1195)
- `20260421_1658_golden_passage_cluster-vllm_26b-baseline-ceiling` (`26b-baseline-ceiling`, N=1195)
- `20260421_1709_rag_hyde_cluster-vllm_31b-full-matrix` (`31b-full-matrix`, N=1195)
- `20260421_1726_subagent_rag_cluster-vllm_26b-subagent-1` (`26b-subagent-1`, N=1195)
- `20260421_1730_subagent_hybrid_cluster-vllm_26b-subagent-2` (`26b-subagent-2`, N=1195)
- `20260421_1758_rag_hyde_cluster-vllm_e2b-full-matrix-redo` (`e2b-full-matrix-redo`, N=1195)
- `20260421_1833_rag_hyde_cluster-vllm_26b-seed99-repeat` (`26b-seed99-repeat`, N=1195)
- `20260421_1839_snap_only_in_final_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1902_rag_simple_cluster-vllm_e4b-n200-prompt-fix` (`e4b-n200-prompt-fix`, N=200)

### G.2 Empty-retrieval contaminated runs from local Mac

- `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50` (`api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL`, N=50) — `legal_passages` collection was empty locally; 50/50 rows had empty retrieval.

### G.3 Smoke / test runs

- `20260421_0229_rag_hyde_cluster-vllm_smoke-31b`
- `20260425_2344_llm_only_or-gemma4-26b_api-smoke`
- `20260425_2354_llm_only_or-gemma4-26b_api-musique-smoke2`
- `20260426_1720_planning_table_or-gemma4-26b_api-musique-ptable-smoke`
- `20260426_1923_llm_only_groq-qwen_api-smoke-groq-qwen`
- `20260426_1923_llm_only_groq-llama70b_api-smoke-groq-llama70b`
- `20260426_1923_llm_only_groq-kimi_api-smoke-groq-kimi`
- `20260426_1923_llm_only_groq-scout_api-smoke-groq-scout`
- `20260426_1925_llm_only_groq-kimi_api-smoke-groq-kimi-v2`
- `20260426_1925_llm_only_groq-scout_api-smoke-groq-scout-v2`
- `20260426_1925_llm_only_groq-llama70b_api-smoke-groq-llama70b-v2`
- `20260426_1925_llm_only_groq-qwen_api-smoke-groq-qwen-v2`
- `20260426_1935_llm_only_groq-qwen_api-smoke-qwen-thinkfix`
- `20260426_2044_rag_multi_query_or-gemma4-26b_api-musique-multiquery-smoke`
- `20260426_2203_iterative_planning_table_or-gemma4-26b_api-musique-iter-ptable-smoke`
- `20260426_2206_advisor_planning_table_or-gemma4-26b_api-musique-advisor-smoke`
- `20260426_2246_multi_hyde_diverse_or-gemma4-26b_api-musique-multi-hyde-div-gemma-smoke`
- `20260426_2258_multi_hyde_diverse_or-gemma4-26b_api-musique-multi-hyde-div-gemma-smoke2`
- `20260427_0012_iter_hyde_groq-llama70b_api-musique-iter-hyde-llama-smoke`
- `20260427_0134_friend_foe_attribution_or-gemma27b_friend-foe-smoke`
- `20260427_0300_iter_hyde_or-gemma27b_bug-fix-smoke`
- `20260427_0301_multi_hyde_diverse_or-gemma27b_bug-fix-smoke`

### G.4 Zero-call API failures

- `20260426_1917_llm_only_groq-llama70b_api-cross-llama70b` (`api-cross-llama70b`, N=100) — summary has 0 correct, 0 avg LLM calls, 0 input/output tokens.
- `20260426_1917_llm_only_deepseek_api-cross-deepseek` (`api-cross-deepseek`, N=100) — summary has 0 correct, 0 avg LLM calls, 0 input/output tokens.

---

## Section H — Top-1 retrieval-depth ablation (audited 2026-04-28)

Scope: Llama 70B Groq x MuSiQue x N=200 paired top-1 vs top-5 retrieval-depth ablation, seed=42. The `--retrieval-k` CLI flag landed in commit `b286279`; audit doc `docs/audits/2026-04-28_top1_ablation_audit.md` verifies all top-1 rows have exactly one `evidence_store` item and one `retrieved_ids` item, with exact 200-row `idx` intersections against the top-5 baselines.

| Method | Top-5 detail log | Top-1 detail log | Paired N | Top-5 EM | Top-1 EM | Delta | McNemar p | Audit | Sign-off |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` | 200 | 27.5% | 13.0% | -14.5pp | 4.176981747e-07 | MINOR; `retrieval_k=1` proof clean, but 23/200 abstention-like predictions and one runaway/truncated output; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `rag_multi_query` | `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl` | `logs/eval_rag_multi_query_groq-llama70b_20260428_0029_detail.jsonl` | 200 | 29.0% | 14.0% | -15.0pp | 5.299581744e-06 | MINOR; `retrieval_k=1` proof clean, but 25/200 abstention-like predictions and one runaway/truncated output; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `rag_snap_hyde` | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | `logs/eval_rag_snap_hyde_groq-llama70b_20260428_0025_detail.jsonl` | 200 | 24.0% | 14.0% | -10.0pp | 0.001193242962 | MINOR; `retrieval_k=1` proof clean, no obvious final truncation, but 27/200 abstention-like predictions; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `multi_hyde_diverse` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260428_0019_detail.jsonl` | 200 | 35.5% | 19.0% | -16.5pp | 5.417768989e-07 | MINOR; `retrieval_k=1` proof clean, no obvious final truncation, but 20/200 abstention-like predictions; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |

Citation guidance: cite as a clean retrieval-depth ablation with caveat that top-1 is an under-context stress test and materially increases abstention-like predictions. Do not frame the lower top-1 EM as a harness/retrieval-k failure; the audit proves top-1 retrieval was applied on 800/800 top-1 rows.

---

## Section I — Choice-aware retrieval probes (2026-05-13)

Scope: `or-gemma4-26b`, q20 seed-42 retrieval-only diagnostics for
LegalBench-SCALR, CaseHOLD, and BarExamQA. These rows are probe-only and are not
paper-facing result claims. Use them to decide q50/q100 follow-ups, not as final
method comparisons.

| Dataset | Detail log | Health | Sign-off |
|---|---|---|---|
| LegalBench-SCALR | `logs/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q20_k10_tuned.jsonl` | 140 rows; 0 errors; 0 parse failures; 0 answer-artifact rows; 0 empty retrieval rows | ⚠️ PROBE-ONLY |
| CaseHOLD | `logs/choice_aware_retrieval_casehold_or-gemma4-26b_q20_k10_explicit.jsonl` | 160 rows; 0 errors; 0 parse failures; 0 answer-artifact rows; 0 empty retrieval rows | ⚠️ PROBE-ONLY |
| BarExamQA | `logs/choice_aware_retrieval_barexam_or-gemma4-26b_q20_k10_combined.jsonl` | 160 combined rows; 0 errors; 0 parse failures; 0 answer-artifact rows; 0 empty retrieval rows | ⚠️ PROBE-ONLY |

Provider note: the first BarExam `snap_choice_hyre` attempt hit an OpenRouter
upstream DekaLLM 401 and stopped under `NO_SILENT_FALLBACK=1`. The completed
BarExam `snap_choice_hyre` rows were rerun with
`OPENROUTER_PROVIDER_IGNORE=dekallm`; OpenRouter fallback remained disabled.

Summary and interpretation: `docs/choice_aware_retrieval_probe_2026-05-13.md`.

---

## Section J - Choice-aware retrieval q50 follow-up (2026-05-14)

Scope: `or-gemma4-26b`, q50 seed-42 retrieval-only diagnostics for
LegalBench-SCALR and CaseHOLD. These rows are probe-only and are not
paper-facing result claims. Use them to decide method selection and downstream
q20/q50 checks, not as final comprehensive comparisons.

| Dataset | Detail log | Health | Qrel alignment | Sign-off |
|---|---|---|---|---|
| LegalBench-SCALR | `logs/choice_aware_retrieval_legalbench_scalr_or-gemma4-26b_q50_k10_combined.jsonl` | 300 rows; 0 errors; 0 parse failures; 0 fallback rows; 0 answer-artifact rows; 0 empty retrieval rows; 0 think-tag rows | 50/50 unique gold ids found | ⚠️ PROBE-ONLY |
| CaseHOLD | `logs/choice_aware_retrieval_casehold_or-gemma4-26b_q50_k10_combined.jsonl` | 300 rows; 0 errors; 0 parse failures; 0 fallback rows; 0 answer-artifact rows; 0 empty retrieval rows; 0 think-tag rows | 50/50 unique gold ids found | ⚠️ PROBE-ONLY |

Provider note: q50 OpenRouter runs fixed the model id at
`google/gemma-4-26b-a4b-it` and used
`OPENROUTER_PROVIDER_IGNORE=dekallm,deepinfra` for the successful continuation
chunks. OpenRouter model fallback remained disabled. Future retries may route
across same-model OpenRouter providers if explicit and logged; silent changes
to model id, method, prompt, or cache are still invalid.

Summary and interpretation: `docs/choice_aware_retrieval_q50_2026-05-14.md`.

---

## Section K - Incremental comprehensive signoff (2026-05-20)

Scope: full-corpus exact-scored rows promoted after the 2026-05-20 overnight
dashboard refresh. These rows use the active comprehensive grid conventions in
`current_status.md` and the source logs named below.

Additional cache hydration:

| Dataset / scope | Cache evidence | Health | Sign-off |
|---|---|---|---|
| HousingQA raw/golden doc cache | `caches/retrieval_doc/full/housing_qfull_seed42_raw_golden_k10_doc_cache.jsonl`; built from `caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl` and `caches/retrieval/full/housing_qfull_seed42_golden_neighbors_k10.jsonl` with `--include-effective --include-gold-dataset housing --strict` | hydrated 4500/4500 distinct docs, missing=0; verification found 4500 doc-cache rows, 4500 needed retrieval/golden ids, 0 missing, 0 extras; intended for strict replay of remaining Housing raw/golden answer rows without reopening the full Housing collection | ✅ RETRIEVAL-DOC-CACHE-CLEAN |

Full answer rows:

| Dataset | Provider | Mode | Detail log | Accuracy | Health | Sign-off |
|---|---|---|---|---:|---|---|
| Legal-Link-EU | `or-gemma4-26b` | `rag_rewrite` | `logs/eval_rag_rewrite_or-gemma4-26b_20260519_224936_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-rag_rewrite-nfull-k5_detail.jsonl` | 855/1127 = 75.9% | dynamic rewrite retrieval, no deterministic retrieval cache; explicit OpenRouter route pinned/logged as `OPENROUTER_PROVIDER_ONLY=Parasail` on all 1127 rows for `google/gemma-4-26b-a4b-it`; `CROSS_ENCODER_MAX_CHARS=22000` persisted on all rows, CE doc truncation 0 and query truncation 0; retrieved/evidence length 5 on all rows, 613/1127 gold retrieved, Hit@5 0.5439 / MRR@5 0.4312, source_doc retrieved on 467/1127 rows and target_doc on 351/1127; rewrite JSON parse kind `json` on all rows, 0 rewrite-format retries, 0 partial-JSON repairs, 0 rewrite fallbacks; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 2.00, max output 2002 tokens, max final-answer chars 6339 at `complex_legallink_32011R0045_32010R0950_repeals__row0827`; custom no-silent audit: provider/mode/dataset exactly `or-gemma4-26b`/`rag_rewrite`/`legal_link_eu`, exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 3 valid same-model answer-format retries, 1 output row at >=1900 tokens with intact final line; by subject: completes 126/161, corrects 108/161, extends_application 131/161, extends_validity 114/161, implicitly_repeals 117/161, rendered_obsolete_by 143/161, repeals 116/161; paired vs `llm_only`: +1.69pp, b/c=140/121, p=0.265; paired vs `rag_simple`: -2.66pp, b/c=126/156, p=0.0840; paired vs `rag_hyde`: +1.69pp, b/c=138/119, p=0.261; paired vs `golden_plus_neighbors`: -18.90pp, b/c=30/243, p=1.32e-42 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE |
| Legal-Link-EU | `or-gemma4-26b` | `snap_hyre` | `logs/merged/eval_snap_hyre_or-gemma4-26b_20260520_legal_link_eu_nfull_k5_merged_detail.jsonl` | 883/1127 = 78.3% | strict Snap-HyRE generation/retrieval-cache replay; merged clean prefix `logs/eval_snap_hyre_or-gemma4-26b_20260519_204857_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-snap_hyre-nfull-k5_detail.jsonl` with Novita tail `logs/eval_snap_hyre_or-gemma4-26b_20260519_211957_legal_link_eu_local-snap-hyre-or-gemma4-26b-legal_link_eu-snap_hyre-nfull-k5_detail.jsonl` using `scripts/merge_detail_logs.py --on-duplicate last`, producing 1127 unique rows; provider route is same model `google/gemma-4-26b-a4b-it` with prefix `OPENROUTER_PROVIDER_ONLY=Parasail` and tail `OPENROUTER_PROVIDER_ONLY=Novita`; tail logged one transient same-route upstream 429 retry and continued without model fallback; 1127/1127 HyRE cache hits and retrieval-cache hits, retrieved/evidence length 5 on all rows, 765/1127 gold retrieved, Hit@5 0.6788 / MRR@5 0.5473, source_doc retrieved on 603/1127 rows and target_doc on 497/1127; `CROSS_ENCODER_MAX_CHARS=22000` persisted on all rows, CE doc truncation 0 and query truncation 0; `scripts/analyze_detail_flags.py`: rows 1127, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.01, max output 1828 tokens, max final-answer chars 5814 at `complex_legallink_32011R0045_32010R0950_repeals__row0827`; custom no-silent audit: exact final `Answer: (X)` lines on all rows, truthy fallback rows 0, think tags 0, 11 valid same-model answer-format retries, 0 near-cap rows; paired vs `llm_only`: +4.17pp, b/c=162/115, p=0.00562; paired vs `rag_simple`: -0.18pp, b/c=134/136, p=0.951; paired vs `rag_hyde`: +4.17pp, b/c=149/102, p=0.00361; paired vs `rag_rewrite`: +2.48pp, b/c=143/115, p=0.0926; paired vs `golden_plus_neighbors`: -16.42pp, b/c=28/213, p=2.07e-36 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/SAME-MODEL-ROUTE |
| HousingQA | `groq-llama8b` | `rag_rewrite` | `logs/merged/eval_rag_rewrite_groq-llama8b_20260520_housing_nfull_k5_merged_detail.jsonl` | 4342/6853 = 63.4% | dynamic rewrite retrieval, no deterministic retrieval cache; merged failed-closed prefix `logs/eval_rag_rewrite_groq-llama8b_20260520_034432_housing_local-snap-hyre-groq-llama8b-housing-rag_rewrite-nfull-k5_detail.jsonl`, same-model/same-evidence one-row repair `logs/eval_rag_rewrite_groq-llama8b_20260520_035859_housing_local-snap-hyre-groq-llama8b-housing-rag_rewrite-repairone-nfull-k5_detail.jsonl`, and clean tail `logs/eval_rag_rewrite_groq-llama8b_20260520_035956_housing_local-snap-hyre-groq-llama8b-housing-rag_rewrite-nfull-k5_detail.jsonl` with `scripts/merge_detail_logs.py --on-duplicate last`, producing 6853 unique rows; the prefix stopped on `hqa_California_5349` because `NO_SILENT_FALLBACK=1` rejected an `Answer: Unknown` response with `missing_required_final_answer_line`, then the retry prompt was hardened and the repaired row superseded the failed row; runner stdout logged `NO_SILENT_FALLBACK=1`, `LLM_MAX_COMPLETION_TOKENS=2048`, `EVAL_MIN_COMPLETION_TOKENS=2048`, and `EVAL_FINAL_FORMAT_RETRY=1`; provider route `{}` on all rows for `llama-3.1-8b-instant`; retrieved/evidence length 5 on all rows, 702/6853 gold retrieved, Hit@5 0.1024 / Recall@5 0.0693 / MRR@5 0.0601; rewrite JSON parse kind `json` on all 6853 rows, 0 rewrite-format retries, 0 partial-JSON repairs, 0 rewrite fallbacks; CE query truncation 0 and CE doc truncation rows 3515, a reranker-input caveat only; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 2.05, max output 2127 cumulative tokens, max final-answer chars 5317 at `hqa_Virginia_5218`; custom no-silent audit: exact final `Answer: Yes/No` lines on all rows, truthy fallback rows 0, think tags 0, 357 valid same-model answer-format retries, 0 single-call near-cap rows, 0 retry-near-cap rows; paired vs `llm_only`: +7.98pp, b/c=1695/1148, p=9.45e-25; paired vs `rag_simple`: +6.08pp, b/c=1340/923, p=1.80e-18; paired vs `rag_hyde`: +5.02pp, b/c=1322/978, p=7.77e-13; paired vs `snap_hyre`: +5.91pp, b/c=1379/974, p=6.89e-17 | ⚠️ COMPREHENSIVE-CITE-WITH-REPAIR/RETRY/RERANKER-TRUNCATION-CAVEAT |
| HousingQA | `groq-llama70b` | `rag_rewrite` | `logs/merged/eval_rag_rewrite_groq-llama70b_20260520_housing_nfull_k5_merged_detail.jsonl` | 3827/6853 = 55.8% | dynamic rewrite retrieval, no deterministic retrieval cache; merged clean prefix `logs/eval_rag_rewrite_groq-llama70b_20260520_060948_housing_local-snap-hyre-groq-llama70b-housing-rag_rewrite-nfull-k5_detail.jsonl` and clean tail `logs/eval_rag_rewrite_groq-llama70b_20260520_090221_housing_local-snap-hyre-groq-llama70b-housing-rag_rewrite-nfull-k5_detail.jsonl` with `scripts/merge_detail_logs.py --on-duplicate last`, producing 6853 unique rows; this is an unfiltered national-corpus HousingQA row, not a state-filtered row; provider route `{}` on all rows for `llama-3.3-70b-versatile`; retrieved/evidence length 5 on all rows, 674/6853 gold retrieved, Hit@5 0.0984 / Recall@5 0.0674 / MRR@5 0.0593; rewrite JSON parse kind `json` on all 6853 rows, 0 rewrite-format retries, 0 partial-JSON repairs, 0 rewrite fallbacks; CE query truncation 0 and CE doc truncation rows 3304, a reranker-input caveat only; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 2.00, max output 819 tokens, max final-answer chars 4388 at `5861`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_rewrite`/`housing`, exact final `Answer: Yes/No` lines on all rows, truthy fallback rows 0, think tags 0, 21 valid same-model answer-format retries, 0 single-call near-cap rows, 0 retry-near-cap rows; run stdout ended with a post-OK shell syntax message after the detail log and `logs/experiments.jsonl` summary were written, but `bash -n scripts/local/run_answer_cell.sh` now passes and the merged source log passed independent audits; paired vs `llm_only`: +11.09pp, b/c=1606/846, p=7.08e-54; paired vs `rag_simple`: +8.55pp, b/c=1247/661, p=1.55e-41; paired vs `rag_hyde`: -3.02pp, b/c=695/902, p=2.45e-07; paired vs `snap_hyre`: +3.43pp, b/c=928/693, p=5.82e-09; paired vs `golden_passage`: -11.44pp, b/c=610/1394, p=2.65e-70; paired vs `golden_plus_neighbors`: -10.17pp, b/c=600/1297, p=8.51e-59 | ⚠️ COMPREHENSIVE-CITE-WITH-RETRY/RERANKER-TRUNCATION/POST-OK-WRAPPER-CAVEAT |
| HousingQA state-filtered | `groq-llama8b` | `rag_simple` | `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` | 4269/6853 = 62.3% | jurisdiction-filtered raw retrieval row with `housing_state_filter=true` and per-row `retrieval_where={"state": ...}`; strict raw retrieval/doc-cache replay: 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 2532/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`: Hit@5 0.3695 / Recall@5 0.2413 / MRR@5 0.2330, Hit@10 0.4811 / Recall@10 0.3370 / MRR@10 0.2480; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max output 2051 tokens, max final-answer chars 8986 at `3781`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_simple`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, evidence-length failures 0, 73 output rows at >=1900 tokens with intact final lines; paired vs `llm_only`: +6.92pp, b/c=1336/862, p=4.14e-24, CI [5.59, 8.23] | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/NEAR-CAP-CAVEAT |
| HousingQA state-filtered | `groq-llama8b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl` | 4047/6853 = 59.1% | jurisdiction-filtered HyDE row with `housing_state_filter=true` and per-row `retrieval_where={"state": ...}`; strict HyDE/retrieval/doc-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 1847/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl`: Hit@5 0.2695 / Recall@5 0.1737 / MRR@5 0.1688, Hit@10 0.3450 / Recall@10 0.2333 / MRR@10 0.1787; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max output 2051 tokens, max final-answer chars 6212 at `hqa_North Carolina_5043`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`rag_hyde`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, HyRE-cache misses 0, evidence-length failures 0, 144 valid same-model answer-format retries with max retry output 3 tokens, 98 first-call near-cap rows with intact final lines; paired vs `llm_only`: +3.68pp, b/c=1286/1034, p=1.83e-07, CI [2.28, 5.05]; paired vs state-filtered `rag_simple`: -3.24pp, b/c=923/1145, p=1.15e-06, CI [-4.57, -1.94] | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY/NEAR-CAP-CAVEAT |
| HousingQA state-filtered | `groq-llama8b` | `snap_hyre` | `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl` | 4043/6853 = 59.0% | jurisdiction-filtered Snap-HyRE row with `housing_state_filter=true` and per-row `retrieval_where={"state": ...}`; runner stdout logged `OK dataset=housing provider=groq-llama8b mode=snap_hyre` and wrote `logs/experiments.jsonl`; strict Snap-HyRE generation/retrieval/doc-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 2026/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`: Hit@5 0.2956 / Recall@5 0.1886 / MRR@5 0.1888, Hit@10 0.3696 / Recall@10 0.2496 / MRR@10 0.1986; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.02, max output 2051 tokens, max final-answer chars 5357 at `836`; custom no-silent audit: provider/mode/dataset exactly `groq-llama8b`/`snap_hyre`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, HyRE-cache misses 0, evidence-length failures 0, 140 valid same-model answer-format retries with max retry output 3 tokens, 0 retry-near-cap rows, 0 single-call near-cap rows; CE doc truncation appears on 3382 rows as a reranker-input caveat, not final-answer truncation; paired vs `llm_only`: +3.62pp, b/c=1277/1029, p=2.63e-07, CI [2.25, 4.99]; paired vs state-filtered `rag_simple`: -3.30pp, b/c=910/1136, p=6.39e-07, CI [-4.58, -2.03]; paired vs state-filtered `rag_hyde`: -0.06pp, b/c=987/991, p=0.946, CI [-1.34, 1.20] | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY/RERANKER-TRUNCATION-CAVEAT |
| HousingQA state-filtered | `groq-llama70b` | `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl` | 4258/6853 = 62.1% | jurisdiction-filtered raw retrieval row with `housing_state_filter=true` and per-row `retrieval_where={"state": ...}`; strict raw retrieval/doc-cache replay: 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 2532/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`: Hit@5 0.3695 / Recall@5 0.2413 / MRR@5 0.2330, Hit@10 0.4811 / Recall@10 0.3370 / MRR@10 0.2480; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 770 tokens, max final-answer chars 4101 at `hqa_Georgia_2331`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_simple`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, evidence-length failures 0, answer-format retries 0, near-cap rows 0; runner stdout logged `OK dataset=housing provider=groq-llama70b mode=rag_simple` and wrote `logs/experiments.jsonl`, then printed a post-OK shell EOF message from the wrapper, but `bash -n scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh` now passes and the full detail log passed independent audits; paired vs `llm_only`: +17.38pp, b/c=1707/516, p=4.63e-148, CI [16.08, 18.66]; paired vs state-filtered `snap_hyre`: +2.50pp, b/c=816/645, p=8.49e-06, CI [1.39, 3.59] | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/POST-OK-WRAPPER-CAVEAT |
| HousingQA state-filtered | `groq-llama70b` | `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` | 4263/6853 = 62.2% | jurisdiction-filtered HyDE row with `housing_state_filter=true` and per-row `retrieval_where={"state": ...}`; strict HyDE/retrieval/doc-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 2395/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl`: Hit@5 0.3495 / Recall@5 0.2277 / MRR@5 0.2260, Hit@10 0.4257 / Recall@10 0.2896 / MRR@10 0.2362; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 734 tokens, max final-answer chars 4009 at `243`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`rag_hyde`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, HyRE-cache misses 0, evidence-length failures 0, 2 valid same-model answer-format retries with max retry output 3 tokens and 0 retry-near-cap rows, near-cap rows 0; CE doc truncation appears on 3395 rows as a reranker-input caveat, not final-answer truncation; runner stdout logged `OK dataset=housing provider=groq-llama70b mode=rag_hyde` and wrote `logs/experiments.jsonl`, then printed the same post-OK shell EOF message seen on prior Groq wrapper rows; independent detail, cache, and state-filter audits passed; paired vs `llm_only`: +17.45pp, b/c=1724/528, p=3.02e-147, CI [16.17, 18.75]; paired vs state-filtered `rag_simple`: +0.07pp, b/c=719/714, p=0.916, CI [-1.01, 1.18]; paired vs state-filtered `snap_hyre`: +2.57pp, b/c=758/582, p=1.69e-06, CI [1.52, 3.60] | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY/RERANKER-TRUNCATION/POST-OK-WRAPPER-CAVEAT |
| HousingQA state-filtered | `groq-llama70b` | `snap_hyre` | `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl` | 4087/6853 = 59.6% | jurisdiction-filtered Snap-HyRE row merged from failed-closed prefix `logs/eval_snap_hyre_groq-llama70b_20260520_143106_housing_local-snap-hyre-groq-llama70b-housing-snap_hyre-nfull-k5_detail.jsonl` and clean tail `logs/eval_snap_hyre_groq-llama70b_20260520_172759_housing_local-snap-hyre-groq-llama70b-housing-snap_hyre-nfull-k5_detail.jsonl` with `scripts/merge_detail_logs.py --on-duplicate last`, producing 6853 unique rows and replacing the previously blocked false-positive no-silent artifact row; strict Snap-HyRE generation/retrieval/doc-cache replay: 6853/6853 HyRE cache hits, 6853/6853 retrieval-cache hits, 6853/6853 hydrated document-cache hits, retrieved/evidence length 5 on all rows, 0 empty evidence rows, 1584/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`: Hit@5 0.2311 / Recall@5 0.1473 / MRR@5 0.1427, Hit@10 0.3130 / Recall@10 0.2079 / MRR@10 0.1535; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 747 tokens, max final-answer chars 4194 at `5575`; custom no-silent audit: provider/mode/dataset exactly `groq-llama70b`/`snap_hyre`/`housing`, provider route `{}` on all rows, exact final `Answer: Yes/No` lines on all rows, fallback keys 0, think tags 0, state-filter missing rows 0, cache misses 0, HyRE cache misses 0, evidence-length failures 0, near-cap rows 0; paired vs `llm_only`: +14.88pp, b/c=1679/659, p=6.68e-102, CI [13.53, 16.20] | ✅ COMPREHENSIVE-CLEAN-STATEFILTER |
| HousingQA state-filtered q500 diagnostic | `or-gemma4-26b` | `snap_hyre_exemplar` | `logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260521_023301_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre_exemplar-n500-k5_detail.jsonl` | 314/500 = 62.8% | q500 diagnostic only, not a full-N core row; explicit same-model OpenRouter route `OPENROUTER_PROVIDER_ONLY=Cloudflare` for `google/gemma-4-26b-a4b-it`; runner stdout logged `NO_SILENT_FALLBACK=1`, `EVAL_HOUSING_STATE_FILTER=1`, `REQUIRE_RETRIEVAL_CACHES=1`, `RETRIEVAL_DOC_CACHE_STRICT=1`, `LLM_MAX_COMPLETION_TOKENS=2048`, and exact q500 state-filter doc-cache path; generation cache resolved through the fixed state-filter exemplar fallback to `caches/generation/probes/housing_q500_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage.jsonl`; strict retrieval/doc/HyRE-cache replay: 500/500 retrieval-cache hits, 500/500 document-cache hits, 500/500 HyRE cache hits, evidence length 5 on all rows, 206/500 gold retrieved; retrieval cache audit from `caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl`: Hit@5 0.4120 / Recall@5 0.2738 / MRR@5 0.2647, versus canonical q500 `snap_hyre` Hit@5 0.3820 / MRR@5 0.2429; `scripts/analyze_detail_flags.py`: rows 500, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 1425 tokens, max final-answer chars 6472 at `1238`; `scripts/audit_housing_statefilter_detail.py --provider or-gemma4-26b --mode snap_hyre_exemplar --expected-rows 500 --require-hyre-cache`: wrong provider/mode/dataset 0, missing state filter 0, retrieval/doc/HyRE cache misses 0, bad evidence length 0, missing exact final answer 0, fallback 0, think tags 0; paired answer comparison vs canonical q500 `snap_hyre` (`logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl`): -0.20pp, b/c=36/37, p=1.0, CI [-3.6, 3.005]. Interpretation: retrieval-lift and answer-parity diagnostic; scale-eligible after required core rows, not a replacement for full-N core rows. | ⚠️ DIAGNOSTIC-Q500-CLEAN-STATEFILTER/PARITY-NOT-LIFT |

### HousingQA Gemma state-filter finalization - 2026-05-21T18:53:26Z

| HousingQA state-filtered | `or-gemma4-26b` | `rag_simple` | `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl` | 4531/6853 = 66.1% | jurisdiction-filtered `rag_simple` row with `housing_state_filter=true` and strict cache replay; rows 6853/6853, retrieved/evidence length 5 on 6853/6853 rows, 2532/6853 gold retrieved, Hit@5 0.3695 / MRR@5 0.2330; health counters: errors 0, missing predictions 0, state-filter misses 0, retrieval-cache misses 0, doc-cache misses 0, HyRE-cache misses 0, missing exact final answers 0, fallback rows 0, think tags 0, answer-format retries 13, provider-route misses 0, near-cap rows 0, max output 1973 tokens, max final chars 8310 at `hqa_South Dakota_3188` | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY-CAVEAT |
| HousingQA state-filtered | `or-gemma4-26b` | `rag_hyde` | `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl` | 4456/6853 = 65.0% | jurisdiction-filtered `rag_hyde` row with `housing_state_filter=true` and strict HyDE/retrieval/doc-cache replay; explicit OpenRouter route `OPENROUTER_PROVIDER_ONLY=Cloudflare` on all 6853 rows for `google/gemma-4-26b-a4b-it`; rows 6853/6853, retrieved/evidence length 5 on all rows, 2099/6853 gold retrieved; retrieval cache audit from `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`: Hit@5 0.3063 / Recall@5 0.2042 / MRR@5 0.1964, Hit@10 0.4013 / Recall@10 0.2751 / MRR@10 0.2090; `scripts/analyze_detail_flags.py`: rows 6853, errors 0, missing predictions 0, parse failures 0, empty retrieval rows 0, long rows 0, fallback markers 0, avg LLM calls 1.00, max output 2024 tokens, max final-answer chars 8950 at `hqa_South Dakota_1660`; `scripts/audit_housing_statefilter_detail.py`: wrong provider/mode/dataset 0, missing state filter 0, retrieval/doc/HyRE cache misses 0, bad evidence length 0, missing exact final answer 0, fallback 0, think tags 0; custom route/length audit: provider route Cloudflare 6853/6853, answer-format retries 6, near-cap rows 6 with intact final lines, CE doc truncation 0, CE query truncation 0; paired vs state-filtered `rag_simple`: -1.09pp, b/c=640/715, p=0.0444 | ⚠️ COMPREHENSIVE-CITE-STATEFILTER/RETRY/NEAR-CAP-CAVEAT |

2026-05-23 completion audit: no additional HousingQA `or-gemma4-26b` full-N
answer row is promoted. `llm_only` remains partial at 3680/6853 rows,
`snap_hyre` remains partial at 3942/6853 rows, and no HousingQA Gemma
`golden_passage` detail log was found. The full-N `snap_hyre` state-filter
retrieval cache remains citable only as retrieval exposure: Hit@5 0.3807 /
MRR@5 0.2452 from
`caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`.
See `docs/housingqa_gemma26_completion_audit_2026-05-23.md`.
