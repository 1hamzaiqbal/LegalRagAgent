# CLAUDE.md

## Update 2026-05-12 (Snap-HyRE comprehensive pivot)

**Current north star**: fixed-method Snap-HyRE, not a bottleneck-aware adaptive
controller. The active branch is `snap_hyre_comprehensive`, tracking
`shrango/snap_hyre_comprehensive`. The goal is to evaluate one straightforward
Snap-HyRE method across four legal benchmarks, with retrieval exposure as a
first-class outcome and downstream answer accuracy as the harder second
outcome.

**Start with these current docs**:
- `docs/snap_hyre_comprehensive_plan_2026-05-12.md` - active plan for four
  legal benchmarks, three models, top-k selection, retrieval caches, method
  ladder, and launch discipline.
- `docs/snap_hyre_experiment_runbook_2026-05-12.md` - concrete method ladder,
  cache workflow, validation gate, launch order, and open questions.
- `docs/literature_snap_hyre_2026-05-12.md` - notes from L-MARS /
  LegalSearchQA, Zheng et al. BarExamQA/HousingQA, and LRAGE.
- `docs/top_k_prelaunch_probe_2026-05-14.md` - current shared-k decision:
  use k=5 for main answer sweeps, with k=1..10 retrieval curves as analysis.
- `docs/comprehensive_run_status_2026-05-14.md` - live comprehensive-run ledger
  with completed rows, blocked rows, cache health, and current launch gates.
- `docs/README.md` - current documentation map and archive locations.
- `docs/signoff_log.md` - cite-or-not gate for any reported result.
- `docs/compiled_results.md` and `logs/experiments.jsonl` - historical ledger
  and machine-readable summaries.

**Active method story**:
- Primary row: `snap_hyre`, with `rag_snap_hyde_2call` kept as the legacy
  alias/provenance name.
- Main controls: `llm_only`, `rag_simple`, `rag_hyde`, `golden_passage`,
  `golden_plus_neighbors`, and `rag_rewrite`.
- Main benchmarks: BarExamQA, HousingQA, CaseHOLD, LegalBench-SCALR. HousingQA
  stays unless smoke/audit evidence shows the yes/no format is not
  interpretable for the fixed-method story.
- Current launch priority is LegalBench-SCALR, then BarExamQA and CaseHOLD.
  HousingQA remains in scope but should be deferred behind those three unless
  a specific comparison needs it.
- Main models: API-only small-model replacement, Gemma 4 26B, and Llama 3.3
  70B Versatile.
- Execution default: use API providers for all three current comprehensive axes:
  `or-ministral-8b` for the small-model row, `or-gemma4-26b` for Gemma 4 26B,
  and `groq-llama70b` for Llama 3.3 70B. Historical Gemma 4 E4B provenance used
  `cluster-vllm` with `LLM_MODEL=google/gemma-4-E4B-it`; keep those rows as
  historical evidence, but do not make vLLM a launch requirement for the current
  API-only comprehensive package. Do not substitute `or-gemma3n-e4b`; that is
  Gemma 3n E4B, not the historical Gemma 4 E4B checkpoint.
- Main metrics: downstream accuracy, Hit/Recall@1/5/10, MRR@10, gold retrieved
  but wrong, gold missing but correct, conditional accuracy, calls, tokens,
  latency, and health status.
- Shared answer depth: `RETRIEVAL_K=5` for the comprehensive answer grid. Use
  k=1..10 retrieval-cache curves for retrieval exposure analysis; reserve k=10
  answer runs for later targeted ablation, not as a launch blocker.

**Run-control rules**:
- Do not launch broad method sweeps. Keep at most 2-3 active jobs.
- Use generation caching (`scripts/build_generation_cache.py` for new full
  caches, or `scripts/build_hyre_cache.py` for older detail-log extraction, plus
  `--hyre-cache-path`) before repeated HyDE/Snap-HyRE answer runs.
- Use retrieval-id caches (`scripts/build_retrieval_cache.py`,
  `scripts/audit_retrieval_cache.py`, and `--retrieval-cache-path`) before
  large top-k sweeps where possible.
- Run `scripts/audit_retrieval_id_alignment.py` before treating Hit@k/MRR as
  valid retrieval-exposure claims; some datasets may have gold labels that are
  not Chroma document ids for the active collection.
- Use `scripts/compile_retrieval_cache_matrix.py` for top-k selection tables and
  `scripts/hpc/slurm_snap_hyre_answer_sweep.sh` for one deliberate
  dataset/model answer cell at a time.
- Every promoted row must have copied detail logs, `analyze_detail_flags.py`,
  retrieval scoring where gold ids exist, and a signoff entry.
- Full-corpus means every available row. If provider budget or rate limits
  block a full row, mark it as blocked/provisional rather than substituting an
  unannounced cap.
- Local answer cells must use `LLM_MAX_COMPLETION_TOKENS>=2048`; the local
  runner fails closed below `EVAL_MIN_COMPLETION_TOKENS` so stale `.env` caps
  cannot silently truncate answer runs.
- Answer wrappers default `EVAL_FINAL_FORMAT_RETRY=1`; this is a logged
  same-model/same-evidence retry only for malformed final answer formatting.
  When the prior response already produced a parseable prediction, the retry
  is format-only and must preserve that prediction instead of reopening the
  full reasoning task.
- For discrete answer tasks, the exact last non-empty final answer line is the
  scoring source of truth. If it is exactly `Answer: (X)` or `Answer: Yes/No`,
  that value overrides earlier answer mentions. If it is absent, the same-model
  final-answer retry must repair the row before it can pass
  `NO_SILENT_FALLBACK`.
- Generation-cache wrappers default `EVAL_GENERATION_FORMAT_RETRY=1`; this is
  a logged same-model retry only for malformed generated answer/passage blocks.
  It must not change the intended provider, method, cache scope, question set,
  or model label.
- Snap-HyRE generation is strict about both blocks: the generated passage must
  parse and the snap block must contain an exact required final answer line
  (`Answer: (X)` or dataset-specific equivalent). With `NO_SILENT_FALLBACK=1`,
  missing snap final lines are blockers, not harmless metadata.
- `rag_rewrite` validates query-rewriter JSON and logs `rewrite_parse_ok` plus
  rewrite-format retry metadata. With `NO_SILENT_FALLBACK=1`, malformed rewrite
  JSON that is not repaired by the same-model format retry fails before
  retrieval instead of becoming raw-question RAG.
- `rag_rewrite` also has an explicit partial-JSON repair for malformed rewrite
  outputs that contain parseable model-generated `primary` / `alternatives`
  strings but fail to close as valid JSON. This is logged as
  `rewrite_parse_kind=partial_json` and `rewrite_partial_json_repair=true`; it
  is not a raw-question fallback and must be cited as a repair caveat when used.
- Answer wrappers require `NO_SILENT_FALLBACK` to be truthy and fail before
  launch if it is disabled.
- HousingQA full golden-neighbor cache uses the explicit
  `retrieval_backend=stored_gold_embedding` path plus
  `CROSS_ENCODER_MAX_CHARS=4096` for cross-encoder-only reranking inputs. This
  avoided local OOM on the 1.8M-document Chroma index and is recorded in the
  cache metadata; do not describe it as arbitrary text re-embedding of the gold
  passage.
- As of 2026-05-15, all three full-corpus SCALR `llm_only` rows are signed off
  clean: `groq-llama70b` 425/571, `or-gemma4-26b` 417/571, and
  `or-ministral-8b` 384/571.
- The full SCALR `groq-llama70b` `rag_simple` row is 416/571 = 72.9%, below
  `llm_only` 425/571 = 74.4% by -1.58pp (McNemar p=0.328). It used 571/571
  raw retrieval-cache hits, retrieved gold on 283/571 rows, and has raw
  retrieval Hit@5 0.4956 / MRR@5 0.3447. The row is clean: zero errors,
  missing predictions, parse failures, fallback keys, exact-final-line issues,
  empty retrieval rows, answer retries, long rows, or near-cap outputs.
- The full SCALR `groq-llama70b` `golden_passage` oracle row is 534/571 =
  93.5%, +20.67pp over `rag_simple` (McNemar p=1.82e-34) and +19.09pp over
  `llm_only` (p=8.63e-32). Cite with the retry caveat: 8 explicit same-model
  final-answer repairs, all 5-token repairs and none near the 2048-token cap.
- The full SCALR `groq-llama70b` `golden_plus_neighbors` row is 474/571 =
  83.0%, +10.16pp over `rag_simple` (McNemar p=2.56e-10) and +8.58pp over
  `llm_only` (p=1.41e-08), but -10.51pp versus `golden_passage`
  (p=2.73e-17). Cite with the retry caveat: 2 explicit same-model final-answer
  repairs, both 5-token repairs and none near the 2048-token cap. This repeats
  the SCALR pattern that adding retrieved neighbors can dilute the gold-only
  oracle.
- As of 2026-05-15, the full SCALR `groq-llama70b` `rag_hyde` and
  `snap_hyre` generation/retrieval caches are signed off for answer replay.
  Both have 571/571 generated rows with zero errors, missing passages,
  fallbacks, parse failures, answer-artifact passages, or retries. Retrieval
  metrics: `rag_hyde` Hit@5 0.6147 / Hit@10 0.6953 / MRR@10 0.5015;
  `snap_hyre` Hit@5 0.5517 / Hit@10 0.6462 / MRR@10 0.4126. On Groq SCALR,
  `snap_hyre` improves over raw question Hit@5 0.4956 but is below `rag_hyde`.
- The full SCALR `groq-llama70b` `rag_hyde` answer row is 402/571 = 70.4%,
  below `rag_simple` 416/571 = 72.9% (-2.45pp, McNemar p=0.0925) and below
  `llm_only` 425/571 = 74.4% (-4.03pp, p=0.00140). It used 571/571 HyDE and
  retrieval-cache hits and retrieved gold on 351/571 rows. The row is clean:
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, answer retries, empty retrieval rows, long rows, or
  near-cap outputs. Cite as retrieval-positive but answer-negative.
- The full SCALR `groq-llama70b` `snap_hyre` answer row is 407/571 = 71.3%,
  below `rag_simple` 416/571 = 72.9% (-1.58pp, McNemar p=0.281) and below
  `llm_only` 425/571 = 74.4% (-3.15pp, p=0.0222), but slightly above
  `rag_hyde` 402/571 = 70.4% (+0.88pp, p=0.542). It used 571/571 HyRE and
  retrieval-cache hits and retrieved gold on 315/571 rows. The row is clean:
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, answer retries, empty retrieval rows, long rows, or
  near-cap outputs. Cite as retrieval-positive over raw but answer-negative.
- The full SCALR `groq-llama70b` `rag_rewrite` answer row is 409/571 = 71.6%,
  below `rag_simple` 416/571 = 72.9% (-1.23pp, McNemar p=0.450) and below
  `llm_only` 425/571 = 74.4% (-2.80pp, p=0.0365), but slightly above
  `snap_hyre` 407/571 = 71.3% (+0.35pp, p=0.890) and `rag_hyde` 402/571 =
  70.4% (+1.23pp, p=0.382). The row is clean with a retry caveat: 571/571
  rewrite JSON parses, zero rewrite retries, zero partial-JSON repairs, zero
  fallback keys, zero exact-final-line issues, 5 explicit same-model
  final-answer repairs, and no near-cap repairs. Dynamic rewrite retrieval
  exposure is Hit@5 0.5762 / MRR@5 0.4327, above raw retrieval but below Groq
  `rag_hyde`.
- As of 2026-05-16, the full CaseHOLD `groq-llama70b` `rag_hyde` and
  `snap_hyre` generation/retrieval caches are signed off for answer replay.
  Both have 3600/3600 generated rows with zero errors, missing passages,
  fallbacks, parse failures, answer-artifact passages, or think artifacts.
  `snap_hyre` initially had two malformed snap final-line metadata rows
  (`ch_test_1108`, `ch_test_3118`); they were regenerated with the same
  provider/model and merged under the stricter no-silent-fallback guard before
  signoff. Retrieval metrics: `rag_hyde` Hit@5 0.5122 / Hit@10 0.5914 /
  MRR@10 0.4090; `snap_hyre` Hit@5 0.4497 / Hit@10 0.5289 / MRR@10 0.3390.
  Both are far above raw CaseHOLD question retrieval Hit@5 0.1794, but
  `rag_hyde` is stronger than `snap_hyre` on this retrieval-only check.
- As of 2026-05-16, the full BarExamQA `groq-llama70b` `rag_hyde` and
  `snap_hyre` generation/retrieval caches are signed off for answer replay.
  Both generation caches have 1195/1195 rows with zero errors, missing
  passages, fallback keys, parse failures, or answer-artifact passages. The
  retrieval caches are clean with zero duplicate keys, missing indices, empty
  rows, short rows, or rows without gold. `rag_hyde` retrieval is Hit@5 0.1046
  / Hit@10 0.1757 / MRR@10 0.0609; `snap_hyre` retrieval is Hit@5 0.1105 /
  Hit@10 0.1849 / MRR@10 0.0663. Compared with BarExamQA raw retrieval
  Hit@5 0.0142 / MRR@5 0.0068, both are retrieval-positive, with `snap_hyre`
  slightly above `rag_hyde`.
- The full BarExamQA `groq-llama70b` `llm_only` row is 940/1195 = 78.7% and
  clean: zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, answer retries, long rows, or near-cap outputs. The
  `or-gemma4-26b` BarExam `llm_only` partial log from 2026-05-16 stopped at
  9/1195 rows as an explicit pacing probe and is not citable.
- The full BarExamQA `or-gemma4-26b` `llm_only` row is 966/1195 = 80.8%.
  It has an explicit same-model route caveat: a clean 51-row OpenRouter prefix
  was merged with a 1144-row `OPENROUTER_PROVIDER_ONLY=Cloudflare` tail on
  `google/gemma-4-26b-a4b-it`. The failed DekaLLM 401 row `mbe_60` is excluded
  and superseded after `NO_SILENT_FALLBACK` blocked it. The merged row is clean:
  1195 rows, zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think-tag artifacts, or long-answer rows. There are
  3 valid answer-format retries and 4 rows at >=1900 output tokens, all with
  intact final `Answer:` lines; one naive fallback-text hit is incidental legal
  explanation text, not provider/method fallback.
- The full BarExamQA `or-gemma4-26b` `rag_simple` row is 932/1195 = 78.0%,
  below its `llm_only` row by -2.85pp (McNemar b/c=78/112, p=0.0164). It used
  strict raw retrieval-cache replay with 1195/1195 cache hits, 0 empty
  retrieval rows, and 17/1195 gold retrieved. Retrieval exposure is Hit@5
  0.0142 / MRR@5 0.0068 from
  `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_simple.md`. Cite
  with the retry caveat: 3 logged same-model answer-format retries
  (`mbe_576`, `mbe_989`, `mbe_1124`), all with intact final `Answer:` lines;
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, or think tags.
- The full BarExamQA `or-gemma4-26b` `golden_passage` row is 939/1195 =
  78.6%. It is answer-flat versus strict `rag_simple` (+0.59pp, McNemar
  b/c=102/95, p=0.669) and directionally below `llm_only` (-2.26pp,
  b/c=78/105, p=0.0543), despite oracle gold retrieval on 1195/1195 rows.
  Cite with the retry/near-cap caveat: 4 logged same-model answer-format
  retries (`mbe_83`, `mbe_312`, `mbe_625`, `mbe_989`), 4 original outputs at
  >=1900 tokens with intact final `Answer:` lines, max output 2023 tokens, and
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think tags, or long rows.
- The full BarExamQA `or-gemma4-26b` `golden_plus_neighbors` row is 964/1195 =
  80.7%. It improves over strict `rag_simple` by +2.68pp (McNemar b/c=116/84,
  p=0.0281), is flat versus `llm_only` (-0.17pp, b/c=91/93, p=0.941), and is
  directionally above `golden_passage` (+2.09pp, b/c=91/66, p=0.0551). Cite
  with the retry/near-cap caveat: strict golden-neighbor cache replay retrieved
  gold on 1195/1195 rows, 5 logged same-model answer-format retries
  (`mbe_532`, `mbe_562`, `mbe_989`, `mbe_1131`, `mbe_563`), 3 rows at >=1900
  output tokens with intact final `Answer:` lines, max output 2007 tokens, and
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think tags, or empty retrieval rows.
- The full BarExamQA `or-gemma4-26b` `rag_hyde` row is 959/1195 = 80.3%.
  It is directionally above strict `rag_simple` by +2.26pp (McNemar b/c=113/86,
  p=0.0650), flat versus `llm_only` (-0.59pp, b/c=94/101, p=0.668), flat
  versus `golden_plus_neighbors` (-0.42pp, b/c=94/99, p=0.773), and
  directionally above `golden_passage` (+1.67pp, b/c=113/93, p=0.185). Cite
  with the retry/near-cap caveat: strict HyDE/retrieval cache replay used
  1195/1195 HyDE cache hits and retrieval-cache hits, retrieved list length 5
  on all rows, 136/1195 gold retrieved, retrieval exposure Hit@5 0.1138 /
  MRR@5 0.0542, 5 logged same-model answer-format retries (`mbe_141`,
  `mbe_291`, `mbe_576`, `mbe_899`, `mbe_989`), 5 rows at >=1900 output tokens
  with intact final `Answer:` lines, max output 2103 tokens, and zero errors,
  missing predictions, parse failures, fallback keys, exact-final-line issues,
  think tags, or empty retrieval rows.
- The full BarExamQA `or-gemma4-26b` `snap_hyre` row is 980/1195 = 82.0%.
  It is positive over strict `rag_simple` by +4.02pp (McNemar b/c=121/73,
  p=0.000699), directionally above `rag_hyde` by +1.76pp (b/c=84/63,
  p=0.0987), and directionally above `llm_only` by +1.17pp (b/c=103/89,
  p=0.348). Cite with the retry/near-cap/transient caveat: strict HyRE and
  retrieval cache replay used 1195/1195 HyRE cache hits and retrieval-cache
  hits, retrieved list length 5 on all rows, 144/1195 gold retrieved,
  retrieval exposure Hit@5 0.1205 / MRR@5 0.0602, 3 logged same-model
  answer-format retries (`mbe_989`, `mbe_1131`, `mbe_288`), 4 rows at >=1900
  output tokens with intact final `Answer:` lines, max output 2025 tokens, and
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think tags, or empty retrieval rows. Visible
  Cloudflare 502/upstream idle retries recovered in-place under the pinned same
  provider/model and did not substitute models or methods.
- The full BarExamQA `or-gemma4-26b` `rag_rewrite` row is 964/1195 = 80.7%.
  It improves over strict `rag_simple` by +2.68pp (McNemar b/c=126/94,
  p=0.0364), is tied with `golden_plus_neighbors` (b/c=98/98, p=1.000),
  directionally above `rag_hyde` by +0.42pp (b/c=88/83, p=0.760), flat versus
  `llm_only` (-0.17pp, b/c=102/104, p=0.944), and below `snap_hyre` by
  -1.34pp (b/c=75/91, p=0.244). Cite with the retry/near-cap/transient caveat:
  dynamic rewrite retrieval had 1195/1195 rewrite JSON parses, zero
  rewrite-format retries, zero partial-JSON repairs, zero raw-question
  fallbacks, 146/1195 gold retrieved, and retrieval exposure Hit@5 0.1222 /
  MRR@5 0.0604 from
  `docs/generated/retrieval_qrels_barexam_or-gemma4-26b_rag_rewrite.md`. The
  row has 4 logged same-model answer-format retries (`mbe_501`, `mbe_763`,
  `mbe_989`, `mbe_486`), 3 rows at >=1900 total output tokens after retry,
  max output 2099 tokens, max final-answer chars 7618, and zero errors, missing
  predictions, parse failures, fallback keys, exact-final-line issues, think
  tags, long rows, empty retrieval rows, rewrite parse failures, or partial JSON
  repairs. Visible Cloudflare upstream idle retries recovered in-place under the
  pinned same provider/model and did not substitute models or methods.
- The full BarExamQA `or-ministral-8b` `llm_only` row is 679/1195 = 56.8%.
  It has no retrieval evidence by design and is signed with a retry/near-cap
  caveat rather than as a fully clean row: 1195 rows, zero errors, missing
  predictions, parse failures, fallback keys, exact-final-line issues, think
  tags, or long rows; 74 logged same-model answer-format retries; 9 rows at
  >=1900 total output tokens, with retried near-cap rows ending in a short exact
  final `Answer:` line and non-retried near-cap rows retaining intact final
  `Answer:` lines. Max output is 2070 total tokens after retry accounting, max
  final-answer length is 9911 chars at `mbe_266`, and average calls are 1.06.
- The full BarExamQA `or-ministral-8b` `rag_simple` row is 680/1195 = 56.9%,
  answer-flat versus `llm_only` (+0.08pp, McNemar b/c=156/155, p=1.000). It
  used strict raw retrieval-cache replay with 1195/1195 cache hits, retrieved
  list length 5 on all rows, 0 empty evidence rows, and 17/1195 gold retrieved;
  retrieval exposure is Hit@5 0.0142 / MRR@5 0.0068. Cite with a retry/near-cap
  caveat: 27 valid same-model answer-format retries, 22 rows at >=1900 total
  output tokens, all near-cap rows were retry rows ending with short exact
  final `Answer:` lines, and there were zero errors, missing predictions, parse
  failures, fallback keys, exact-final-line issues, think tags, or long rows.
- The full BarExamQA `or-ministral-8b` `golden_passage` row is 772/1195 =
  64.6%, +7.70pp over strict `rag_simple` (McNemar b/c=205/113,
  p=2.78e-07) and +7.78pp over `llm_only` (b/c=206/113, p=2.14e-07).
  It is an oracle-gold row with caveats: 1195/1195 gold retrieved, retrieved
  list length 1 on all rows, zero errors, missing predictions, parse failures,
  fallback keys, exact-final-line issues, think tags, or analyzer long rows;
  37 valid same-model answer-format retries; 34 rows at >=1900 output tokens;
  max output 2112 tokens; and one verbose non-retried row `mbe_233` at 10173
  final-answer chars with intact final `Answer: (C)`.
- The full BarExamQA `or-ministral-8b` `golden_plus_neighbors` row is
  755/1195 = 63.2%, +6.28pp over strict `rag_simple` (McNemar b/c=205/130,
  p=4.93e-05), +6.36pp over `llm_only` (b/c=199/123, p=2.70e-05), and
  -1.42pp versus `golden_passage` (b/c=111/128, p=0.301). It used strict
  golden-neighbor retrieval-cache replay with 1195/1195 cache hits, retrieved
  list length 5 on all rows, and 1195/1195 gold retrieved. Cite with a
  retry/near-cap caveat: 30 valid same-model answer-format retries, 28 rows at
  >=1900 total output tokens, zero runner near-cap failures under the 2048-token
  margin, max output 2087 tokens, max final-answer chars 9848, and zero errors,
  missing predictions, parse failures, fallback keys, exact-final-line issues,
  think tags, or long rows.
- The full BarExamQA `groq-llama70b` `rag_simple` row is 891/1195 = 74.6%,
  significantly below `llm_only` by -4.10pp (McNemar b/c=66/115, p=0.000334).
  It is operationally clean under strict raw retrieval-cache replay: 1195/1195
  cache hits, zero empty retrieval rows, zero errors, missing predictions, parse
  failures, fallback keys, or exact-final-line issues. Cite with the retry
  caveat: two explicit same-model final-answer repairs (`mbe_272`, `mbe_202`),
  retry max 5 tokens, no near-cap repairs, max output 1169 tokens. Exact-gold
  retrieval exposure is very low: Hit@5/Recall@5 0.0142 and MRR@5 0.0068.
- The full BarExamQA `groq-llama70b` `golden_plus_neighbors` row is 930/1195 =
  77.8%. It improves over `rag_simple` by +3.26pp (McNemar b/c=136/97,
  p=0.0126), but remains slightly below `llm_only` by -0.84pp (b/c=97/107,
  p=0.529). This is a clean answer row with a retry caveat: 1195/1195
  golden-neighbor cache hits, 1195/1195 gold retrieved, zero errors, missing
  predictions, parse failures, fallback keys, exact-final-line issues, empty
  retrieval rows, or near-cap repairs; two explicit same-model final-answer
  repairs (`mbe_273`, `mbe_1098`). Retrieval exposure is oracle by construction
  at Hit@5/MRR@5 1.0000.
- The full BarExamQA `groq-llama70b` `golden_passage` row is 946/1195 = 79.2%.
  It improves over `rag_simple` by +4.60pp (McNemar b/c=137/82, p=0.000246),
  is flat versus `llm_only` at +0.50pp (b/c=100/94, p=0.720), and is
  directionally above `golden_plus_neighbors` by +1.34pp (b/c=76/60,
  p=0.198). This is a clean oracle row with a retry caveat: 1195/1195 rows had
  gold injected/retrieved, zero errors, missing predictions, parse failures,
  fallback keys, exact-final-line issues, empty retrieval rows, or near-cap
  repairs; three explicit same-model final-answer repairs (`mbe_2`, `mbe_197`,
  `mbe_1125`), retry max 5 tokens, max output 1029 tokens.
- The full BarExamQA `groq-llama70b` `rag_hyde` row is 958/1195 = 80.2%, now
  the best signed BarExamQA Llama 70B row in this comprehensive pass. It is
  significantly above strict `rag_simple` by +5.61pp (McNemar b/c=137/70,
  p=3.73e-06), directionally above `llm_only` by +1.51pp (b/c=101/83,
  p=0.210), and directionally above `golden_passage` by +1.00pp (b/c=106/94,
  p=0.437). This is a clean generated-cache replay row with a retry caveat:
  1195/1195 HyDE cache hits, 1195/1195 retrieval-cache hits, 125/1195 gold
  retrieved, zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, empty retrieval rows, or near-cap repairs; three
  explicit same-model final-answer repairs (`mbe_90`, `mbe_95`, `mbe_131`),
  retry max 5 tokens, max output 1061 tokens.
- The full BarExamQA `groq-llama70b` `snap_hyre` row is 953/1195 = 79.7%.
  It is significantly above strict `rag_simple` by +5.19pp (McNemar
  b/c=138/76, p=2.70e-05), directionally above `llm_only` by +1.09pp
  (b/c=103/90, p=0.388), and statistically flat/slightly below `rag_hyde` by
  -0.42pp (b/c=79/84, p=0.754). It is also directionally above
  `golden_passage` by +0.59pp (p=0.681) and `golden_plus_neighbors` by
  +1.92pp (p=0.124). The row is clean with a retry caveat: 1195/1195 HyRE
  cache hits, 1195/1195 retrieval-cache hits, 132/1195 gold retrieved, zero
  errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, empty retrieval rows, answer artifacts, or near-cap
  outputs; one explicit same-model final-answer repair (`mbe_2`), retry max
  5 tokens, max output 1265 tokens.
- The full BarExamQA `groq-llama70b` `rag_rewrite` row is 923/1195 = 77.2%.
  It improves over strict `rag_simple` by +2.68pp (McNemar b/c=133/101,
  p=0.0425), but is directionally below `llm_only` by -1.42pp (b/c=102/119,
  p=0.282), below `rag_hyde` by -2.93pp (b/c=84/119, p=0.0168), below
  `snap_hyre` by -2.51pp (b/c=85/115, p=0.0400), and directionally below
  `golden_passage` by -1.92pp (b/c=105/128, p=0.149). The row is clean with a
  retry/repair caveat: 1195/1195 rewrite JSON parses, zero rewrite-format
  retries, one logged partial-JSON repair, zero fallback keys, zero
  exact-final-line issues, 11 explicit same-model final-answer repairs, and no
  near-cap repairs. Dynamic rewrite retrieval exposure is Hit@5 0.1222 /
  MRR@5 0.0565, above raw BarExamQA retrieval but still a low absolute
  exact-gold-recall regime.
- The full CaseHOLD `groq-llama70b` `llm_only` row is 2585/3600 = 71.8%.
  It is signed with an explicit retry caveat: zero errors, missing
  predictions, parse failures, fallback keys, exact-final-line issues, long
  rows, or near-cap outputs; 39 logged same-model final-answer repairs, all
  `missing_marker` and 5-token repair outputs; max output 774 tokens and max
  final-answer chars 3845. This is the current CaseHOLD no-retrieval anchor
  for the priority benchmark set.
- The full CaseHOLD `or-gemma4-26b` `llm_only` row is 2614/3600 = 72.6%,
  directionally above the Llama 70B `llm_only` row by +0.81pp (McNemar
  b/c=356/327, p=0.284). Cite with the explicit retry/near-cap caveat: no
  retrieval evidence by design, provider/mode/dataset exactly
  `or-gemma4-26b`/`llm_only`/`casehold`, exact final `Answer: (X)` lines on
  all rows, zero errors, missing predictions, parse failures, fallback keys,
  think tags, or long rows; 24 same-model answer-format retries and 19 rows at
  >=1900 total output tokens had intact final answer lines.
- The full CaseHOLD `groq-llama70b` `rag_simple` row is 2547/3600 = 70.8%,
  directionally below `llm_only` by -1.06pp (McNemar b/c=215/253, p=0.0871).
  It is signed with an explicit retry caveat: 3600/3600 raw retrieval-cache
  hits, 0 empty retrieval rows, retrieved list length 5 on every row, 646/3600
  gold retrieved, retrieval exposure Hit@5 0.1794 / MRR@5 0.1015; zero errors,
  missing predictions, parse failures, fallback keys, exact-final-line issues,
  long rows, or near-cap outputs; 23 logged same-model final-answer repairs,
  max output 935 tokens and max final-answer chars 4850. This establishes the
  strict raw-RAG CaseHOLD comparator for the remaining CaseHOLD methods.
- The full CaseHOLD `groq-llama70b` `golden_passage` row is 3511/3600 = 97.5%.
  It is strongly above `rag_simple` by +26.78pp (McNemar b/c=968/4,
  p=1.86e-282) and above `llm_only` by +25.72pp (b/c=927/1, p=8.19e-277).
  This is a clean oracle row with a retry caveat: 3600/3600 rows had gold
  injected/retrieved, retrieved list length 1 on every row, zero empty evidence
  rows, zero errors, missing predictions, parse failures, fallback keys, or
  final-line prediction issues; 46 explicit same-model final-answer repairs,
  max output 878 tokens, max final-answer chars 4845, and no near-cap repairs.
  Retrieval exposure is oracle by construction at Hit@1/Hit@5/MRR@5 1.0000.
- The full CaseHOLD `groq-llama70b` `golden_plus_neighbors` row is
  2859/3600 = 79.4%. It is significantly above strict `rag_simple` by +8.67pp
  (McNemar b/c=459/147, p=2.70e-38) and above `llm_only` by +7.61pp
  (b/c=411/137, p=8.67e-33), but far below `golden_passage` by -18.11pp
  (b/c=5/657, p=1.10e-187). This is a clean golden-neighbor replay row with a
  retry caveat: 3600/3600 cache hits, retrieved list length 5 and neighbor list
  length 4 on every row, 3600/3600 gold retrieved, zero empty evidence rows,
  zero errors, missing predictions, parse failures, fallback keys, or
  final-line prediction issues; 19 explicit same-model final-answer repairs,
  max output 943 tokens, max final-answer chars 4989, and no near-cap repairs.
  `call_trace.response` previews were clipped on 2794/3600 rows by
  `EVAL_TRACE_MAX_CHARS=800`, but full `final_answer` values were stored
  separately and ended with exact `Answer: (X)` lines on all rows. Retrieval
  exposure is oracle by construction at Hit@1/Hit@5/MRR@5 1.0000, while the
  downstream drop versus gold-only is a clear CaseHOLD neighbor-dilution signal.
- The full CaseHOLD `groq-llama70b` `rag_hyde` row is 2532/3600 = 70.3%.
  It has an explicit mixed same-model provider caveat: Groq produced 2639
  valid rows, then stopped on a spend-alert threshold; `ch_test_2639` plus
  960 tail rows were replayed through OpenRouter paid
  `meta-llama/llama-3.3-70b-instruct`, with the final 667-row segment pinned
  to `OPENROUTER_PROVIDER_ONLY=AkashML`. This was an explicit recovery, not a
  silent fallback. The merged row is clean under strict generated/retrieval
  cache replay: 3600/3600 HyDE and retrieval-cache hits, 0 empty retrieval
  rows, 1844/3600 gold retrieved, retrieval exposure Hit@5 0.5122 / MRR@5
  0.3983, 21 logged answer-format retries, and zero errors, missing
  predictions, parse failures, fallback keys, exact-final-line issues,
  think-tag artifacts, or near-cap outputs. Downstream is retrieval-positive
  but answer-flat/negative: -0.42pp vs `rag_simple` (p=0.535) and -1.47pp vs
  `llm_only` (p=0.0169).
- The full CaseHOLD `groq-llama70b` `snap_hyre` row is 2538/3600 = 70.5%.
  It has explicit mixed same-model provider and cache-repair caveats:
  OpenRouter paid same-model prefix supplied 581 rows while Groq spend was
  blocked; after the user reset Groq spend, repaired `ch_test_581` and rows
  `ch_test_582`-`ch_test_3599` ran on Groq, for provider mix
  `or-llama70b-paid` 581 / `groq-llama70b` 3019. The invalid OpenRouter
  `ch_test_581` answer row was excluded, and generation-cache row
  `ch_test_581` had only a formatting repair to standalone `Answer: (A)`.
  The merged row is clean under strict generated/retrieval cache replay:
  3600/3600 HyRE and retrieval-cache hits, 0 empty retrieval rows, retrieved
  list length 5 on every row, 1619/3600 gold retrieved, answer-log retrieval
  Hit@5 0.4497 / MRR@5 0.3286, 16 logged answer-format retries, and zero
  errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think-tag artifacts, or near-cap outputs. Downstream
  is flat vs strict raw RAG (-0.25pp, p=0.722), below `llm_only` (-1.31pp,
  p=0.0295), and flat/slightly above `rag_hyde` (+0.17pp, p=0.812).
- The full CaseHOLD `groq-llama70b` `rag_rewrite` row is 2542/3600 = 70.6%.
  It is a clean all-Groq dynamic rewrite row with 3600/3600 rewrite parses,
  zero rewrite retries, zero partial-JSON repairs, 0 empty retrieval rows, and
  1623/3600 gold retrieved. Retrieval exposure is Hit@5 0.4508 / MRR@5
  0.3319 from `docs/generated/retrieval_qrels_casehold_groq-llama70b_rag_rewrite.md`.
  The answer log has 88 explicit same-model answer-format retries and zero
  errors, missing predictions, parse failures, fallback keys, exact-final-line
  issues, think-tag artifacts, or near-cap outputs. Downstream is flat vs
  strict raw RAG (-0.14pp, p=0.859), flat vs `snap_hyre` (+0.11pp, p=0.890),
  flat vs `rag_hyde` (+0.28pp, p=0.675), and below `llm_only` (-1.19pp,
  p=0.0589).
- The full SCALR `or-ministral-8b` `rag_simple` strict rerun is 388/571 =
  68.0%, versus `llm_only` 384/571 = 67.3% (McNemar p=0.752). It used 571/571
  raw retrieval-cache hits and has raw retrieval Hit@5 0.4956 / MRR@5 0.3447.
  Cite with the retry caveat: 16 explicit same-model final-answer repairs,
  including 11 original responses at >=2000 output tokens. The earlier
  `20260515_082923` row is superseded because exact-final-line audit found six
  source-safety failures.
- The full SCALR `or-ministral-8b` `golden_passage` oracle row is 532/571 =
  93.2%, +25.22pp over strict `rag_simple` and +25.92pp over `llm_only`
  (both p < 1e-40). Cite with the retry caveat: 22 explicit same-model
  final-answer repairs, four with original responses at >=2000 output tokens.
- The full SCALR `or-ministral-8b` `golden_plus_neighbors` row is 440/571 =
  77.1%, +9.11pp over strict `rag_simple` and +9.81pp over `llm_only`, but
  -16.11pp versus `golden_passage`. Cite with the retry caveat: 5 explicit
  same-model final-answer repairs, three with original responses at >=2000
  output tokens. This supports the emerging SCALR read that extra neighbors can
  dilute gold-only evidence rather than improve the oracle control.
- As of 2026-05-15, the full SCALR `or-ministral-8b` `rag_hyde` and
  `snap_hyre` generation/retrieval caches are signed off for answer replay.
  Both have 571/571 generated rows with zero errors, missing passages,
  fallbacks, parse failures, answer-artifact passages, or retries.
  Retrieval metrics: `rag_hyde` Hit@5 0.6025 / Hit@10 0.6865 / MRR@10 0.4506;
  `snap_hyre` Hit@5 0.6200 / Hit@10 0.7040 / MRR@10 0.5110.
- The full SCALR `or-ministral-8b` `rag_hyde` answer row is 406/571 = 71.1%,
  +3.15pp over strict `rag_simple` (McNemar p=0.0385) and +3.85pp over
  `llm_only` (McNemar p=0.0230). It used 571/571 HyDE and retrieval-cache hits
  and retrieved gold on 344/571 rows. Cite with the retry caveat: 18 explicit
  same-model final-answer repairs, 14 with original responses at >=2000 output
  tokens.
- The full SCALR `or-ministral-8b` `snap_hyre` answer row is 399/571 = 69.9%,
  +1.93pp over strict `rag_simple` (McNemar p=0.260), +2.63pp over `llm_only`
  (p=0.110), and -1.23pp versus small-model `rag_hyde` (p=0.457). It used
  571/571 HyRE and retrieval-cache hits and retrieved gold on 354/571 rows.
  Retrieval exposure is positive versus raw question retrieval and above
  `rag_hyde`, but downstream answer accuracy is weaker than `rag_hyde`. Cite
  with the retry caveat: 9 explicit same-model final-answer repairs, 7 with
  original responses at >=2000 output tokens. A naive fallback-string scan found
  one incidental use of the legal word "fallback" in a generated CERCLA query;
  targeted fallback-key audit found zero fallback keys or provider/method
  substitution markers.
- The full SCALR `or-ministral-8b` `rag_rewrite` row is 399/571 = 69.9%,
  tied with `snap_hyre`, +1.93pp over strict `rag_simple` (McNemar p=0.228),
  +2.63pp over `llm_only` (p=0.119), and -1.23pp versus `rag_hyde` (p=0.470).
  Dynamic rewrite retrieval is positive over raw question retrieval:
  Hit@5 0.6497 / MRR@5 0.5185, with gold retrieved on 371/571 rows. Cite with
  the retry/repair caveat: 5 same-model rewrite-format retries, 1 explicit
  partial-JSON repair on `scalr_538`, and 31 same-model final-answer repairs.
  Three failed partial logs are superseded by the merged clean detail log:
  `scalr_110`, `scalr_431`, and `scalr_538`.
- As of 2026-05-15, the full SCALR `or-gemma4-26b` `snap_hyre`
  generation/retrieval cache is signed off for answer replay: generation is
  571/571 with one explicit same-model format retry on `scalr_273`, and
  retrieval improves over raw question retrieval from Hit@5 0.4956 to 0.7268.
- The corresponding full SCALR `or-gemma4-26b` `snap_hyre` answer row is
  422/571 = 73.9%, versus `rag_simple` 419/571 = 73.4% and `llm_only` 417/571
  = 73.0%; paired deltas are not significant. Cite this as a strong retrieval
  lift with answer-flat downstream behavior. Caveat: 10 explicit same-model
  answer-format retries occurred, including five original responses at
  >=2000/2048 output tokens before final-line repair.
- The full SCALR `or-gemma4-26b` `rag_hyde` row is 412/571 = 72.2%, below
  `rag_simple` and `snap_hyre` downstream despite clean generated/retrieval
  caches. Retrieval is still strongly positive over raw question retrieval:
  Hit@5 0.7075 vs raw 0.4956. Cite as retrieval-positive but answer-flat/weak,
  with the retry caveat: 8 explicit same-model answer-format retries, including
  three original responses at >=2000/2048 output tokens.
- The full SCALR `or-gemma4-26b` `rag_rewrite` row is 422/571 = 73.9%, tied
  with `snap_hyre` and +0.53pp over `rag_simple` downstream. The rewrite path
  had 571/571 valid rewrite JSON parses, 0 rewrite retries/fallbacks, and
  Hit@5 0.6743 / MRR@5 0.5212, so it is retrieval-positive over raw question
  retrieval but below `snap_hyre`/`rag_hyde` generated retrieval. Cite with the
  retry caveat: 9 explicit same-model final-answer format repairs, all valid
  same-prediction final-line repairs.
- The full HousingQA `groq-llama70b` `llm_only` row is 3067/6853 = 44.8%.
  It is a clean no-retrieval anchor: zero errors, missing predictions, parse
  failures, fallback keys, exact-final-line issues, think tags, answer retries,
  long rows, or near-cap outputs; all rows used provider/mode/dataset
  `groq-llama70b`/`llm_only`/`housing`, had retrieved list length 0 by design,
  and ended with exact `Answer: Yes/No` lines.
- The full HousingQA `groq-llama70b` `rag_simple` row is 3241/6853 = 47.3%,
  +2.54pp over `llm_only` (McNemar b/c=894/720, p=1.63e-05). It is clean under
  strict raw retrieval-cache replay: 6853/6853 cache hits, retrieved list length
  5 on all rows, 0 empty evidence rows, 193/6853 gold retrieved, retrieval
  exposure Hit@5 0.0282 / MRR@5 0.0148, zero errors, missing predictions, parse
  failures, fallback keys, exact-final-line issues, think tags, answer retries,
  long rows, or near-cap outputs.
- The full HousingQA `groq-llama70b` `golden_passage` row is 4611/6853 =
  67.3%, +19.99pp over `rag_simple` (McNemar b/c=1817/447, p=5.25e-195),
  +22.53pp over `llm_only` (b/c=1929/385, p=3.09e-246), and +1.27pp over
  `golden_plus_neighbors` (b/c=540/453, p=0.00632). Cite with the explicit
  retry caveat: oracle gold injected/retrieved on 6853/6853 rows,
  evidence-store length 1 on all rows, retrieval exposure Hit@5/MRR@5 1.0000,
  zero errors, missing predictions, parse failures, fallback keys,
  exact-final-line issues, think tags, long rows, or near-cap outputs; 18
  same-model final-answer format retries occurred.
- The full HousingQA `groq-llama70b` `golden_plus_neighbors` row is 4524/6853 =
  66.0%, +18.72pp over `rag_simple` (McNemar b/c=1702/419, p=1.17e-182) and
  +21.26pp over `llm_only` (b/c=1829/372, p=2.91e-230). It is clean with one
  retry caveat: strict golden-neighbor retrieval-cache replay, 6853/6853 cache
  hits, evidence-store length 5 on all rows, 0 empty evidence rows, 6853/6853
  gold retrieved, retrieval exposure Hit@5 1.0000 / MRR@5 1.0000, zero errors,
  missing predictions, parse failures, fallback keys, exact-final-line issues,
  think tags, long rows, or near-cap outputs; one same-model final-answer format
  retry occurred on row `4192`.
- LegalSearchQA is related work unless converted into a frozen corpus task.

**Archived pivot**: the May 9-11 diagnostic/adaptive-controller package was
archived under `docs/archive/diagnostic_adaptation_2026-05-12/` and
`scripts/archive/diagnostic_adaptation_2026-05-12/`. Use it for provenance, not
as the active branch narrative.

## Archived Update 2026-05-11 (diagnostic-adaptation meeting package)

**Current north star**: source-gated diagnostic adaptation for legal RAG. The
meeting package frames Snap-HyRE/HyRE as one intervention family inside a
bottleneck-aware controller: calibration traces route each benchmark toward
baseline RAG, legal query rewrite, Snap-HyRE/HyRE, state metadata filtering,
option grounding, verifier policies, disagreement arbitration, or
reject/escalate.

**Archived docs for this package**:
- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_prep_2026-05-11_diagnostic_adaptation.md`
- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_eval_expansion_status_2026-05-11.md`
- `docs/archive/diagnostic_adaptation_2026-05-12/meeting_package_audit_2026-05-11.md`
- `docs/signoff_log.md` remains the cite-or-not gate for rows from this sprint.

**Latest source-gated deltas**:
- Snap-only controls are complete across the four legal benchmarks with copied
  detail logs and `analyze_detail_flags.py` validation: BarExam 85.5%,
  HousingQA 55.0%, CaseHOLD 72.5%, and LegalBench-SCALR 72.5%, all at 2.00
  calls. CaseHOLD uses clean capped replacement `67867`; the earlier 74.0%
  row is superseded because it had a long-answer outlier.
- BarExam HyRE-only (`rag_hyde`) is a modest positive retrieval control at
  82.0%, above baseline retrieval but below snap-only reasoning and the
  stronger fixed Snap-HyRE v2 route.
- HousingQA HyRE-only (`rag_hyde`) is a clean negative control at 50.0%,
  below snap-only, state-filter retrieval, and the verifier route.
- CaseHOLD HyRE-only (`rag_hyde`) is a weak/negative control at 71.5%, below
  current baseline retrieval, snap-only, and diverse HyRE-family rows.
- HousingQA fixed Snap-HyRE (`rag_snap_hyde_2call`) is a clean negative
  control at 51.5%; CaseHOLD fixed Snap-HyRE is weak/negative at 72.0%;
  BarExam fixed Snap-HyRE lands at 84.5%, below snap-only and adaptive
  Snap-HyRE v2.
- Groq Llama 70B held-out sanity rows are mostly clean: Housing verifier and
  SCALR frontier transfer directionally, BarExam selected route underperforms
  its baseline slice, and CaseHOLD selected route is rejected due health-gate
  failures.
- SCALR HyRE-only uncapped completed at 71.0% but is rejected as a clean report
  row because one final answer ran away to 267,458 chars / 70,593 output
  tokens. Capped rerun `67864` completed the eval at 148/200 = 74.0% with a
  clean detail log, no long-answer rows, and one missing prediction; the SLURM
  wrapper failed only after results were written because
  `scripts/postprocess_adaptive_hyre_sweep.py` was missing. Cite as
  detail-log clean but wrapper-caveated.
- CaseHOLD direct option-table is repaired and clean but weak at 70.0% on the
  held-out slice; cite it as an option-conversion bottleneck signal, not as a
  positive route.

**Do not promote pending rows**: the capped full-SCALR replacement job `67897`
must remain pending until its stdout, detail logs, and local validation all pass. Use
`docs/archive/diagnostic_adaptation_2026-05-12/meeting_eval_expansion_status_2026-05-11.md`
for the archived queue state before citing anything from this sprint. The full-SCALR `rag_simple` half of `67863` is copied
locally and structurally complete at 424/571, but it has three long-answer rows
and is health-gated rather than reportable. The paired frontier half of `67863`
also hit a 232,797-character answer and was cancelled, so `67863` is not a
promoted full-corpus result.

**Run-control note**: CaseHOLD capped snap-only `67866` was cancelled after row
12 produced a 157,678-character answer, so it cannot replace the original
health-caveated snap-only row. `llm_config.py` now sends OpenRouter caps through
`extra_body={"max_tokens": ...}` because LangChain rewrites `max_tokens` to
`max_completion_tokens`; replacement `67867` landed cleanly at 145/200 with no
long-answer rows and is the current snap-only CaseHOLD source.

**N>=500 scale-up note**: canonical larger-slice jobs are now running under the
same gates: `67897` full-SCALR capped baseline/frontier, `67915` SCALR
rewrite retry after `67914` failed on `a40-2206`, `67911` BarExam
baseline/rewrite/adaptive-v2, `67912` Housing
state-filter/rewrite/verifier, and `67913` CaseHOLD baseline/rewrite/diverse.
Do not promote them until copied detail logs pass local validation.

## Update 2026-05-01 (meeting prep)

**Current meeting frame**: lead with bottleneck-typed retrieval, not a finished
new RAG recipe. `snap_hyde_2call` is a useful fixed-cost probe, but the forward
method hypothesis is adaptive snap-HyDE: use the HyDE/reasoning budget on top of
the active bottleneck, whether that is query framing, query diversity, candidate
depth, metadata filtering, or answer-option conversion. The defensible current
contribution is the diagnostic matrix that separates retrieval depth,
candidate-set size, query formulation, evidence use, metadata filtering, and
answer-option anchoring.

**Latest source-gated deltas**:
- CaseHOLD repaired cluster job `58283` landed: `rag_simple` 69.5% vs
  `rag_snap_hyde_2call` 72.0%, +2.5pp, b/c=16/11, p=0.4421. Gold retrieval is
  now meaningful for this pair and jumps 16.0% -> 47.0%, but answer accuracy is
  still not a reliable lift. See `docs/casehold_repaired_rerun_2026-05-01.md`.
- Housing state-filter job `58282` is invalid as a method result: both k=5 and
  k=10 were `_FAILED-EMPTY-RETRIEVAL`. Root cause was state-metadata casing
  (`California` vs `california`). The fixed k=5 run `58799` landed at 61.5%,
  and the chunked k=10 completion `58937` landed at 62.5% with 0 empty
  retrieval rows and 98/200 gold retrieved. Cite as a metadata-filtering signal,
  not as a generic deeper-retrieval result. See
  `docs/housing_state_filter_followup_2026-05-01.md`.
- Top-k sensitivity should be described as a retrieval-policy stress test or
  first-pass bottleneck signal, not a complete causal diagnosis.

Current citation gates: `docs/signoff_log.md`,
`docs/meeting_state_2026-05-01.md`, `docs/snap_hyde_2call_2026-04-28.md`,
`docs/top1_ablation_2026-04-28.md`, `docs/compiled_results.md`, and
`logs/experiments.jsonl`.

## Update 2026-04-28 (post-2026-04-27 meeting)

**Headline shift**: snap_hyde_2call is the new MuSiQue Llama 70b winner at **+9.5pp p=0.008** (vs prior multi_hyde_diverse +8.0pp p=0.0195). The paper-grade story has pivoted from "multi_hyde_diverse wins multi-hop" to a **bottleneck taxonomy** measurable via a single ablation: top-1 vs top-5 retrieval depth.

**Cleanest cross-dataset evidence** (added 2026-04-28):
- MuSiQue × Llama 70b rag_simple top-1 vs top-5 = **-14.5pp p=4.18e-07** (catastrophic — retrieval-bottlenecked)
- BarExam × Gemma 4 26B rag_simple top-1 vs top-5 = **-0.5pp p=1.00 NS** (depth-flat; full-corpus lift is better read as answer anchoring / evidence use)
- ~14pp gap is the bottleneck-taxonomy signature, method-independent.

**Mechanism finding** (new): the Llama-vs-Gemma snap_hyde_2call split on MuSiQue is explained by HyDE-passage gold-recall delta. Llama 70b's HyDE passages improve gold-hit by +2.5pp; Gemma 27B's degrade it by -7.5pp. Same parametric floor (snap_only_in_final: Llama 9.5%, Gemma 9.0%), opposite retrieval and EM outcomes.

**Open**: full-corpus N=2400 Llama 70b paired blocked by Groq RPD limit (1K/day vs 4800 calls needed). Cerebras would unlock it but no API key configured.

Historical citation gates for the 2026-04-28 pivot were
`docs/signoff_log.md`, `docs/snap_hyde_2call_2026-04-28.md`,
`docs/lit_review_2026-04-28.md`, `docs/top1_ablation_2026-04-28.md`, and
`docs/meeting_notes_042726.md`; prefer the 2026-05-01 gates above for current
meeting claims.

Source-of-truth context for working in this codebase. Verify claims against `main.py` before relying on them.

## Environment Note

`uv` may not be on PATH in every shell. Prefer `uv` when available, otherwise fall back to `~/.local/bin/uv`.

## Project Summary

Legal RAG research repo with two distinct surfaces:
- `main.py` = the full LangGraph agentic pipeline / demo system
- `eval/` = the current research harness, where adaptive retrieval variants are compared under a fixed evaluation setup

Current research direction: the original heavy pipeline underperformed, but the long-term goal is still a strong full agentic system. For now, the project rebuilds toward that goal **atomically**: simpler adaptive retrieval strategies are the default baseline, and extra structure only stays when it proves itself in `eval/eval_harness.py`.

## Key Documentation

- `docs/README.md` — concise documentation map. Start here when deciding
  which docs are current vs historical.
- `docs/meeting_state_2026-05-01.md` — meeting-ready state of findings,
  blockers, open jobs, and defensible interpretation.
- `docs/signoff_log.md` — cite-or-not gate for result claims.
- `docs/compiled_results.md` and `logs/experiments.jsonl` — audited ledger and
  machine-readable run summaries.
- `docs/benchmark_method_birdseye_2026-04-30.md` — benchmark/method map and
  harness coverage.
- `reports/final_class_report/main.pdf` and
  `reports/final_class_report/main.tex` — current class-report draft.
- `docs/housing_state_filter_followup_2026-05-01.md` and
  `docs/casehold_repaired_rerun_2026-05-01.md` — latest dataset-specific gates.
- `docs/hpc_setup_log.md` and `docs/cluster_workflow.md` — cluster paths,
  venvs, bad nodes, and launch workflow.
- `RESEARCH.md` and `EXPERIMENTS.md` — historical running logs, not current
  claim entrypoints.

## Runtime Architecture

Source of truth: `main.py`

### Graph

```
START → router_node → planner_node → parallel_executor_node → parallel_synthesizer_node
                          ↑                                          |
                          └── parallel_replanner_node ←──────────────┘ (if incomplete)
                                                                     └→ END (if complete)
```

### Nodes

**router_node** — Lightweight LLM call to choose which ChromaDB collection(s) to search. Current registry: `legal_passages`, `housing_statutes`. Falls back to `legal_passages`.

**planner_node** — Decomposes the question into `PlanningStep`s. Outputs:
- `complexity`: `"simple"` / `"moderate"` / `"complex"` (LLM decides)
- Steps: each with `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`
- Hard cap: 5 steps max. `max_retries` capped at 3.

Loads `skills/planner.md`. Falls back to a single-step plan on parse failure.

**parallel_executor_node** — Executes ALL pending steps, each with its own internal escalation chain via `_execute_step_with_escalation()`:

Per-step execution:
- `rag_search`: LLM query rewrite → multi-query retrieval → cited synthesis → judge
- `web_search`: DuckDuckGo → scrape top 2 URLs (trafilatura) → cited synthesis → judge
- `direct_answer`: LLM answers from doctrine (no retrieval) → judge

Per-step escalation (if judge says insufficient, up to `max_retries`):
- `rag_search` → rewrite query → `direct_answer` (web skipped for doctrinal queries)
- `web_search` → `direct_answer`

**parallel_synthesizer_node** — Aggregates all completed steps into an IRAC answer (`skills/synthesizer.md`), then runs a completeness check. If gaps identified, returns `missing_topics` and routes back to replanner. Max 3 rounds.

**parallel_replanner_node** — Creates new `PlanningStep`s from the synthesizer's `missing_topics` and feeds them back to the executor.

### Shared State (`LegalAgentState`)

- `agent_metadata` — provider, model, timestamps
- `inputs` — `{"question": "..."}`
- `run_config` — `{"max_steps": 7, "max_parallel_rounds": 3}`
- `collections` — chosen search collection(s), populated by `router_node`
- `planning_table` — list of `PlanningStep`
- `evidence_store` — accumulated retrieved passages (all steps)
- `final_answer` — synthesizer output
- `audit_log` — per-node trace entries with timestamps
- `completeness_verdict` — synthesizer's completeness check result
- `parallel_round` — current planner→executor→synthesizer iteration

`PlanningStep` fields: `step_id`, `sub_question`, `authority_target`, `retrieval_hints`, `action_type`, `max_retries`, `rewrite_attempt`, `status`, `result`, `confidence`, `evidence_ids`, `retry_of`, `judge_verdict`.

### Logging

Two modes controlled by `--verbose` CLI flag or `VERBOSE=1` env var:
- **Compact** (default): step breakdown with evidence source counts, LLM call/token totals
- **Verbose**: full passage text with cross-encoder scores, query rewrite alternatives, web search URLs and scraped content previews, sub-answer previews, per-LLM-call token counts

## Skills

4 prompt files in `skills/`, loaded by `main.py`:

| Skill file | Loaded as | Purpose |
|---|---|---|
| `planner.md` | `planner` | Decompose question into research steps |
| `query_rewriter.md` | `query_rewriter` | Rewrite sub-question into primary + 2 alternative queries (JSON) |
| `synthesize_and_cite.md` | `synthesize_and_cite` | Per-step cited synthesis with `[Source N]` format |
| `synthesizer.md` | `synthesizer` | Final IRAC synthesis with `[Evidence N]` citations |

## Retrieval Stack

Source of truth: `rag_utils.py`

- **Vector store**: ChromaDB persisted to `./chroma_db/`
  - `legal_passages`: 686,324 barexam passages
  - `housing_statutes`: 1,837,403 housing statutes
- **Embedding model**: `Alibaba-NLP/gte-large-en-v1.5` (1024d, 8192 tokens)
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid retrieval**: BM25 + bi-encoder candidates pooled, deduplicated by `idx`, cross-encoder reranks to top k
- **BM25 corpus cap**: BM25 skipped for collections >1M docs (housing_statutes uses dense-only)
- **Multi-query retrieval**: pools candidates across all query variants, deduplicates, reranks against primary query
- **Cross-step dedup**: `exclude_ids` filters out passages already retrieved in prior steps

## LLM Configuration

Source of truth: `llm_config.py`

- Provider selection via `LLM_PROVIDER` env var (default: `deepseek`)
- `get_llm()` returns a cached `ChatOpenAI` instance (LRU cache keyed on temperature + provider)
- `_llm_call()` adds retry handling (3 attempts for transient errors)
- Run `uv run python llm_config.py` to list all providers

## Commands

```bash
# Install
uv sync

# Configure
cp .env.example .env   # then add API keys

# Download datasets
uv run python utils/download_data.py                # BarExam QA
uv run python utils/download_housingqa.py           # HousingQA
uv run python utils/download_new_datasets.py        # CaseHOLD, Legal-RAG-QA, Australian Legal QA

# Build vector stores (GPU-optimized)
uv run python utils/fast_embed.py barexam           # Full barexam (~2.2 hr on RTX 3070)
uv run python utils/fast_embed.py housing           # Full housing (~6 hr on RTX 3070)
uv run python utils/fast_embed.py housing --resume  # Resume interrupted embedding
uv run python utils/fast_embed.py status            # Check collection sizes

# Run agent
uv run python main.py simple                        # Simple doctrinal question
uv run python main.py multi_hop                      # Multi-step reasoning
uv run python main.py medium                         # Medium complexity
uv run python main.py simple --verbose               # Verbose output

# Evals (all via eval_harness.py)
uv run python eval/eval_harness.py --mode llm_only --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100
uv run python eval/eval_harness.py --mode rag_snap_hyde --provider groq-llama70b --questions 100 --dataset housing
uv run python eval/eval_harness.py --mode golden_passage --provider groq-llama70b --questions 100

# List providers
uv run python llm_config.py

# Regression tests (sanitizer + snap stripping — lock in leak fixes)
uv run python tests/test_sanitizer.py
```

## Methodology integrity — read this BEFORE running new evals

Two harness bugs were patched on 2026-04-22 that invalidated all
prior BarExam numbers:

1. `f95f316` — `format_question_prompt` and `_fmt_intermediate` were
   reading `row["question"]` but never `row["prompt"]`. 445/1195
   BarExam rows (37%) carry a shared fact pattern in the `prompt`
   column; without it the model saw stems like
   `"Is Farmer obligated to make the $4,000 payment?"` (47 chars)
   with no facts.

2. `3d5ff05` — 11 retrieval/rerank call sites used
   `raw_question = str(row["question"])` (e.g., `snap_rag`, gap
   investigations, entity search). They also dropped the prompt
   column, so the vector store was being queried with the bare
   47-char stem too.

Every BarExam result before commit `3d5ff05` is a pre-prompt-fix
reference. Relative rankings (mode-vs-mode, size-vs-size) survive
because the bug hit all modes equally; absolute numbers do not.

Run `python tests/test_formatter.py` and `python tests/test_sanitizer.py`
before any new submission. The full pre-submission checklist lives in
`docs/rigour_signoff.md`.

## Result Snapshot / Direction Notes

**Source of truth**: `docs/signoff_log.md` for cite-or-not status,
`docs/meeting_state_2026-05-01.md` for current meeting wording,
`docs/compiled_results.md` for audit details, and
`docs/snap_hyde_2call_2026-04-28.md` /
`docs/top1_ablation_2026-04-28.md` for the bottleneck-taxonomy pivot. Numbers
below are post-fix BarExam Tier 3 or MuSiQue Tier 2 unless explicitly marked
direction-only.

### Llama 70b MuSiQue (Tier 2 N=200, current method vehicle)

| Mode | EM | Δ vs `rag_simple` | McNemar p | Verdict |
|---|---:|---:|---:|---|
| `rag_simple` | 27.5% | — | — | baseline |
| **`snap_hyde_2call`** | **37.0%** | **+9.5pp** | **0.0079** | **SIG; current MuSiQue vehicle** |
| `iterative_planning_table` | 36.0% | +8.5pp | 0.0533 | TRENDING-SIG |
| `multi_hyde_diverse` | 35.5% | +8.0pp | 0.0195 | SIG — superseded headline, still citeable |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | NS |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | NS |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | NS |
| `subagent_rag` | 15.5% | -12.0pp | 0.0007 | SIG negative |

### Gemma 4 26B-A4B BarExam (post-fix N=1195, 8/8 modes landed)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **81.17%** | **+3.09pp** ← winning method |
| `snap_only_in_final` | 80.59% | +2.51pp |
| `llm_only` | 79.75% | +1.67pp |
| `rag_hyde` | 78.91% | +0.83pp |
| `golden_passage` (oracle) | 78.66% | +0.58pp |
| `subagent_rag` | 78.16% | +0.08pp |
| `rag_simple` (baseline) | 78.08% | — |
| `subagent_hybrid` | 74.23% (rescored; stored 74.14%) | -3.85pp |

### Gemma 4 E4B BarExam (post-fix N=1195, 8 modes landed)

| Mode | EM | Δ vs `rag_simple` |
|---|---|---|
| `rag_snap_hyde` | **62.18%** | **+3.69pp** ← same winner as 26B |
| `subagent_rag` | 60.92% | +2.43pp |
| `snap_hyde_report` | 60.75% | +2.26pp |
| `rag_hyde` | 60.59% | +2.10pp |
| `subagent_hyde` | 60.17% | +1.68pp |
| `subagent_hybrid` | 58.83% | +0.34pp |
| `rag_simple` (baseline) | 58.49% | — |
| `snap_only_in_final` | 57.82% | -0.67pp |

### Cross-family BarExam llm_only N=100 board

| Model | EM | Architecture |
|---|---|---|
| Llama 3.3 70b | 81% | 70B dense |
| **Gemma 4 26B-A4B** | **79.75%** | **25B/3.8B-active MoE** (cluster N=1195) |
| Qwen3 30B MoE | 70% | 30B/3B-active MoE (N=100) |
| Qwen3 32b dense | 68% (caveat: 13/100 truncated mid-`<think>` at 2048-token cap; true likely 70-78%) | 32B dense |
| Gemma 3 27b | 68% | 27B dense |
| Llama 4 Scout 17b | 67% | 17B MoE |

### MuSiQue (multi-hop) — older direction-only rows

| Mode | Gemma 4 26B | Gemma 3 27B | Llama 70b |
|---|---|---|---|
| `rag_simple` (baseline) | 26.7% (N=30) | **22.0%** (N=100) | **21.0%** (N=100) |
| `rag_multi_query` | 23.3% (N=30) | — | 20.0% (N=30) |
| `planning_table_no_snap` v2 | 23.3% (N=30) | — | 20.0% (N=30) |
| `iterative_planning_table` | 20.0% (N=30) | — | 23.3% (N=30) |
| `advisor_planning_table` (cheap-plan) | 23.3% (N=30) | — | 23.0% (N=100, +2pp p=0.82, cost-parity not lift) |
| **`multi_hyde_diverse`** (direction-only N=100 context) | — | **30.0% (N=100, +8pp p=0.134 trending)** | **33.0% (N=100, +12pp p=0.023)** |
| `rag_snap_hyde` | 20.0% (N=30) | — | 13.3% (N=30) |
| `golden_passage` (oracle) | 62% (N=30) | — | 47% (N=30) |

These N=100/N=30 rows are preserved as direction-only context. They are superseded for confirmed claims by the Llama 70b N=200 table above and the Gemma 3 27B N=200 NULL.

`advisor_planning_table` is a cost-parity method (86% strong-LLM input-token / 43% output-token reduction vs `iter_ptable` at parity EM, audit `a5bbd0b5840ac0da6`), not an accuracy lift method.

### Working interpretation (current)

- **Current frame:** retrieval behavior is bottleneck-typed. MuSiQue is
  retrieval-depth sensitive; BarExam is depth-flat; CaseHOLD/SCALR mainly probe
  option disambiguation and answer anchoring.
- **`rag_snap_hyde` is the proven winner on legal MC**: +3-4pp over
  `rag_simple` at both E4B and 26B sizes (clean cross-size lift, not noise).
- **Multi-hop QA split:** `snap_hyde_2call` is the current MuSiQue Llama 70B
  winner, while the earlier `multi_hyde_diverse` lift remains useful as a
  mechanism/superseded-headline result.
- **Bug-fix decomposition**: formatter (`f95f316`) added +5.44pp at 26B llm_only/snap_only_in_final identically; retrieval-query (`3d5ff05`) added +1.85pp marginal on RAG modes
- **Showing snap answer letter to the final agent always hurts** — strip the letter, keep the reasoning (regression-tested in `tests/test_sanitizer.py`)
- **Cross-model**: Gemma 4 26B-A4B beats Qwen3 30B MoE by +9.75pp at the same MoE class; +12pp over Gemma 3 27b dense

Use `docs/README.md` and `docs/meeting_state_2026-05-01.md` for the current
queue/handoff. Use `RESEARCH.md` and `EXPERIMENTS.md` as historical process logs
only.

## Eval Scripts

| Script | Notes |
|---|---|
| `eval/eval_harness.py` | Unified multi-model harness (65 modes, 7 datasets) |
| `eval/eval_config.py` | Config, question loading, answer extraction, EVAL_MODES dict |
| `eval/eval_analyze.py` | Post-hoc analysis of JSONL logs |
| `eval/run_experiment_queue.py` | Queue runner for batched eval submissions |
| `eval/run_embedding_comparison.py` | Embedding A/B harness for retrieval comparisons |

## Running Evals

### Environment requirements

- **HuggingFace offline mode**: HF Hub may be unreachable from this network. Always set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when running evals. The embedding model (`gte-large-en-v1.5`) is cached locally in `~/.cache/huggingface/hub/`.
- **uv**: Prefer `uv`; if it is missing from PATH in the current shell, use `~/.local/bin/uv`.
- **API keys**: All in `.env`. Groq, DeepSeek, Google, OpenRouter, OpenAI, Cerebras.

### Known Cluster Issues

- Node `r28-1801`: RTX 2080, exclude for >4B models
- Node `a100-2207`: vLLM engine init fails. Exclude.
- Node `a100s-2307`: bad vLLM node. Exclude.
- Always use `--exclude=r28-1801,a100-2207,a100s-2307`

### Launch pattern (IMPORTANT for agents)

```bash
# Single run (recommended — monitor before scaling up)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python eval/eval_harness.py \
  --mode decompose_rag --provider groq-scout --questions 200 --dataset barexam

# Background run — do NOT pipe through grep/tail (eats errors and buffers output)
# Instead, run directly and redirect to a file:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup uv run python eval/eval_harness.py \
  --mode rag_snap_hyde --provider groq-llama70b --questions 200 --dataset barexam \
  > /tmp/eval_run.log 2>&1 &
```

**Common pitfalls:**
- Piping through `grep` or `tail` buffers stdout and hides errors — run failed silently
- Launching 6+ concurrent Groq calls hits rate limits (Llama: 1K RPD / 100K TPD, Scout: 1K RPD / 500K TPD)
- Parallel runs that share the same Groq model will contend on rate limits — run one per provider at a time
- Detail log and experiments.jsonl are written at END of run (not incrementally) — killing mid-run loses all results

### Groq rate limits

| Provider | Model | RPD | TPD |
|---|---|---|---|
| groq-llama70b | llama-3.3-70b-versatile | 1,000 | 100,000 |
| groq-scout | llama-4-scout-17b-16e-instruct | 1,000 | 500,000 |

Decompose_rag uses ~8 LLM calls per question (split + 3×snap + 3×hyde + synthesize). N=200 = ~1,600 calls. Fits within 1K RPD only because retry logic spaces them out — but running 2 evals on the same model will exceed limits.

### Monitoring a running eval

```bash
# Count completed questions in a background run's output
grep -c "PASS\|FAIL" /tmp/eval_run.log

# Current accuracy
echo "$(grep -c PASS /tmp/eval_run.log) / $(grep -c 'PASS\|FAIL' /tmp/eval_run.log)"

# Check for errors
grep -i "error\|traceback\|rate.limit" /tmp/eval_run.log | tail -5

# Watch for running eval processes
pgrep -a python | grep eval
```

### Output files

- **Detail log**: `logs/eval_{mode}_{provider}_{YYYYMMDD_HHMM}_detail.jsonl` — one JSON record per question
- **Summary**: appended to `logs/experiments.jsonl` — one JSON record per run
- Both written ONLY when run completes successfully

### Analyzing results

Use `eval/eval_analyze.py` for post-hoc analysis of JSONL detail logs.

```bash
# List all results for a mode
python3 -c "import json; [print(f\"{d['timestamp']} {d['mode']:25s} {d['provider']:20s} acc={d['accuracy']}  N={d['n_questions']}\") for d in (json.loads(l) for l in open('logs/experiments.jsonl')) if 'decompose' in d.get('mode','')]"
```

## Datasets

| Dataset | Collection | Docs | QA format | Source |
|---|---|---|---|---|
| BarExam QA | `legal_passages` | 686,324 | MC (A-D) | `reglab/barexam_qa` |
| HousingQA | `housing_statutes` | 1,837,403 | Yes/No | `reglab/housing_qa` |
| MuSiQue | in-row BM25 / `musique_passages` on cluster | per-row passages | short-answer multi-hop | local CSV build |
| LegalBench-SCALR | `legalbench_scalr_holdings` | 571 | MC (A-E) | LegalBench SCALR |
| Legal-RAG-QA | `legal_rag_passages` | 190 | Open-ended | `isaacus/legal-rag-qa` |
| Australian Legal QA | `australian_legal` | 2,124 | Open-ended | `isaacus/open-australian-legal-qa` |
| CaseHOLD | `casehold_holdings` | 50,291 | MC (A-E) | `coastalcph/lex_glue` (case_hold) |

## Data (gitignored)

- `datasets/barexam_qa/` — Passage CSVs and QA splits
- `datasets/housing_qa/` — Statute CSVs and QA pairs
- `datasets/casehold/`, `datasets/musique/`, `datasets/legalbench_scalr/` —
  added evaluation datasets
- `chroma_db/` — Persisted ChromaDB vector store
- `logs/` — Eval output logs

## Editing Guidance

- `main.py` is the source of truth for the pipeline (all runtime logic is currently here). Verify architecture claims against it before updating docs.
- If you change step schema or routing, audit both `main.py` and the skill prompt contracts in `skills/`.
- `web_scraper.py` is a standalone module (testable via CLI) imported by main.py for web_search steps.
- `utils/fast_embed.py` bypasses LangChain for bulk embedding — sentence-transformers with fp16 + chunked processing. Supports `--resume`.
- `vectorless` modes (`vectorless_direct`, `vectorless_role`, etc.) are multi-turn LLM reasoning, not real corpus search. The name is historical.
- Real structured search (entity graph, case summaries) is being built in `utils/build_entity_graph.py` and `utils/build_case_summaries.py`.
- Validity checklist: check answer change rate > 0%, evidence retrieval > 50%,
  snap accuracy consistency, and empty-retrieval guards. See
  `docs/rigour_signoff.md` and `docs/signoff_log.md`.
- Verify the current working branch with `git branch --show-current` before
  relying on branch-specific notes.
- Sequential pipeline code archived in branch `archive/sequential-pipeline`.
- See `docs/README.md` for the current documentation path.
