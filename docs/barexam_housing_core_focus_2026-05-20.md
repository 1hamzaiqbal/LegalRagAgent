# BarExamQA and HousingQA Core Focus - 2026-05-20

This note supersedes the broader four-benchmark operating queue for the current
run window. The immediate goal is to finish BarExamQA and HousingQA for the
three core retrieval methods:

- `rag_simple`
- `rag_hyde`
- `snap_hyre`

`golden_passage`, `golden_plus_neighbors`, `rag_rewrite`, and exemplar variants
are secondary diagnostics. Use them only after the core rows are healthy or when
they directly explain a core-row failure.

## Current Scope

- Models: `groq-llama8b`, `or-gemma4-26b`, `groq-llama70b`.
- Benchmarks: BarExamQA and HousingQA only.
- Main answer depth: `RETRIEVAL_K=5`.
- BarExamQA uses the canonical full-corpus setup already represented in
  `current_status.md` and `docs/signoff_log.md`.
- HousingQA core retrieval rows must use the jurisdiction metadata filter:
  `EVAL_HOUSING_STATE_FILTER=1` / `--housing-state-filter`.

## BarExamQA

The core BarExamQA rows are already complete and signed for all three models:

| Method | `groq-llama8b` | `or-gemma4-26b` | `groq-llama70b` |
|---|---:|---:|---:|
| `rag_simple` | signed | signed | signed |
| `rag_hyde` | signed | signed | signed |
| `snap_hyre` | signed | signed | signed |

Paper-facing claims should still cite `docs/signoff_log.md`, not this handoff.

## HousingQA State-Filtered Core Queue

HousingQA is now treated as a jurisdiction-filtered retrieval benchmark. Older
unfiltered national-corpus Housing rows are provenance or ablation rows only.

Known state-filtered state after the 2026-05-21 10:01 CDT manual refresh:

| Method | `groq-llama8b` | `or-gemma4-26b` | `groq-llama70b` |
|---|---:|---:|---:|
| `rag_simple` | signed, 4269/6853 = 62.3% | blocked partial, 3691/6853 deduped rows with 3 OpenRouter key-limit failures to supersede | signed, 4258/6853 = 62.1% |
| `rag_hyde` | signed, 4047/6853 = 59.1% | generation/cache build queued after Gemma `rag_simple` | signed, 4263/6853 = 62.2% |
| `snap_hyre` | signed, 4043/6853 = 59.0% | generation/cache build queued after Gemma `rag_hyde` | signed merged row, 4087/6853 = 59.6% |

The 70B `snap_hyre` tail was relaunched after fixing a cache-scope bug in
`scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh`. The wrapper
now exports `CACHE_SCOPE=qfull_seed42_statefilter` into `run_answer_cell.sh` so
tail repairs reuse the full deterministic generation and retrieval caches instead
of looking for nonexistent sample-scoped caches.

Signed state-filter rows added to `docs/signoff_log.md`:

- HousingQA `groq-llama8b` `rag_simple`:
  `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl`.
- HousingQA `groq-llama8b` `rag_hyde`:
  `logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl`.
- HousingQA `groq-llama8b` `snap_hyre`:
  `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl`.
- HousingQA `groq-llama70b` `rag_simple`:
  `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl`.
- HousingQA `groq-llama70b` `rag_hyde`:
  `logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl`.
- HousingQA `groq-llama70b` `snap_hyre`:
  `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`.

Blocked answer jobs:

- Gemma `rag_simple` was sharded across three explicit same-model Cloudflare
  chunk sessions:
  `housing_gemma_rag_simple_chunk_3368_4530_20260521_112618`,
  `housing_gemma_rag_simple_chunk_4530_5690_20260521_112618`, and
  `housing_gemma_rag_simple_chunk_5690_end_20260521_112618`.
  The original Cloudflare tail `housing_gemma_rag_simple_cloudflare_tail_20260521_043003`
  was intentionally stopped after 3282 clean tail rows; the three chunks use
  `[3368:4530]`, `[4530:5690]`, and `[5690:]` with a small overlap for
  dedupe safety. All three chunks failed closed at 2026-05-21 06:38 CDT on
  OpenRouter `403 Key limit exceeded (monthly limit)`, leaving three bad rows
  that must be superseded before merge/signoff:
  `hqa_Nebraska_2941`, `hqa_Ohio_6341`, and `hqa_Texas_4530`.
- `housing_gemma_core_queue_20260521_073328` is no longer live; it exited before
  doing API work. `bash -n` now passes on
  `scripts/local/run_housing_gemma_core_queue.sh`, but the queue should not be
  relaunched until the exact same-model OpenRouter route is available again.
  Once relaunched, it builds full Housing Gemma `rag_hyde`/`snap_hyre`
  generation caches, state-filter retrieval/doc caches, and answer rows on
  explicit same-model Cloudflare routing. The q500 exemplar retry has completed.

The q500 answer gate runner is idempotent as of 2026-05-21 00:09 CDT: it skips a
q500 mode if a 500-row state-filtered/cache-replay detail log already passes the
same no-silent/final-line/fallback/think checks. The Gemma core queue also waits
for the slot queue, so early q500 execution does not race the full Gemma
HyDE/Snap-HyRE cache builds. The older waiting-only Gemma follow-up/core tmux
sessions from `043157` and `043502` were killed and restarted at `0017` so the
live sleepers used these hardened queue scripts. After canonical q500 finished,
the first exemplar attempt exposed a cache resolver bug in `run_answer_cell.sh`
for state-filtered `snap_hyre_exemplar`: it checked the state-filtered
real-passage generation filename, then fell back to a non-realpassage filename
before trying the existing unfiltered real-passage generation cache. That
resolver is patched and `bash -n` passes. The duplicate q500 follow-up queue and
old Gemma core queue were killed; the later Gemma core queue
`housing_gemma_core_queue_20260521_073328` exited before API work and is not
currently live. The older
Groq core queues from `044000` and `0019` were killed and restarted; the latest
queue `housing_groq_core_queue_20260521_060520` stopped after the post-OK 70B
HyDE wrapper caveat. The remaining 8B `snap_hyre` row was completed, audited,
and signed from the explicit state-filter cache replay session.

Latest live status refresh at 2026-05-21 10:11 CDT:

- `or-gemma4-26b` `rag_simple` recovery launched at 2026-05-21 10:14 CDT on
  the exact same OpenRouter Gemma route (`google/gemma-4-26b-a4b-it`,
  `OPENROUTER_PROVIDER_ONLY=Cloudflare`). The live detail log is
  `logs/eval_rag_simple_or-gemma4-26b_20260521_101450_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`.
  At 2026-05-21 10:28 CDT it had 388 parsed rows and the midstream structural
  audit found zero JSON errors, row errors, missing final answer lines,
  retrieval/doc-cache misses, missing state filters, fallback markers, or think
  tags. Do not sign this row until the wrapper finishes, merges, and the strict
  full-N audit passes.

- `groq-llama70b` `rag_hyde`: complete, audited, and signed at 4263/6853 =
  62.2%; retrieval Hit@5/MRR@5 0.3495/0.2260; 0 state-filter/cache/doc/HyRE/
  final-line/fallback/think issues; 2 valid same-model answer-format retries;
  post-OK wrapper EOF caveat after the detail log and `logs/experiments.jsonl`
  summary were written.
- `groq-llama8b` `snap_hyre`: complete, audited, and signed at 4043/6853 =
  59.0%; retrieval Hit@5/MRR@5 0.2956/0.1888; 0
  state-filter/cache/doc/HyRE/final-line/fallback/think issues; 140 valid
  same-model answer-format retries with 3-token repairs; CE doc truncation
  appears on 3382 rows as a reranker-input caveat, not final-answer truncation.
  Paired comparison is +3.62pp vs `llm_only`, -3.30pp vs state-filtered
  `rag_simple`, and -0.06pp vs state-filtered `rag_hyde`.
- `or-gemma4-26b` `rag_simple`: 3691/6853 deduped combined
  prefix+Cloudflare-tail/chunk rows. Of these, 3688 rows are clean and 3 rows
  are failed-closed OpenRouter key-limit rows. The partial row has f_acc
  2445/3691 = 66.2% and gold hit@5 1340/3691 = 36.3%, but it is not merge- or
  signoff-ready. All chunk launches logged `NO_SILENT_FALLBACK=1`,
  `housing_state_filter=on`, strict full state-filter retrieval/doc caches, and
  explicit `OPENROUTER_PROVIDER_ONLY=Cloudflare` for
  `google/gemma-4-26b-a4b-it`. The failed rows are useful evidence that the
  no-silent-fallback guard worked; they must be superseded with same-model
  reruns before audit/signoff.
  A direct sandbox key check at 2026-05-21 08:27 CDT reached OpenRouter and
  failed closed with `limit_remaining=0 < required 0.01`
  (`usage=23.912568061`, `limit=20`). Earlier sandbox network preflights either
  reached OpenRouter and reported `limit_remaining=0` or failed closed on
  transient DNS before budget lookup. The Gemma resume/core launchers are guarded by
  `scripts/check_openrouter_key_status.py` and should fail before writing new
  failed rows until API reachability and budget are restored.
  The full readiness-wrapper path was also retried at 2026-05-21 10:07 CDT:
  it passed offline readiness, exact-model checks, and the expected partial
  merge failure check, then failed closed on DNS after bounded retries before
  budget lookup or row launch. No
  `docs/generated/housing_gemma_signoff_candidates_20260521_150713.md` file
  was created.
  `current_status.md` was refreshed again at 2026-05-21 10:11 CDT and remains
  at 67/78 signed, 0/78 active, with the same three missing `or-gemma4-26b`
  HousingQA core rows.
  The strict final gate at 2026-05-21 09:22 CDT passed the no-lock and
  syntax/Python helper checks, then failed at the expected missing full clean
  Gemma `rag_simple` row.
  Launch-on-ready watcher attempts reached OpenRouter at 2026-05-21 09:10 CDT,
  09:19 CDT, and 09:22 CDT and failed closed at `limit_remaining=0`. The
  recorded pids `3636546`, `3643843`, and `3646233` were later not live, so
  their stale watcher locks were removed. No watcher is currently active, and
  no Housing launch locks, answer rows, or signoff-candidate files were created.
  The old 2026-05-20 8B `snap_hyre` prefix+repair-tail covers 6853 unique
  rows but fails the state-filter audit with `missing_state_filter=6853`, so it
  is provenance only and must not be merged/promoted for the main matrix.
- Gemma q500 canonical `snap_hyre` completed in
  `logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl`
  with `NO_SILENT_FALLBACK=1`, state filtering, strict cache replay, and
  explicit Cloudflare routing; full audit passes at 315/500 = 63.0%, Hit@5
  0.3820, with 0 state-filter/cache/doc/HyRE/fallback/think/final-line issues.
  Exemplar q500 completed in
  `logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260521_023301_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre_exemplar-n500-k5_detail.jsonl`.
  Full strict audit passes at 314/500 = 62.8%, Hit@5 206/500 = 41.2%, with 0
  state-filter/cache/doc/HyRE/final-line/fallback/think issues and max output
  1425 tokens. Paired answer comparison vs canonical q500 is answer-parity but
  not lift: -0.2pp, b/c=36/37, p=1.0, CI [-3.6, 3.005]. Retrieval exposure
  improves directionally: canonical Hit@1/5/10 0.1640/0.3820/0.4560 and
  MRR@5/10 0.2429/0.2528; exemplar Hit@1/5/10 0.1840/0.4120/0.4960 and
  MRR@5/10 0.2647/0.2760. Treat exemplar as scale-eligible after required core
  rows, not as a higher-priority replacement for core methods.

Efficiency/token diagnostics are now generated from the same detail logs by
`scripts/compile_efficiency_metrics.py`. The current operational snapshot is in
`docs/generated/housingqa_statefilter_efficiency_2026-05-21.md` with a matching
CSV at `docs/generated/housingqa_statefilter_efficiency_2026-05-21.csv`. Treat
active/partial rows there as diagnostics only until the full-N row is signed.

Gemma `rag_simple` route note:

- The initial default OpenRouter route was stopped after 88 clean rows because it
  was too slow for full-N completion. The later Cloudflare tail used
  `OPENROUTER_PROVIDER_ONLY=Cloudflare` for the same
  `google/gemma-4-26b-a4b-it` model, then the sharded continuation failed closed
  on the OpenRouter monthly key limit.
- The route mix should be cited as an explicit same-model caveat for
  `rag_simple`. Do not require all `rag_simple` rows to show
  `openrouter_provider_only=Cloudflare`, because that would incorrectly reject
  the known 88-row default-route prefix. Generated Gemma rows are still expected
  to run on the explicit Cloudflare route after key reset.
- `scripts/check_openrouter_key_status.py --min-limit-remaining 0.01` must pass
  before any Gemma OpenRouter queue is relaunched. The latest row-level blocker
  remains the OpenRouter monthly key limit; recent sandbox network preflights
  either reach OpenRouter and report `limit_remaining=0` or fail closed on
  transient DNS before budget lookup. The key-status preflight now retries
  transient DNS/5xx/429 failures by default (`--retries`, `--retry-delay`) but
  still fails closed on missing keys, invalid responses, or insufficient
  `limit_remaining`.
- `scripts/check_openrouter_chat_route.py` is now wired into the after-reset
  wrapper and direct Gemma launch helpers after their budget guards. It performs
  a tiny exact-route OpenRouter completion with provider fallbacks disabled
  before any resume/cache/answer rows can be written. It also honors
  `OPENROUTER_PROVIDER_ONLY`, `OPENROUTER_PROVIDER_ORDER`, and
  `OPENROUTER_PROVIDER_IGNORE` for explicit same-model route control.
- `scripts/check_expected_provider_model.py` is now wired into the Gemma
  after-reset launchers. It accepts `or-gemma4-26b` resolving to
  `google/gemma-4-26b-a4b-it` and rejects near-miss aliases such as direct Google
  `gemma` (`gemma-3-27b-it`) before any generation or answer calls run.
- The configured OpenRouter free alias is not being treated as a silent
  substitute. `or-gemma4-26b-free` resolves to
  `google/gemma-4-26b-a4b-it:free`, which fails the current exact-model guard
  for the canonical `or-gemma4-26b` row.
- `scripts/check_expected_provider_model.py --allow-openrouter-free-suffix`
  and `OPENROUTER_ALLOW_FREE_SUFFIX=1` now exist as explicit noncanonical route
  validators for that suffix. They are disabled by default and do not change
  the strict canonical launcher path. A 2026-05-21 08:37 CDT non-launching
  after-reset preflight with `PROVIDER=or-gemma4-26b-free`,
  `MODEL_LABEL=or-gemma4-26b`, and `OPENROUTER_ALLOW_FREE_SUFFIX=1` passed the
  opt-in guard, then failed closed at the same OpenRouter budget check before
  any row work. Direct tiny completion smokes at 08:42 CDT showed the `:free`
  route is not currently useful for this run: unconstrained free routing returned
  upstream Google AI Studio 429, and Cloudflare-pinned free routing returned 404
  because only `google-ai-studio` was available.
- Do not switch the remaining `or-gemma4-26b` rows to direct Google in the
  current environment. `llm_config.py` maps direct Google providers to Gemma 3
  IDs, and a non-generating Google model-list check returned `API_KEY_INVALID`.
  Gemma 3 or Gemini would be a model-identity change, not a fallback.
- Early tail audit confirmed row-level provider route metadata records
  `{"openrouter_provider_only": "Cloudflare"}` on every checked tail row.
- When complete, merge the prefix
  `logs/eval_rag_simple_or-gemma4-26b_20260520_230419_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`
  and Cloudflare tail
  `logs/eval_rag_simple_or-gemma4-26b_20260520_233010_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`
  with `scripts/merge_detail_logs.py --on-duplicate last`, then audit/sign the
  merged log with a same-model route-recovery caveat.

Prepared exemplar gate:

- Canonical Gemma 26B q500 state-filter cache:
  `caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
  with doc cache
  `caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10_doc_cache.jsonl`.
- Exemplar Gemma 26B q500 state-filter cache:
  `caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl`
  with doc cache
  `caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10_doc_cache.jsonl`.
- Retrieval gate result on the matched q500 slice: canonical `snap_hyre`
  Hit@5 0.3820 / MRR@5 0.2429; exemplar `snap_hyre_exemplar` Hit@5 0.4120 /
  MRR@5 0.2647. This was enough lift to justify the completed q500 answer gate.
- The q500 answer gate runner is
  `scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh`. It runs
  canonical `snap_hyre` and `snap_hyre_exemplar` separately so each mode uses
  its matching strict document cache.

## Operational Order

Readiness check before restart:
`scripts/local/check_housing_gemma_readiness.sh` runs the offline 9-cell gate,
Gemma failed-row offset check, exact-model check, cache snapshot, and blocked
`rag_simple` merge-gate probe without touching OpenRouter. Use
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` after key/API
reset to include the OpenRouter preflight without launching rows.

One-command restart after OpenRouter key/account and API reachability reset:
`scripts/local/run_housing_gemma_after_key_reset.sh`. It now runs local shell
syntax and Python compile checks before the exact-model and OpenRouter
preflights, then performs the key-budget preflight, resumes the three failed
`rag_simple` spans, runs the merge/audit gate, then launches Gemma
`rag_hyde`/`snap_hyre`, runs
`scripts/local/audit_housing_gemma_core_rows.sh` on the generated rows, and
calls `scripts/local/finalize_housing_gemma_signoff.sh`. The finalizer appends
missing Gemma signoff rows only after the signoff summarizer accepts full clean
logs, refreshes `current_status.md`, and runs the focused 9-cell completion
audit. The wrapper also writes candidate signoff Markdown rows to
`docs/generated/housing_gemma_signoff_candidates_<timestamp>.md` for review
traceability, but only after the model-identity and OpenRouter budget preflights
pass.
For full launches, this wrapper now acquires
`/tmp/housing_gemma_after_key_reset.lock` before the model and budget preflights,
records launch metadata in the lock directory, and exits with code 11 if another
full Gemma Housing continuation appears active. `PREFLIGHT_ONLY=1` skips the
lock because it does not launch resume, cache-build, answer, audit, or signoff
work.
The direct Gemma helpers now have their own duplicate-launch locks too:
`resume_housing_gemma_rag_simple_after_key_reset.sh` uses
`/tmp/housing_gemma_rag_simple_resume.lock`, and
`run_housing_gemma_core_queue.sh` uses `/tmp/housing_gemma_core_queue.lock`.
The full exemplar scale-up uses `/tmp/housing_gemma_exemplar_full.lock`.
`scripts/local/check_housing_gemma_readiness.sh` reports all four lock
directories and exits with code 12 when any are present, unless
`ALLOW_EXISTING_LOCKS=1` is set for read-only inspection after checking whether
the lock is stale.
Use `PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh` to
check only the exact-model route and OpenRouter budget after reset without
launching resume, cache-build, answer, audit, or signoff work.
After the restart wrapper completes, run
`scripts/local/verify_housing_statefilter_goal_complete.sh` as the final
completion gate. It checks for launch locks, validates the Housing helper
syntax/Python files, runs the Gemma signoff finalizer, refreshes
`current_status.md`, and requires `scripts/audit_housing_statefilter_goal.py` to
pass without `--allow-incomplete`. By default it also requires the full-N
Gemma `snap_hyre_exemplar` diagnostic because the q500 gate was clean and
retrieval-positive with answer parity.
The Housing queue wrappers now preserve intentionally empty `WAIT_PATTERNS` /
`WAIT_PATTERN` values, so this restart wrapper can bypass stale tmux wait names
after a live-job audit confirms there are no active Housing answer cells.
The Gemma core queue also threads `HYRE_CACHE_ROOT`, `RETRIEVAL_CACHE_ROOT`, the
exact generated/retrieval/document cache paths, and question/seed/depth values
into the answer wrapper, avoiding default cache-root fallback if the queue is
launched with explicit cache-root overrides.
As of 2026-05-21 07:58 CDT, `scripts/local/run_housing_gemma_core_queue.sh` and
`scripts/local/run_housing_gemma_exemplar_full_after_core.sh` force
`NO_SILENT_FALLBACK=1` instead of inheriting a possibly false parent
environment value. Their generation-cache postchecks also use broad fallback
detection and exact 6853-row checks before retrieval-cache construction. The
answer wrappers already forced the same guard. As of 2026-05-21 08:14 CDT, a
temp `.env` probe confirmed `scripts/local/run_answer_cell.sh` restores explicit launcher values for
`LLM_MAX_COMPLETION_TOKENS`, `EVAL_MIN_COMPLETION_TOKENS`,
`NO_SILENT_FALLBACK`, and `OPENROUTER_PROVIDER_ONLY` after sourcing `.env`, so
stale local `.env` values cannot silently weaken a strict launcher. As of
2026-05-21 08:19 CDT, a `uv run` dotenv probe confirmed generation-cache Python
preserves exported strict values (`LLM_PROVIDER=or-gemma4-26b`,
`NO_SILENT_FALLBACK=1`, `OPENROUTER_PROVIDER_ONLY=Cloudflare`) over hostile
dotenv values as well.
Its launch lock was tested with a forced-impossible OpenRouter budget threshold:
the queue acquired `/tmp/housing_gemma_core_queue.lock`, failed at preflight
before cache or answer work, and removed the lock on exit.
The Gemma audit helper merges state-filtered full-N detail logs, checks
generation-cache health for generated methods, reruns detail/retrieval audits,
and prints the detail logs that should be added to `docs/signoff_log.md`. Clean
rows are also passed through `scripts/summarize_housing_statefilter_signoff.py`,
which rejects incomplete/bad rows by default and emits a ready-to-paste table
row. Its detail-log selector matches both actual provider and stable
`MODEL_LABEL`, so explicit same-model routes remain visible without silently
changing the row identity.
The signoff finalizer now requires the focused 9-cell core audit to pass before
it appends a `snap_hyre_exemplar` row. A 2026-05-21 08:09 CDT temp-signoff probe
with `MODES=snap_hyre_exemplar` exited before appending because the current core
audit is still 6/9.
`scripts/audit_housing_statefilter_detail.py` now uses the same broad fallback
detection as the signoff summarizer and requires the last non-empty final answer
line to exactly match the parsed prediction: `Answer: Yes` or `Answer: No`.
`bash -n` passes on the Housing queue/restart/audit scripts, and
`python3 -m py_compile` passes on the status/provider guard scripts.

1. After the OpenRouter key/account is reset or replaced, resume Housing
   state-filtered `or-gemma4-26b` `rag_simple` with
   `scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`, which
   reruns the three failed chunk spans in parallel by default. Set
   `RUN_PARALLEL=0` to make it sequential. The failed chunk offsets are:
   `SAMPLE_START=3478 SAMPLE_END=4530`,
   `SAMPLE_START=4634 SAMPLE_END=5690`, and `SAMPLE_START=5796`. These starts
   intentionally include the failed rows so merge `--on-duplicate last`
   supersedes the failed-closed records. The resume script now verifies these
   offsets against `datasets/housing_qa/questions.csv` before API preflights:
   `3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and
   `5796 -> hqa_Texas_4530`.
   Then run `scripts/local/merge_audit_housing_gemma_rag_simple.sh`; it should
   fail until all 6853 rows are present and every state-filter/cache/final-line
   gate is clean.
2. Build full Housing Gemma generation caches for `rag_hyde` and `snap_hyre`;
   then build their state-filtered retrieval/doc caches and run the answer rows.
   As of 2026-05-21 08:23 CDT, no full Gemma Housing `rag_hyde` or `snap_hyre`
   generation/retrieval/doc caches exist yet. The canonical route remains
   `or-gemma4-26b` -> `google/gemma-4-26b-a4b-it` through OpenRouter with
   `OPENROUTER_PROVIDER_ONLY=Cloudflare` and OpenRouter fallbacks disabled.
3. Scale the HousingQA `snap_hyre_exemplar` diagnostic to full-N after the three
   Gemma core rows. The q500 gate is clean and retrieval-positive with answer
   parity, so `scripts/local/run_housing_gemma_exemplar_full_after_core.sh` is
   now the post-core scale-up path.

Keep answer concurrency at the deliberate 2-3 job target. Cache-building can run
alongside one answer job when it is not forcing live LLM calls.

## Validation Gates

For each completed core row:

- `NO_SILENT_FALLBACK=1` in the runner log.
- Housing state filter present in the detail log or retrieval cache metadata.
- Strict retrieval cache and document cache hits on cached retrieval rows.
- `scripts/audit_housing_statefilter_detail.py` passes for promoted full-N
  Housing state-filter detail logs. Use `--require-hyre-cache` for `rag_hyde`
  and `snap_hyre` rows.
- `scripts/analyze_detail_flags.py` passes with zero errors, missing predictions,
  parse failures, fallback markers, empty evidence, and think tags.
- If a run is a repair tail, merge with `--on-duplicate last` and audit the merged
  detail log, not just the tail.
