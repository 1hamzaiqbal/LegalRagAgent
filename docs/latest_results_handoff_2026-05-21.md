# Latest Results Handoff - 2026-05-21

Use this as the first paste/readme for an agent trying to understand the current
results state. It is a navigation note, not itself a citation gate.

## Latest Live Checkpoint

As of 2026-05-22 13:32 CDT, HousingQA state-filtered core is 8/9 signed. The
newly promoted row is `or-gemma4-26b` `rag_hyde`:
`logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl`,
4456/6853 = 65.0%, Hit@5 0.3063, Recall@5 0.2042, MRR@5 0.1964. It is signed
in `docs/signoff_log.md` with state-filter, strict cache-replay,
same-model-route, final-answer-line, fallback, think-tag, and near-cap audits.

The only remaining HousingQA core blocker is `or-gemma4-26b` `snap_hyre`.
Its full-N generation, retrieval, and document caches are complete; the audited
state-filter retrieval cache is Hit@5 0.3807, Recall@5 0.2505, MRR@5 0.2452,
which is above raw state-filter retrieval (Hit@5 0.3695, Recall@5 0.2413,
MRR@5 0.2330). The answer row is active at
`logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl`.
`current_status.md` was refreshed at 13:32 CDT and showed 474/6853 rows with
zero structural failures; live stdout has continued beyond that snapshot
without visible provider, fallback, or truncation errors. Do not launch a
duplicate Gemma core queue while `/tmp/housing_gemma_core_queue.lock` is live.

As of 2026-05-21 13:58 CDT, HousingQA `or-gemma4-26b` `rag_simple` is merged,
audited, signed, and reflected in `current_status.md`. The signed detail log is
`logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`;
the row is 4531/6853 = 66.1%, Hit@5 0.3695, MRR@5 0.2330, with zero strict
health failures and 13 logged same-model answer-format retries. The focused
Housing core gate is now 7/9 complete. The remaining core rows are
`or-gemma4-26b` `rag_hyde` and `or-gemma4-26b` `snap_hyre`.

The generated-method phase appears active host-side: `/tmp/housing_gemma_core_queue.lock`
exists with pid `3819545` from 18:55 UTC, and
`caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` grew from
55 rows at 13:57 CDT, to 95 rows at 13:58 CDT, to 333 rows at 14:03 CDT, to
1150 rows at 14:21 CDT. Treat this as active
work even if this sandbox cannot see the pid. Do not remove the lock or launch
another Gemma core queue unless the cache stops changing and the lock is
confirmed stale from a full-permission shell.
`current_status.md` was patched to treat fresh partial generation caches as an
active signal even when the host-side process is not visible in the sandbox.

As of 2026-05-21 13:15 CDT, the middle Gemma `rag_simple` recovery shards
finished cleanly and the tail is now active. Three new sample-suffixed shard
logs launched at 18:13 UTC: `s5796-e6148`, `s6148-e6500`, and `s6500-eend`.
`current_status.md` was refreshed at 13:14 CDT and shows HousingQA
`or-gemma4-26b` `rag_simple` at 85.5% active, with 5860/6853 rows visible and
zero errors, missing predictions, empty retrieval rows, state-filter misses,
retrieval/doc-cache misses, fallback markers, think tags, or near-cap rows.
Do not launch another tail run while those three logs are growing.

As of 2026-05-21 12:43 CDT, the Housing Gemma `rag_simple` recovery is still
partly live through three sample-suffixed shard logs launched at 17:34 UTC:
`s3479-e3829`, `s3829-e4180`, and `s4180-e4530`. These logs are growing and
cover the previously collided first span. A new helper,
`scripts/report_housing_gemma_rag_simple_gaps.py`, reports the current covered
and missing canonical Housing rows from the state-filter retrieval cache. At
12:43 CDT it showed 5072/6853 covered, with the only missing ranges in the
active middle shards plus the untouched tail `5849:6853`
(`hqa_Utah_49` through `hqa_Wyoming_8829`). The current sandbox cannot launch
new OpenRouter work because the preflight fails closed on DNS, so do not start
new Gemma rows here; let the growing shards finish, then rerun the gap helper.

Operational hardening made after the collision: sample-suffixed logs are now
included by `scripts/audit_housing_statefilter_goal.py`,
`scripts/local/merge_audit_housing_gemma_rag_simple.sh`,
`scripts/local/audit_housing_gemma_core_rows.sh`, and
`scripts/local/finalize_housing_gemma_signoff.sh`. This matters because the
collision-safe reruns no longer end in the old unsuffixed
`nfull-k5_detail.jsonl` pattern.
`scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh` also accepts
an opt-in `RAG_SIMPLE_RANGES="start:end ..."` override; use it after checking
the gap helper so a full-permission continuation can target only the remaining
holes instead of rerunning broad historical spans.

As of 2026-05-21 12:33 CDT, `current_status.md` had been refreshed to
2026-05-21 12:32:55 CDT and shows 67/78 signed, 67/78 full-row complete,
1/78 active, 1/78 partial stale, and 9/78 not started. The dashboard still
marks HousingQA `or-gemma4-26b` `rag_simple` as active because partial detail
logs exist, but no complete full-N Gemma `rag_simple` row is promotable yet.

Important recovery caveat: the first three-way parallel recovery launched at
10:14 CDT with the same detail-log tag for all chunks. The harness streams rows
but rewrites the detail log at run end, so same-second parallel chunks collided.
The final unsuffixed detail log currently preserves the middle span
(`hqa_Ohio_6341` through `hqa_Tennessee_7216`, 1056 rows), but the first and
third spans must come from the repaired sample-suffixed reruns below. Do not
promote or merge the collided recovery as if it were three clean chunks.

The answer wrapper has been patched so chunked runs include sample spans in the
tag, and the direct Housing cached runner now defaults `UV_CACHE_DIR=/tmp/uv-cache`.
Two same-model Cloudflare reruns started with clean sample-suffixed logs:
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s3478-e4530_detail.jsonl`
stalled at 48/1052, and
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s5796-eend_detail.jsonl`
stalled at 53/1057. Both partial logs had zero row errors, cache misses,
doc-cache misses, missing state filters, fallback markers, or think tags in the
latest spot checks, but they are incomplete and must not be merged as complete
spans. A foreground relaunch at 17:31 UTC with `UV_CACHE_DIR=/tmp/uv-cache`
failed closed during the preflight smoke with OpenRouter `Connection error`
before writing new rows.

## Read Order

1. `CLAUDE.md` - active branch mission, models, benchmark set, method contract,
   and launch rules.
2. `docs/README.md` - canonical documentation map and archive guidance.
3. `current_status.md` - current operational matrix. It is useful for completion
   and metric snapshots, but paper claims still need signoff.
4. `docs/signoff_log.md` - source of truth for citeable rows and caveats.
5. `docs/compiled_results.md` and `logs/experiments.jsonl` - historical and
   machine-readable ledgers. Do not cite raw JSONL tail rows unless signoff
   confirms the merged/detail log is clean.
6. `docs/barexam_housing_core_focus_2026-05-20.md`,
   `docs/housingqa_statefilter_goal_checklist_2026-05-21.md`, and
   `docs/housingqa_handoff_to_next_agent_2026-05-21.md` - current HousingQA
   blocker, state-filter contract, continuation commands, and audit gates.
7. `docs/snap_hyre_paper_agent_handoff_2026-05-20.md`,
   `docs/paper_meeting_handoff_2026-05-20.md`, and
   `docs/snap_hyre_good_example_handoff_2026-05-20.md` - paper-facing narrative
   notes and concrete positive Snap-HyRE example. Check them against the newer
   status/signoff docs before repeating numbers.
8. `docs/benchmark_paradigm_audit_2026-05-20.md` - why CaseHOLD and
   LegalBench-SCALR are not in the active exact-scored main matrix.

## Current Snapshot

`current_status.md` was last refreshed at 2026-05-21 10:11:24 CDT. It reports:

- Overall: 67/78 signed, 67/78 full-row complete, 0/78 active,
  2/78 partial stale, 9/78 not started.
- BarExamQA: all listed cells are 100% signed.
- Legal-Link-EU: all listed cells are 100% signed.
- MASLegalBench: all applicable cells are 100% signed; golden rows are not
  applicable because there are no official per-question gold evidence ids.
- HousingQA: the main state-filtered Groq rows are signed, but Gemma 26B
  Housing rows remain the active blocker.

The active HousingQA core target is only the state-filtered 3x3 core:
`rag_simple`, `rag_hyde`, and `snap_hyre` across `groq-llama8b`,
`or-gemma4-26b`, and `groq-llama70b`. That core is 6/9 complete. The missing
three are all `or-gemma4-26b`: `rag_simple` is blocked partial at 3691/6853
with 3 failed-closed OpenRouter key-limit rows, and full-N `rag_hyde` /
`snap_hyre` have not yet been run.

## Results Orientation

Use these as orientation only; verify exact claims in `docs/signoff_log.md`.

- BarExamQA is currently the cleanest positive Snap-HyRE story, especially on
  Gemma 26B. In `current_status.md`, Gemma 26B `snap_hyre` is 82.0% final
  accuracy versus 78.0% `rag_simple`, 80.3% `rag_hyde`, and 80.7%
  `rag_rewrite`; its retrieval exposure is also slightly higher than HyDE.
- BarExamQA 8B also shows Snap-HyRE above raw RAG in final accuracy
  (56.9% vs 54.5%), while the 70B row is strong but not uniformly best against
  HyDE/rewrite.
- Legal-Link-EU has a retrieval-positive Snap-HyRE signal for larger models:
  Gemma 26B `snap_hyre` has much higher generated-query retrieval exposure than
  Gemma `rag_hyde`, and Llama 70B `snap_hyre` has the strongest generated-query
  retrieval exposure. Downstream answer accuracy is more mixed and should be
  framed carefully.
- MASLegalBench retrieval metrics are same-source-document proxies, not official
  gold-passage Hit@k. Do not describe them as gold evidence retrieval.
- HousingQA now requires jurisdiction state filtering for all retrieval methods
  in the active matrix. Without the state filter, the comparison is not the
  intended main result. Current signed Housing state-filter rows show raw RAG is
  strong; generated methods are not yet universally better there.
- The Gemma q500 Housing `snap_hyre_exemplar` diagnostic is clean and
  retrieval-positive but answer-parity, not answer-lift: canonical q500
  `snap_hyre` is 315/500 = 63.0%, Hit@5 0.3820, MRR@5 0.2429; exemplar q500 is
  314/500 = 62.8%, Hit@5 0.4120, MRR@5 0.2647.

## Current Blocker And Continuation

The immediate Gemma blocker partially cleared after the 10:07 CDT safe-fail:
a same-model `or-gemma4-26b` HousingQA `rag_simple` recovery launched at
2026-05-21 10:14 CDT. It is replacing the three failed row spans through
`OPENROUTER_PROVIDER_ONLY=Cloudflare` and is still running as of the latest
local check.

Live recovery artifacts:

- `logs/eval_rag_simple_or-gemma4-26b_20260521_101450_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_3478_4530_20260521_151441.out`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_4634_5690_20260521_151441.out`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_5796_end_20260521_151441.out`

At 2026-05-21 10:28 CDT, the live detail log had 388 parsed rows with zero
JSON errors, row errors, missing final answer lines, retrieval-cache misses,
doc-cache misses, missing state filters, fallback markers, or think tags.

Recent safe-fail modes:

- OpenRouter reachable but budget guard fails with `limit_remaining=0 <
  required 0.01`.
- Sandbox DNS fails before budget lookup.

Latest check: `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`
was rerun at 2026-05-21 10:07 CDT. It found no Housing launch locks, verified
the focused gate remains `complete=6/9`, checked the three Gemma `rag_simple`
resume offsets and exact model label, confirmed the current `rag_simple` merge
gate still fails closed at 3691/6853, then failed closed on OpenRouter DNS after
bounded retries before budget lookup or row launch. No signoff-candidate file
was created.

Do not silently substitute Gemma 3, Gemini, direct Google, or
`google/gemma-4-26b-a4b-it:free`. The `:free` suffix exists only as an explicit
noncanonical escape hatch if the experiment label and signoff caveat are
changed deliberately.

Safe continuation sequence after budget/API reachability is fixed:

```bash
scripts/local/check_housing_gemma_readiness.sh
CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh
scripts/local/run_housing_gemma_after_key_reset.sh
scripts/local/verify_housing_statefilter_goal_complete.sh
```

The final verification command is expected to fail until the three missing
Gemma Housing core rows and the full-N exemplar diagnostic are complete and
signed.

## Guardrails For The Next Agent

- Cite `docs/signoff_log.md`, not older narrative docs or raw tail logs.
- Treat `current_status.md` as a dashboard, not a paper source by itself.
- Keep CaseHOLD and LegalBench-SCALR out of the current main matrix unless the
  user explicitly re-adds them under the current fixed-method contract.
- Preserve `NO_SILENT_FALLBACK=1`; same-model route caveats must be explicit.
- HousingQA retrieval rows must have state filtering. Unfiltered Housing rows
  are provenance/ablation only.
- For generated methods, strict cache replay matters: HyDE/Snap-HyRE generation
  cache, retrieval-id cache, and retrieval document cache should all be checked
  before signoff.
- Watch for malformed final answer lines, think tags, near-cap/truncation
  artifacts, and fallback markers. Same-model format retries are acceptable only
  when logged and caveated.

## Useful Scripts

- `scripts/update_current_status.py` - refresh operational dashboard.
- `scripts/audit_housing_statefilter_goal.py --allow-incomplete` - current
  focused Housing core count; latest expected result is `complete=6/9`.
- `scripts/audit_housing_statefilter_detail.py` - per-detail Housing state
  filter/cache/final-answer audit.
- `scripts/summarize_housing_statefilter_signoff.py` - produce signoff-ready
  Housing rows only after strict checks pass.
- `scripts/local/check_housing_gemma_readiness.sh` - read-only local or network
  preflight for the remaining Gemma Housing work.
- `scripts/local/run_housing_gemma_after_key_reset.sh` - canonical continuation
  wrapper after OpenRouter route/budget is usable.
- `scripts/local/finalize_housing_gemma_signoff.sh` - append signoff rows only
  for clean full-N Gemma Housing logs.
- `scripts/local/verify_housing_statefilter_goal_complete.sh` - strict final
  completion gate.
- `scripts/analyze_detail_flags.py`, `scripts/score_retrieval_qrels.py`,
  `scripts/audit_retrieval_cache.py`, and `scripts/compute_mcnemar.py` - common
  row health, retrieval, and paired-significance helpers.
