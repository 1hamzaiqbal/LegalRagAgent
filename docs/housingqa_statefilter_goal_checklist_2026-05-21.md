# HousingQA State-Filter Goal Checklist - 2026-05-21

This is the completion-audit checklist for the active HousingQA goal. Do not mark
the goal complete until every required row below is full-N, audited, and either
signed in `docs/signoff_log.md` or explicitly marked as a non-promoted failed
diagnostic with a reason.

Current 2026-05-22 13:32 CDT audit checkpoint: the focused HousingQA
state-filtered core matrix is **8/9 signed**. `or-gemma4-26b` `rag_simple` and
`or-gemma4-26b` `rag_hyde` are now full-N, audited, and signed in
`docs/signoff_log.md`. The remaining core row is `or-gemma4-26b` `snap_hyre`;
its generation, retrieval, and document caches are complete, and its full-N
answer row is actively running in
`logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl`.
At the latest `current_status.md` refresh, that answer row was 474/6853
complete with zero structural health failures. Its audited full-N retrieval
cache is already positive versus raw state-filter retrieval: Hit@5 0.3807 /
Recall@5 0.2505 / MRR@5 0.2452 for Snap-HyRE versus Hit@5 0.3695 /
Recall@5 0.2413 / MRR@5 0.2330 for raw `rag_simple`.

Recent operator checks:

At 2026-05-21 13:58 CDT, Gemma `rag_simple` passed the full merge/audit/signoff
path. `scripts/local/merge_audit_housing_gemma_rag_simple.sh` merged 15 detail
logs into
`logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`;
the audit reported rows 6853/6853, accuracy 4531/6853 = 66.1%, gold retrieved
2532/6853, Hit@5 0.3695, MRR@5 0.2330, and zero errors, missing predictions,
state-filter misses, retrieval/doc-cache misses, malformed final answers,
fallback rows, or think tags. `MODES=rag_simple`
`scripts/local/finalize_housing_gemma_signoff.sh` appended the signoff row and
the focused audit now reports `complete=7/9`.

At the same checkpoint, Gemma generated-method work appears active. The cache
`caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` grew from
55 rows at 13:57 CDT to 95 rows at 13:58 CDT, 333 rows at 14:03 CDT, and 1150
rows at 14:21 CDT, and `/tmp/housing_gemma_core_queue.lock` records
`scripts/local/run_housing_gemma_core_queue.sh` with pid `3819545` created at
2026-05-21T18:55:52Z. Do not clear that lock or launch another core queue from
this sandbox while the cache is growing.
`current_status.md` now treats fresh partial generation caches as active
evidence even when host-side pids are not visible to the sandbox.

At 2026-05-21 13:15 CDT, the middle recovery shards had completed cleanly and
the tail recovery was active. New live artifacts:
`logs/eval_rag_simple_or-gemma4-26b_20260521_131350_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s5796-e6148_detail.jsonl`,
`logs/eval_rag_simple_or-gemma4-26b_20260521_131350_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s6148-e6500_detail.jsonl`,
and
`logs/eval_rag_simple_or-gemma4-26b_20260521_131350_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s6500-eend_detail.jsonl`.
The 13:14 CDT `current_status.md` refresh showed 5860/6853 visible rows for
Gemma `rag_simple`, 85.5% active, with zero errors, missing predictions, empty
retrieval rows, missing state filters, retrieval/doc-cache misses, fallback
markers, think tags, or near-cap rows. Do not launch another tail recovery
while those logs are growing.

At 2026-05-21 12:43 CDT, the sample-suffixed Gemma `rag_simple` shard logs
from the 17:34 UTC recovery were still growing. The new read-only helper
`scripts/report_housing_gemma_rag_simple_gaps.py` reported
`provider=or-gemma4-26b mode=rag_simple canonical=6853 covered=5072
missing=1781 unexpected=0`. The remaining gaps were
`3572:3829`, `3920:4180`, `4270:4530`, and `5849:6853`; the first three are
inside active growing shards, while the final tail is not currently covered by
an active log. The current sandbox `CHECK_NETWORK=1`
`scripts/local/check_housing_gemma_readiness.sh` run failed closed on
OpenRouter DNS before any row launch, so the next safe action is to monitor the
active shard logs to completion, refresh the gap report, and then target only
the remaining missing ranges when API reachability is available.

The Housing Gemma audit/finalization path now accepts collision-safe
sample-suffixed detail logs. `scripts/audit_housing_statefilter_goal.py`,
`scripts/local/merge_audit_housing_gemma_rag_simple.sh`,
`scripts/local/audit_housing_gemma_core_rows.sh`, and
`scripts/local/finalize_housing_gemma_signoff.sh` all match
`nfull-k5*_detail.jsonl`, not only the old unsuffixed `nfull-k5_detail.jsonl`.
`scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh` now accepts
`RAG_SIMPLE_RANGES="start:end ..."` for targeted cleanup after active shard
logs finish. Its `VERIFY_ONLY=1` path validates and prints the chosen ranges
without API calls.

At 2026-05-21 12:33 CDT, `current_status.md` had been refreshed to
2026-05-21 12:32:55 CDT and reported 67/78 signed, 67/78 full-row complete,
1/78 active, 1/78 partial stale, and 9/78 not started. The active row is still
HousingQA `or-gemma4-26b` `rag_simple` in the dashboard, but it remains
incomplete and unpromoted.

Operational correction: the first same-model three-way recovery launched at
10:14 CDT exposed a detail-log collision. All three chunks shared the same
answer-cell tag, and the harness rewrites the detail log at the end, so the
unsuffixed 10:14 detail file cannot stand in for all three chunks. It currently
contains the complete middle span only. `scripts/local/run_answer_cell.sh` now
adds `-s${SAMPLE_START}-e${SAMPLE_END}` to chunked tags, preserving the full-N
canonical tag but preventing parallel sample collisions.

Two repaired same-model Cloudflare reruns started for the missing first and
third spans. At the 12:24 CDT poll,
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s3478-e4530_detail.jsonl`
was at 30/1052 and
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s5796-eend_detail.jsonl`
was at 34/1057. Both showed strict state filtering, retrieval-cache hits, and
document-cache hits on all parsed rows so far.
At 2026-05-21 12:32 CDT, those same repaired partials were still incomplete at
48/1052 and 53/1057, with no row-level structural violations. A foreground
relaunch with `UV_CACHE_DIR=/tmp/uv-cache` failed closed during OpenRouter
preflight smoke with `Connection error`, before new rows were written. The
next required action is to rerun only the first and third spans after API
reachability returns; the middle span is already preserved by the unsuffixed
10:14 detail log.

At 2026-05-21 10:14 CDT, the canonical same-model Gemma `rag_simple`
continuation launched for the three failed row spans with
`OPENROUTER_PROVIDER_ONLY=Cloudflare`, `NO_SILENT_FALLBACK=1`, jurisdiction
state filtering, and strict retrieval/doc-cache replay. Live artifacts:
`logs/eval_rag_simple_or-gemma4-26b_20260521_101450_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`,
`logs/run_housing_statefilter_gemma_rag_simple_resume_3478_4530_20260521_151441.out`,
`logs/run_housing_statefilter_gemma_rag_simple_resume_4634_5690_20260521_151441.out`,
and
`logs/run_housing_statefilter_gemma_rag_simple_resume_5796_end_20260521_151441.out`.
At 2026-05-21 10:28 CDT, the live detail log had 388 parsed rows with zero JSON
errors, row errors, missing final answer lines, retrieval-cache misses,
doc-cache misses, missing state filters, fallback markers, or think tags.

At 2026-05-21 09:39 CDT, a direct
`python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01 --retries 3 --retry-delay 2`
call failed closed on OpenRouter DNS resolution after bounded retries, before
budget lookup. No Gemma row was launched.
At 2026-05-21 09:40 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` found no
Housing launch locks, verified the focused gate remains `complete=6/9`, checked
the three Gemma `rag_simple` resume offsets and exact model label, confirmed
the current `rag_simple` merge gate still fails closed at 3691/6853, and then
stopped at the OpenRouter key-budget guard with `limit_remaining=0`. No Gemma
row was launched.
At 2026-05-21 09:43 CDT, a fresh direct
`python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01 --retries 3 --retry-delay 2`
check again reached OpenRouter and failed closed with `limit_remaining=0 <
required 0.01` (`usage=23.912568061`, `limit=20`). No Gemma row was launched.
At 2026-05-21 09:44 CDT,
`RUN_FINALIZER=0 REQUIRE_FULL_EXEMPLAR=0 scripts/local/verify_housing_statefilter_goal_complete.sh`
found no Housing launch locks, passed syntax/Python helper checks, refreshed
`current_status.md`, and failed at the expected focused audit with
`complete=6/9`. The three missing cells remain the `or-gemma4-26b`
`rag_simple`, `rag_hyde`, and `snap_hyre` rows.
At 2026-05-21 09:45 CDT,
`PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh` passed
local integrity checks and the exact model guard
(`or-gemma4-26b -> google/gemma-4-26b-a4b-it`), then failed closed at the
OpenRouter key-budget guard with `limit_remaining=0`. It skipped the full
launch lock and created no signoff-candidate file.
At 2026-05-21 09:46 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again found no
launch/watch locks, verified the focused gate remains `complete=6/9`, checked
the three Gemma `rag_simple` resume offsets and exact model label, confirmed
the current `rag_simple` merge gate still fails closed at 3691/6853, and then
failed closed on OpenRouter DNS before budget lookup or row launch. It created
no `docs/generated/housing_gemma_signoff_candidates_20260521_144610.md` file.
At 2026-05-21 09:47 CDT,
`python3 scripts/audit_housing_statefilter_goal.py --allow-incomplete`
reconfirmed the offline focused state: the six Groq rows are signed, full-N,
and clean under the state-filter/cache/fallback/final-answer checks; the only
missing core cells remain `or-gemma4-26b` `rag_simple` at 3691/6853,
`rag_hyde` at 0/6853, and `snap_hyre` at 0/6853.
At 2026-05-21 09:48 CDT, a bounded direct OpenRouter key check again reached
the API and failed closed with `limit_remaining=0 < required 0.01`
(`usage=23.912568061`, `limit=20`). No Gemma row was launched.
At 2026-05-21 09:49 CDT, offline
`scripts/local/check_housing_gemma_readiness.sh` passed. It found no Housing
launch locks, reconfirmed the focused gate at `complete=6/9`, verified the
three Gemma `rag_simple` resume offsets and exact model label without API
calls, listed the expected missing full-N Gemma HyDE/Snap caches, and confirmed
the current Gemma `rag_simple` merge gate still fails closed at 3691/6853 with
the three failed rows.
At 2026-05-21 09:50 CDT, `scripts/local/status_monitor.sh status` showed the
recurring `current_status.md` monitor is running (`pid=3627587`, interval
300s), and a lock/process check found no Housing launch locks and no active
Housing eval/cache/answer jobs besides that monitor.
At 2026-05-21 09:50 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again passed
offline checks and the exact model guard, then failed closed on OpenRouter DNS
after bounded retries before budget lookup or row launch. It created no
`docs/generated/housing_gemma_signoff_candidates_20260521_145054.md` file.
At 2026-05-21 09:52 CDT, local continuation syntax checks passed:
`bash -n` over the Housing Gemma launch/finalize/verify scripts and
`python3 -m py_compile` over the provider guards, Housing audits, status
updater, and signoff summarizer. This confirms the remaining blocker is not a
local script syntax failure; it remains the unavailable exact Gemma OpenRouter
budget/route.
At 2026-05-21 09:53 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again passed
offline checks and the exact model guard, then failed closed on OpenRouter DNS
after bounded retries before budget lookup or row launch. It created no
`docs/generated/housing_gemma_signoff_candidates_20260521_145351.md` file.
At 2026-05-21 09:56 CDT, `python3 scripts/update_current_status.py` refreshed
`current_status.md`; the dashboard remains `67/78` signed, `0/78` active, with
HousingQA still blocked at the same three missing `or-gemma4-26b` core rows.
At 2026-05-21 09:59 CDT, a local signoff hygiene check found no
`docs/generated/housing_gemma_signoff_candidates_*.md` or
`docs/generated/housing_gemma_exemplar_signoff_candidates_*.md` files. The only
Gemma state-filtered HousingQA row in `docs/signoff_log.md` remains the q500
`snap_hyre_exemplar` diagnostic, explicitly not a full-N core row.
At 2026-05-21 09:59 CDT, a filesystem scan reconfirmed the expected full-N
Gemma generated-method artifacts are still absent: no full
`or-gemma4-26b` Housing `rag_hyde`, `snap_hyre`, or `snap_hyre_exemplar`
generation/retrieval/doc-cache files exist at the canonical full-cache paths,
and no matching full-N generated-method answer detail logs were found.
At 2026-05-21 10:01 CDT, the Gemma `rag_simple` source-log scan was rerun with
the same state-filter selection rule used by
`scripts/local/merge_audit_housing_gemma_rag_simple.sh`. It found exactly five
state-filtered source logs: 88 clean rows, 3282 clean rows, and three shard logs
with one failed-closed row each. The only bad labels are still
`hqa_Nebraska_2941`, `hqa_Ohio_6341`, and `hqa_Texas_4530`.
At 2026-05-21 10:03 CDT,
`VERIFY_ONLY=1 scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`
verified the resume offsets again:
`3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and
`5796 -> hqa_Texas_4530`; it also accepted the exact Gemma model mapping and
exited before OpenRouter preflight, launch locks, or row writes.
At 2026-05-21 09:57 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again passed
offline checks and the exact model guard, then failed closed on OpenRouter DNS
after bounded retries before budget lookup or row launch. It created no
`docs/generated/housing_gemma_signoff_candidates_20260521_145701.md` file.
At 2026-05-21 10:01 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again passed
offline checks and the exact model guard, then failed closed on OpenRouter DNS
after bounded retries before budget lookup or row launch. It created no
`docs/generated/housing_gemma_signoff_candidates_20260521_150104.md` file.
At 2026-05-21 10:07 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again found no
Housing launch locks, verified the focused gate remains `complete=6/9`, checked
the three Gemma `rag_simple` resume offsets and exact model label, confirmed
the current `rag_simple` merge gate still fails closed at 3691/6853, and then
failed closed on OpenRouter DNS after bounded retries before budget lookup or
row launch. It created no
`docs/generated/housing_gemma_signoff_candidates_20260521_150713.md` file.
At 2026-05-21 10:09 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` repeated the
same non-launching gate: no Housing launch locks, focused gate still
`complete=6/9`, Gemma `rag_simple` resume offsets and exact model label
verified, current `rag_simple` merge gate still failed closed at 3691/6853, and
OpenRouter key-budget guard failed closed with `limit_remaining=0`
(`usage=23.912568061`, `limit=20`). No row was launched, and no
`docs/generated/housing_gemma_signoff_candidates_20260521_150919.md` file was
created.
At 2026-05-21 10:13 CDT,
`CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh` again passed
the local lock, focused-gate, resume-offset, and exact-model checks, then
failed closed at the OpenRouter key-budget guard with `limit_remaining=0`
(`usage=23.912568061`, `limit=20`). No row was launched, and no
`docs/generated/housing_gemma_signoff_candidates_20260521_151304.md` file was
created.

## Success Criteria

Required full-N core rows:

| Method | `groq-llama8b` | `or-gemma4-26b` | `groq-llama70b` |
|---|---|---|---|
| `rag_simple` | signed | blocked partial after OpenRouter 403 key-limit; resume/supersede failed chunk rows before merge | signed |
| `rag_hyde` | signed | not running; full generation/retrieval/doc caches and answer row still required after exact-model route is restored | signed |
| `snap_hyre` | signed | not running; full generation/retrieval/doc caches and answer row still required after exact-model route is restored | signed |

Required diagnostic:

| Diagnostic | Status | Promotion rule |
|---|---|---|
| Gemma 26B HousingQA `snap_hyre_exemplar` q500 answer gate | complete and audited clean for canonical and exemplar | Exemplar improves retrieval exposure and is answer-parity, not answer-lift: +3.0pp Hit@5, +0.0218 MRR@5, answer -0.2pp vs canonical with p=1.0. Treat as scale-eligible after required core rows, not as a higher-priority replacement for core methods. |
| Gemma 26B HousingQA `snap_hyre_exemplar` full-N scale-up | queued after the three Gemma core rows | Because the q500 gate is clean and retrieval-positive with answer parity, full-N scale-up is now tracked as a post-core diagnostic requirement. It is launched by `scripts/local/run_housing_gemma_exemplar_full_after_core.sh` and audited by `scripts/local/audit_housing_gemma_exemplar_full.sh`. |

## Requirement-To-Evidence Checklist

| Objective requirement | Evidence artifact or command | Current state |
|---|---|---|
| Full-N HousingQA `rag_simple`, `rag_hyde`, `snap_hyre` across `groq-llama8b`, `or-gemma4-26b`, and `groq-llama70b` | `current_status.md`, `docs/signoff_log.md`, detail logs named in this file | 6/9 state-filtered core rows signed; `or-gemma4-26b` `rag_simple` is blocked at 3691 deduped rows with 3 failed-closed OpenRouter key-limit rows; Gemma `rag_hyde` and `snap_hyre` are not running and are blocked behind the same exact-model OpenRouter route. |
| Correct citation block for signed rows | `docs/signoff_log.md` | Use the later `HousingQA state-filtered` entries for the six signed Groq core rows. Older unfiltered HousingQA signoff rows remain in the file for provenance but are not the current state-filtered core matrix. |
| No false Gemma signoff promotion | `docs/generated/housing_gemma_signoff_candidates_*.md`; `docs/signoff_log.md` | As of 2026-05-21 09:59 CDT, no Housing Gemma signoff-candidate files exist, and the only Gemma state-filtered HousingQA signoff row is the q500 `snap_hyre_exemplar` diagnostic marked diagnostic-only. No full-N Gemma core row has been promoted. |
| Jurisdiction state filtering for every retrieval method | `scripts/audit_housing_statefilter_detail.py` and row fields `housing_state_filter=true`, `retrieval_where={"state": ...}` | Passed on signed rows and current partial audits; must be rerun on every full merged log before signoff. |
| No silent provider/model fallback | Runner logs must show `no_silent_fallback=1`; detail audits must show no fallback keys; `rg -n "export NO_SILENT_FALLBACK|NO_SILENT_FALLBACK=1" scripts/local/run_housing_gemma_core_queue.sh scripts/local/run_housing_gemma_exemplar_full_after_core.sh scripts/local/run_housing_gemma_after_key_reset.sh scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh` | Confirmed for signed rows, active Gemma `rag_simple`, and Gemma q500 canonical/exemplar; must be checked again at full row signoff. As of 2026-05-21 07:58 CDT, the Gemma core queue and full-N exemplar wrapper force `NO_SILENT_FALLBACK=1` even if the parent environment sets it false, and their generation-cache postchecks use broad fallback detection plus exact 6853-row checks before retrieval-cache construction. As of 2026-05-21 08:04 CDT, the post-run Gemma core and full-exemplar audit scripts also use broad fallback detection for non-boolean fallback markers, and the full-exemplar launcher/audit export `PROVIDER`/`MODEL_LABEL` before the exact-model guard. As of 2026-05-21 08:14 CDT, a temp `.env` probe confirmed `scripts/local/run_answer_cell.sh` restores explicit launcher values for `LLM_MAX_COMPLETION_TOKENS`, `EVAL_MIN_COMPLETION_TOKENS`, `NO_SILENT_FALLBACK`, and `OPENROUTER_PROVIDER_ONLY` after sourcing `.env`, so stale local `.env` values cannot silently weaken a strict launcher. As of 2026-05-21 08:19 CDT, a `uv run` dotenv probe confirmed generation-cache Python preserves exported strict values (`LLM_PROVIDER=or-gemma4-26b`, `NO_SILENT_FALLBACK=1`, `OPENROUTER_PROVIDER_ONLY=Cloudflare`) over hostile dotenv values. |
| OpenRouter budget preflight for Gemma launchers | `scripts/check_openrouter_key_status.py --min-limit-remaining 0.01`; launcher preflights in `scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`, `scripts/local/run_housing_gemma_core_queue.sh`, and q500 run-on-demand branch | Latest safe-fail modes are OpenRouter `limit_remaining=0` when the key endpoint is reachable, or transient DNS failure before budget lookup when it is not. Earlier Gemma row logs also show the row-level blocker as OpenRouter 403 key-limit failures. Gemma OpenRouter launchers still cannot run here. As of 2026-05-21 08:02 CDT, `scripts/check_openrouter_key_status.py` has bounded transient retry knobs (`--retries`, `--retry-delay`; defaults 3 attempts and 5s) so DNS/5xx/429 hiccups do not waste a reset window, while missing keys, invalid data, and insufficient `limit_remaining` still fail closed. As of 2026-05-21 08:08 CDT, `scripts/local/run_housing_gemma_after_key_reset.sh` also runs shell syntax and Python compile checks before exact-model/API preflights, so broken local launch/audit code fails before any row work. As of 2026-05-21 09:34 CDT, a direct `python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01 --retries 3 --retry-delay 2` call reached OpenRouter and failed closed with `limit_remaining=0 < required 0.01` (`usage=23.912568061`, `limit=20`). A masked environment check found no alternate OpenRouter key in the process environment, and `.env` contains only `OPENROUTER_API_KEY`, so the current blocker is account budget on the configured key rather than a missing alternate key variable. `PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh` remains the non-launching route/budget check and exits before any row launch or candidate-file creation when blocked. |
| OpenRouter chat-route preflight | `scripts/check_openrouter_chat_route.py`; wired into `scripts/local/run_housing_gemma_after_key_reset.sh`, `scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`, `scripts/local/run_housing_gemma_core_queue.sh`, `scripts/local/run_housing_gemma_exemplar_full_after_core.sh`, and `scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh` after their budget guards | Added 2026-05-21 as a tiny exact-route completion smoke before any row work. It resolves the configured provider model, preserves the strict canonical/free-suffix guard, disables provider fallbacks by default, honors `OPENROUTER_PROVIDER_ONLY`, `OPENROUTER_PROVIDER_ORDER`, and `OPENROUTER_PROVIDER_IGNORE`, and requires the response content to be exactly `OK`. `python3 -m py_compile scripts/check_openrouter_chat_route.py scripts/check_expected_provider_model.py scripts/check_openrouter_key_status.py` and `bash -n` over the Housing Gemma launch helpers pass. Offline fail-closed probes pass: an empty `OPENROUTER_API_KEY` exits before network work, and `or-gemma4-26b-free` is rejected without `OPENROUTER_ALLOW_FREE_SUFFIX=1`. A blocked canonical `PREFLIGHT_ONLY=1 RUN_OPENROUTER_CHAT_PREFLIGHT=1 scripts/local/run_housing_gemma_after_key_reset.sh` still stops at the key-budget guard before the chat smoke or signoff-file creation, as expected while `limit_remaining=0`. A direct blocked `RUN_OPENROUTER_CHAT_PREFLIGHT=1 scripts/local/run_housing_gemma_core_queue.sh` also stops at the key-budget guard, then removes its launch lock. The explicit `:free` Cloudflare smoke fails closed with 404 because OpenRouter lists only `google-ai-studio` for the free model. |
| Exact same-model fallback check | `scripts/check_expected_provider_model.py`, `llm_config.py`, non-generating Google OpenAI-compatible `/models` check | No configured exact-model fallback is available for canonical launches. `MODEL_LABEL=or-gemma4-26b python3 scripts/check_expected_provider_model.py --provider or-gemma4-26b --expected-model google/gemma-4-26b-a4b-it --expected-label or-gemma4-26b` accepts the canonical route. Direct Google `gemma` is rejected because it resolves to `gemma-3-27b-it`; the current Google key also returned `API_KEY_INVALID` on the model-list check. The configured `or-gemma4-26b-free` alias was checked on 2026-05-21 and is still rejected by default because it resolves to `google/gemma-4-26b-a4b-it:free`, not the canonical `google/gemma-4-26b-a4b-it` model id. As an explicit noncanonical escape hatch only, `scripts/check_expected_provider_model.py --allow-openrouter-free-suffix` or `OPENROUTER_ALLOW_FREE_SUFFIX=1` now accepts that `:free` suffix when the base model exactly matches; a non-launching after-reset preflight with `PROVIDER=or-gemma4-26b-free MODEL_LABEL=or-gemma4-26b OPENROUTER_ALLOW_FREE_SUFFIX=1` passed local integrity checks and this opt-in guard at 2026-05-21 08:37 CDT, then failed closed at the same OpenRouter budget check before row launch. Direct tiny completion smokes at 08:42 CDT showed the `:free` route is not currently usable: unconstrained `google/gemma-4-26b-a4b-it:free` returned upstream Google AI Studio 429, and Cloudflare-pinned `:free` returned 404 because OpenRouter listed only `google-ai-studio` as available. The default remains fail-closed, and the Housing launchers still use the strict canonical check unless an operator deliberately changes the experiment label/route and signoff caveat. Do not substitute Gemma 3, Gemini, direct Google, or the `:free` OpenRouter alias for the remaining `or-gemma4-26b` rows unless the experiment label and signoff explicitly change. |
| Live alternate-key/route check | Masked environment scan plus `llm_config.py` provider map | As of 2026-05-21 09:15 CDT, the live environment did not expose an alternate approved Google/Gemini/OpenRouter key variable for the remaining canonical row. `llm_config.py` still maps direct Google `gemma` to `gemma-3-27b-it`, so direct Google remains a model substitution rather than an exact fallback for `google/gemma-4-26b-a4b-it`. |
| Strict retrieval-cache replay | `REQUIRE_RETRIEVAL_CACHES=1`; detail rows show `retrieval_cache_hit=true`; retrieval cache audits report complete rows | Passed on signed rows and active partials; full-row rerun required when rows finish. |
| Strict document-cache replay | `RETRIEVAL_DOC_CACHE_STRICT=1`; detail rows show `retrieval_doc_cache_hit=true`; `scripts/audit_housing_statefilter_detail.py` | Passed on signed rows and active partials; full-row rerun required when rows finish. |
| HyDE/Snap-HyRE generation cache replay | `hyre_cache_hit=true`; `--require-hyre-cache` in the Housing audit | Passed on signed HyDE/Snap-HyRE rows and active partial audits; required for remaining HyDE/Snap-HyRE rows. |
| Valid final answer lines | `scripts/analyze_detail_flags.py`; `scripts/audit_housing_statefilter_detail.py` checks the last non-empty final line exactly matches the parsed prediction: `Answer: Yes` or `Answer: No` | Passed on signed rows and current partial q500 audit. |
| No fallback, think-token, or truncation artifacts | `scripts/analyze_detail_flags.py`, Housing audit, near-cap output checks | Passed on signed rows and active partial q500 audit; watch answer-format retries as caveats, not failures if final rows remain clean. The Housing detail audit now treats non-empty fallback lists/strings as failures, matching the signoff summarizer. |
| Merged-log audits for repaired/tail runs | `scripts/merge_detail_logs.py --on-duplicate last`, then audit the merged output | Required for Gemma `rag_simple`; completed earlier for 70B `snap_hyre`. |
| Gemma 26B q500 exemplar diagnostic | Canonical and exemplar q500 detail logs, retrieval caches, paired comparison via `scripts/compute_mcnemar.py --key label` | Complete and clean. Re-audited at 2026-05-21 08:25 CDT. Canonical: 315/500 = 63.0%, Hit@1/5/10 0.1640/0.3820/0.4560, MRR@5/10 0.2429/0.2528. Exemplar: 314/500 = 62.8%, Hit@1/5/10 0.1840/0.4120/0.4960, MRR@5/10 0.2647/0.2760. Paired answer comparison remains -0.2pp, b/c=36/37, p=1.0, CI [-3.6, 3.005]. |
| Updated operational status | `python3 scripts/update_current_status.py`; focused docs updated after meaningful checkpoints | `current_status.md` refreshed at 2026-05-21 10:13 CDT after the 10:13 non-launching network readiness check passed local/exact-model gates and failed closed at the OpenRouter budget guard with `limit_remaining=0`. The dashboard remains `67/78` signed and `0/78` active. The six signed Groq Housing rows remain clean under the current state-filter detail validator, and the q500 canonical/exemplar diagnostic remains clean with exemplar retrieval lift and answer parity. Earlier direct OpenRouter key/budget checks at 08:27 CDT, 09:00 CDT, 09:26 CDT, 09:34 CDT, 09:40 CDT, 09:43 CDT, 10:09 CDT, and 10:13 CDT, plus watcher preflights at 09:10, 09:19, and 09:22 CDT, reached OpenRouter and failed closed with `limit_remaining=0`; other sandbox checks, including the 09:46, 09:50, 09:53, 09:57, 10:01, and 10:07 readiness reruns, failed closed earlier on DNS before budget lookup. A 09:45 CDT canonical preflight-only continuation passed local integrity and the exact model guard, then failed closed at the same key-budget guard before any launch lock or signoff-candidate file. No Gemma answer work was launched after the latest checks, and the refreshed dashboard shows no active row. This checklist and `docs/housingqa_handoff_to_next_agent_2026-05-21.md` are aligned to the blocked/no-active-jobs checkpoint. |
| Focused goal audit | `python3 scripts/audit_housing_statefilter_goal.py` | Added as a machine-checkable 9-cell gate. Current output is `complete=6/9` and exits nonzero because the three `or-gemma4-26b` core rows are incomplete. |
| Signoff row helper | `python3 scripts/summarize_housing_statefilter_signoff.py <detail> --provider <provider> --mode <mode>` | Added to generate signoff-ready Markdown rows after a full detail log passes strict gates. It accepts a signed 8B `rag_simple` state-filter row and rejects the blocked Gemma prefix (`expected 6853 rows, found 88`), so partial prefixes cannot become signoff rows. |
| Gemma signoff finalizer | `scripts/local/finalize_housing_gemma_signoff.sh` | Added as a post-run helper. It finds full clean Gemma state-filter detail logs, appends missing signoff rows only after `summarize_housing_statefilter_signoff.py` accepts them, refreshes `current_status.md`, then runs `scripts/audit_housing_statefilter_goal.py`. The exemplar signoff path now first requires the 9-cell core audit to pass (`REQUIRE_CORE_COMPLETE_FOR_EXEMPLAR=1` by default), so `snap_hyre_exemplar` cannot be appended before the core rows are done. A 2026-05-21 08:09 CDT temp-signoff smoke test with `MODES=snap_hyre_exemplar` exited 1 at the 6/9 core audit and appended 0 bytes. |
| Gemma readiness wrapper | `scripts/local/check_housing_gemma_readiness.sh`; set `CHECK_NETWORK=1` to include OpenRouter preflight | Added as a read-only launch check. Offline mode passed again at 2026-05-21 10:07 CDT: it checks both legacy `/tmp` and repo-local `logs/monitors/locks/` Housing lock paths, no launch/watch locks are present, the focused 9-cell gate is still 6/9, failed-row resume offsets and exact Gemma model label verify, full Gemma HyDE/Snap caches are missing as expected, and the blocked `rag_simple` merge gate fails closed as expected with 3691/6853 rows. Network mode at 10:07 CDT reached the `PREFLIGHT_ONLY=1` after-reset wrapper, passed local integrity checks and the exact-model guard, and failed closed on OpenRouter DNS resolution after bounded retries before any budget lookup, row launch, or candidate-file creation. Earlier network reruns alternated between this DNS safe-fail mode and the OpenRouter key-budget safe-fail mode with `limit_remaining=0`; direct `getent hosts` also returned no OpenRouter host records in this sandbox. |
| Canonical continuation sequence | `scripts/local/run_housing_gemma_after_key_reset.sh`; `scripts/local/run_housing_gemma_core_queue.sh` | Rechecked from source at 2026-05-21 09:54 CDT. The after-reset wrapper resumes/merges/audits Gemma `rag_simple`, launches `run_housing_gemma_core_queue.sh`, audits `rag_hyde` and `snap_hyre`, finalizes signoff, and then launches full-N `snap_hyre_exemplar` by default. The core queue explicitly calls both `build_and_run_mode rag_hyde hyde_cache hyde` and `build_and_run_mode snap_hyre hyre_cache snap_hyre`, so one canonical continuation covers both missing generated core rows once the exact-model preflight clears. |
| Gemma budget watcher | `scripts/local/watch_housing_gemma_until_ready.sh`; manager: `scripts/local/housing_gemma_budget_watcher.sh status\|start\|stop` | Added as an optional polling helper for reset windows. It is non-launching by default (`LAUNCH_ON_READY=0`), checks the four Housing launch locks before each poll, runs the existing after-reset wrapper in `PREFLIGHT_ONLY=1` mode, and logs to `logs/monitors/`. A stale-lock cleanup bug exposed by a one-attempt dry run was fixed with explicit cleanup on normal exits; a repeat dry run at 2026-05-21 09:09 CDT exited with code 20 and removed its lock. Later stale lock/process mismatches showed `/tmp` locks were not durable enough for this Codex-started detached watcher, so managed watcher locks now live under `logs/monitors/locks/` while `current_status.md` still checks both repo-local and legacy `/tmp` lock paths. Launch-on-ready watcher attempts at 09:10, 09:19, and 09:22 CDT all passed the exact model guard, failed closed at OpenRouter `limit_remaining=0`, created no Housing launch locks, answer rows, or signoff-candidate files, and then left dead-pid stale locks. The stale locks were removed after `ps` or the watcher manager showed pids `3636546`, `3643843`, and `3646233` were absent. A 09:31 CDT foreground non-launching probe failed closed at `limit_remaining=0`, removed its lock, and exposed a false-positive process matcher in the manager; that matcher now only reports actual `bash scripts/local/watch_housing_gemma_until_ready.sh` processes. As of the 09:31 CDT validation, no watcher process and no repo-local or `/tmp` watcher lock are present. |
| Final completion gate | `scripts/local/verify_housing_statefilter_goal_complete.sh` | Added as the strict "are we actually done?" command. It fails if launch locks are present, runs syntax/Python helper checks, runs the Gemma finalizer, refreshes `current_status.md`, requires the full-N exemplar audit/finalizer by default, and then runs `scripts/audit_housing_statefilter_goal.py` without `--allow-incomplete`. It now also compiles the OpenRouter provider/model/key/chat-route guard scripts. A no-finalizer/no-exemplar smoke at 2026-05-21 09:44 CDT found no launch locks, passed syntax/Python checks, refreshed `current_status.md`, then failed at the expected `complete=6/9` focused-audit gate. Blocked-state check at 2026-05-21 09:08 CDT found no launch locks, passed syntax/Python helper checks, then failed closed at Gemma finalization because no full clean Gemma `rag_simple` detail log exists. |

## Completion Audit

Objective restated as concrete deliverables:

1. Produce full-N HousingQA state-filtered answer rows for
   `rag_simple`, `rag_hyde`, and `snap_hyre` across `groq-llama8b`,
   `or-gemma4-26b`, and `groq-llama70b`.
2. For every promoted row, verify strict retrieval/doc cache replay, state
   filtering, no silent fallback/model swap, no think-token artifacts, valid
   final `Answer: Yes/No` lines, and merged-log audits when repaired.
3. Complete the Gemma 26B q500 `snap_hyre_exemplar` diagnostic and record
   retrieval and answer-parity/lift evidence.
4. Scale Gemma 26B `snap_hyre_exemplar` to full-N after the core Gemma rows
   because q500 was clean and retrieval-positive with answer parity.
5. Keep `current_status.md` plus this focused Housing handoff current with
   completion, r_acc@5, mrr@5, f_acc, active jobs, and caveats.

Current audit result: **not complete**. The six Groq rows and the q500 Gemma
exemplar diagnostic are signed/clean, but the three `or-gemma4-26b` full-N core
rows and the full-N Gemma exemplar scale-up remain incomplete. The latest
row-level Gemma failure is still OpenRouter `403 Key limit exceeded`; recent
sandbox preflights either report `limit_remaining=0` when OpenRouter is
reachable or fail closed on DNS before budget lookup. No Gemma OpenRouter answer
work can be launched from this environment until API reachability and budget are
restored while preserving the exact `google/gemma-4-26b-a4b-it` route.

| Deliverable | Artifact/evidence inspected | Status |
|---|---|---|
| 8B `rag_simple`, `rag_hyde`, `snap_hyre` full-N state-filter rows | `docs/signoff_log.md` rows for HousingQA state-filtered `groq-llama8b`; `current_status.md` Housing matrix | Complete/signed. |
| 70B `rag_simple`, `rag_hyde`, `snap_hyre` full-N state-filter rows | `docs/signoff_log.md` rows for HousingQA state-filtered `groq-llama70b`; `current_status.md` Housing matrix | Complete/signed. |
| Gemma 26B `rag_simple` full-N state-filter row | Five partial detail logs listed below; `current_status.md`; blocked-state merge gate | Incomplete: 3691/6853 deduped rows, with 3 failed-closed key-limit rows that must be superseded before merge/signoff. |
| Gemma `rag_simple` state-filter source logs | `scripts/local/merge_audit_housing_gemma_rag_simple.sh` source selection; `logs/eval_rag_simple_or-gemma4-26b_*housing*rag_simple*nfull*k5_detail.jsonl` | Rechecked 2026-05-21 10:01 CDT with the merge helper's `housing_state_filter=true` selection rule. Exactly five source logs qualify: 88 clean rows, 3282 clean rows, and three shard logs with one failed-closed row each. Only `hqa_Nebraska_2941`, `hqa_Ohio_6341`, and `hqa_Texas_4530` need same-model superseding. |
| Gemma `rag_simple` resume-offset guard | `VERIFY_ONLY=1 scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh` | Rechecked 2026-05-21 10:03 CDT. The script verifies `3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and `5796 -> hqa_Texas_4530`, accepts `or-gemma4-26b -> google/gemma-4-26b-a4b-it`, and exits before OpenRouter preflight or row launch. |
| Gemma 26B `rag_hyde` full-N state-filter row | Cache search for full `housing*qfull*or-gemma4-26b*rag_hyde*`; `current_status.md` | Not started for full-N; no full generation/retrieval/doc cache or answer row found. |
| Gemma 26B `snap_hyre` full-N state-filter row | Cache search for full `housing*qfull*or-gemma4-26b*snap_hyre*`; `current_status.md` | Not started for full-N; no full generation/retrieval/doc cache or answer row found. |
| Gemma 26B `snap_hyre_exemplar` full-N state-filter diagnostic | Cache search for full `housing*qfull*or-gemma4-26b*snap_hyre_exemplar*`; `scripts/local/run_housing_gemma_exemplar_full_after_core.sh`; `scripts/local/audit_housing_gemma_exemplar_full.sh` | Pending after the three Gemma core rows. No full-N exemplar generation/retrieval/doc cache or answer row exists yet. |
| Full-N Gemma generated-method filesystem scan | Canonical cache paths under `caches/hyre/full`, `caches/generation/full`, `caches/retrieval/full`, `caches/retrieval_doc/full`; matching `logs/*or-gemma4-26b*housing*{rag_hyde,snap_hyre,snap_hyre_exemplar}*nfull*k5*detail.jsonl` | Rechecked 2026-05-21 09:59 CDT: all expected full-N Gemma `rag_hyde`, `snap_hyre`, and `snap_hyre_exemplar` cache paths are absent, and no matching full-N generated-method answer detail logs exist. |
| No silent fallback/model swap | `llm_config.py`; `scripts/check_expected_provider_model.py`; `scripts/check_openrouter_key_status.py`; launcher guards; row-level signoff caveats | Guarded. Current exact-model route is blocked; direct Google is not an acceptable substitute in this environment. |
| q500 Gemma `snap_hyre_exemplar` diagnostic | Canonical and exemplar q500 logs/caches; signoff row; idempotent q500 gate rerun | Complete and clean; retrieval improved, answer parity not lift. |
| Status/handoff docs | `current_status.md`, this checklist, `docs/barexam_housing_core_focus_2026-05-20.md` | Updated to the 2026-05-21 10:11 CDT blocked/no-active-jobs state; no watcher lock is currently present. |
| Machine-checkable completion gate | `python3 scripts/audit_housing_statefilter_goal.py` | Fails as expected with `complete=6/9`; use after Gemma rows finish to verify the goal before marking complete. |
| Signoff helper | `python3 scripts/summarize_housing_statefilter_signoff.py` | Produces a signoff-log table row only for signoff-ready Housing state-filter logs; rejects the current blocked Gemma partial. |
| Gemma finalizer | `scripts/local/finalize_housing_gemma_signoff.sh` | Appends missing Gemma signoff rows only after clean full-N logs exist, then refreshes status and re-runs the focused completion gate. |

## Signed Rows

| Row | Detail log | Required audit evidence |
|---|---|---|
| HousingQA state-filtered `groq-llama8b` `rag_simple` | `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/audit_housing_statefilter_detail.py --provider groq-llama8b --mode rag_simple ...` passes. |
| HousingQA state-filtered `groq-llama8b` `rag_hyde` | `logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/audit_housing_statefilter_detail.py --provider groq-llama8b --mode rag_hyde --require-hyre-cache ...` passes. |
| HousingQA state-filtered `groq-llama8b` `snap_hyre` | `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/analyze_detail_flags.py`, `scripts/audit_housing_statefilter_detail.py --provider groq-llama8b --mode snap_hyre --expected-rows 6853 --require-hyre-cache ...`, and `scripts/audit_retrieval_cache.py` pass. |
| HousingQA state-filtered `groq-llama70b` `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/audit_housing_statefilter_detail.py --provider groq-llama70b --mode rag_simple ...` passes. |
| HousingQA state-filtered `groq-llama70b` `rag_hyde` | `logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/analyze_detail_flags.py`, `scripts/audit_housing_statefilter_detail.py --provider groq-llama70b --mode rag_hyde --expected-rows 6853 --require-hyre-cache ...`, and `scripts/audit_retrieval_cache.py` pass. |
| HousingQA state-filtered `groq-llama70b` `snap_hyre` | `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl` | Signed in `docs/signoff_log.md`; `scripts/audit_housing_statefilter_detail.py --provider groq-llama70b --mode snap_hyre --require-hyre-cache ...` passes. |

## Active Or Queued Rows

| Row | Current source/log | Completion action |
|---|---|---|
| `or-gemma4-26b` `rag_simple` | Prefix `logs/eval_rag_simple_or-gemma4-26b_20260520_230419_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`; stopped Cloudflare tail `logs/eval_rag_simple_or-gemma4-26b_20260520_233010_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`; failed-closed chunk logs `logs/eval_rag_simple_or-gemma4-26b_20260521_062626_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`, `logs/eval_rag_simple_or-gemma4-26b_20260521_062627_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`, and `logs/eval_rag_simple_or-gemma4-26b_20260521_062628_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl` | Blocked by OpenRouter 403 key-limit on `hqa_Nebraska_2941`, `hqa_Ohio_6341`, and `hqa_Texas_4530`. After the key/account is reset or replaced, run `scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`; it runs the three recovery spans in parallel by default (`RUN_PARALLEL=0` for sequential) from `SAMPLE_START=3478 SAMPLE_END=4530`, `SAMPLE_START=4634 SAMPLE_END=5690`, and `SAMPLE_START=5796`. Then run `scripts/local/merge_audit_housing_gemma_rag_simple.sh`; if clean, add the signoff row with a same-model Cloudflare route/chunking caveat and refresh `current_status.md`. |
| `or-gemma4-26b` `rag_hyde` | Not running; blocked behind OpenRouter key-limit resolution for Gemma core queue | Build full generation cache, state-filter retrieval/doc caches, answer row, then audit/sign after the exact same-model route is available again. |
| `or-gemma4-26b` `snap_hyre` | Not running; blocked behind OpenRouter key-limit resolution for Gemma core queue | Build full generation cache, state-filter retrieval/doc caches, answer row, then audit/sign after the exact same-model route is available again. |

Blocked-state gate check: `OUT=/tmp/housing_or-gemma4-26b_rag_simple_statefilter_blocked_gate_test_20260521_122202.jsonl scripts/local/merge_audit_housing_gemma_rag_simple.sh` correctly failed with exit code 1 on 2026-05-21 07:22 CDT after selecting only the five state-filtered detail logs: merged rows=3691, errors=3, missing predictions=3, empty retrieval=3, missing state filter=3, retrieval-cache misses=3, doc-cache misses=3, bad evidence length=3, missing exact final=3, fallback=3, and `expected 6853 rows, found 3691`. The older 2026-05-20 unfiltered Gemma `rag_simple` provenance logs are intentionally excluded from this merge gate.

Resume-offset guard: `scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`
now verifies against `datasets/housing_qa/questions.csv` that the three resume
starts map to the failed rows before it checks the provider/model and key:
`3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and
`5796 -> hqa_Texas_4530`. This is an offline guard before any API preflight, so
it should catch stale resume offsets before a recovered key starts spending.

Provider-route note: the canonical Gemma row remains `or-gemma4-26b`, which maps
to OpenRouter model `google/gemma-4-26b-a4b-it` with
`OPENROUTER_PROVIDER_ONLY=Cloudflare`, `allow_fallbacks=false`, and
`require_parameters=true`. The repo also has `or-gemma4-26b-free`, but using that
would be an explicit route/model-label caveat and is not a silent substitute for
the canonical row.
The existing Gemma `rag_simple` row has an 88-row default OpenRouter prefix plus
a Cloudflare tail/chunk continuation. That route mix is acceptable only as an
explicit same-model caveat, not as a silent model swap. The signoff summarizer
can require `--require-openrouter-provider-only Cloudflare` for fully Cloudflare
rows such as generated-method diagnostics, but the Gemma finalizer intentionally
does not require that flag for all `rag_simple` rows so it can preserve the
known same-model prefix.

Single-command continuation after key/API reachability reset:
Run `scripts/local/check_housing_gemma_readiness.sh` first. It is read-only by
default. Then run `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`
to verify OpenRouter API/budget readiness without launching rows.

`scripts/local/run_housing_gemma_after_key_reset.sh` first runs the OpenRouter
model-identity guard and key-budget preflight, then
`resume_housing_gemma_rag_simple_after_key_reset.sh`, then
`merge_audit_housing_gemma_rag_simple.sh`, then the Gemma `rag_hyde`/`snap_hyre`
core queue, then `scripts/local/audit_housing_gemma_core_rows.sh` for the
generated rows, then `scripts/local/finalize_housing_gemma_signoff.sh`, then
`scripts/local/run_housing_gemma_exemplar_full_after_core.sh` unless
`RUN_FULL_EXEMPLAR_AFTER_CORE=0` is set. The finalizer appends missing Gemma
core signoff rows only after
`scripts/summarize_housing_statefilter_signoff.py` accepts full clean logs,
refreshes `current_status.md`, and runs the final 9-cell completion audit. While
OpenRouter quota or API reachability is blocked, this wrapper exits before
launching any answer row or creating a signoff-candidate file; recent network
preflights either fail closed at the budget guard with `limit_remaining=0` or
fail closed on DNS before budget lookup.
For a real full launch, the wrapper now acquires
`/tmp/housing_gemma_after_key_reset.lock` before the model and budget preflights.
The lock records pid, timestamp, provider, route, signoff-output path, and
command metadata, and exits with code 11 if another full launch appears active.
`PREFLIGHT_ONLY=1` deliberately skips this lock so operators can run non-launching
route/budget checks without blocking a later full restart.
The direct helpers also fail closed on duplicate direct launches:
`resume_housing_gemma_rag_simple_after_key_reset.sh` uses
`/tmp/housing_gemma_rag_simple_resume.lock`, and
`run_housing_gemma_core_queue.sh` uses `/tmp/housing_gemma_core_queue.lock`.
The full exemplar scale-up uses `/tmp/housing_gemma_exemplar_full.lock`.
The readiness wrapper reports all four lock directories and exits with code 12 if
any are present, unless `ALLOW_EXISTING_LOCKS=1` is set for read-only inspection
after an operator verifies whether the lock is stale.
To verify the exact route and OpenRouter budget without launching rows after a
reset, run:
`PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh`.
After the Gemma continuation finishes, run the strict completion gate:
`scripts/local/verify_housing_statefilter_goal_complete.sh`. This command is
supposed to fail until all three Gemma core rows and the full-N
`snap_hyre_exemplar` diagnostic are full-N and clean; in the blocked state it
currently stops at the finalizer with "no full clean state-filter detail log
found for mode=rag_simple." When it passes, it has also refreshed
`current_status.md`, finalized the full exemplar signoff row, and run the
focused 9-cell core audit without `--allow-incomplete`.

Queue hardening note: `scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh`
now skips any q500 mode that already has 500 clean state-filtered/cache-replay
rows, and `scripts/local/run_housing_gemma_core_queue.sh` waits for active
exemplar queues before launching Gemma full-cache builds.
The Gemma core queue now passes the exact `QUESTIONS`, `SEED`, `RETRIEVAL_K`,
`HYRE_CACHE_ROOT`, `RETRIEVAL_CACHE_ROOT`, retrieval cache, and document cache
paths into `scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh`,
and that wrapper forwards the cache roots into `run_answer_cell.sh`. A
missing-cache smoke test with explicit `/tmp` cache roots printed the expected
root paths and failed on the missing retrieval cache before answer replay, so
answer runs cannot silently split between custom hydration paths and default
answer-cache lookup roots.
The core queue lock was also tested with a forced-impossible budget preflight
(`OPENROUTER_MIN_LIMIT_REMAINING=999999 WAIT_PATTERNS=""`): it acquired
`/tmp/housing_gemma_core_queue.lock`, failed at the OpenRouter budget guard
before any cache or answer work, and removed the lock on exit.
The Housing queue scripts now preserve intentionally empty `WAIT_PATTERNS` /
`WAIT_PATTERN` values, so a full-permission restart can bypass stale tmux-name
waits after the operator has already audited that no relevant jobs are active.
`scripts/local/audit_housing_gemma_core_rows.sh` now provides the final
post-run gate for Gemma `rag_simple`, `rag_hyde`, and `snap_hyre`: it merges
state-filtered full-N detail logs, checks generation-cache health for generated
methods, reruns `analyze_detail_flags.py`, runs
`audit_housing_statefilter_detail.py`, and audits the retrieval cache before any
signoff row is added. Its detail-log selector keys on the actual provider plus
the stable `MODEL_LABEL`, matching `run_answer_cell.sh` tags if an explicit
same-model route is ever used without silently changing the reported row label.
Clean rows are passed through `scripts/summarize_housing_statefilter_signoff.py`;
the summarizer rejects incomplete or bad detail logs by default.
This lets the q500 exemplar gate run as soon as a Groq slot opens without
duplicating it later. The Groq core queue `housing_groq_core_queue_20260521_060520`
finished and signed 70B `rag_hyde`, but it stopped before launching 8B
`snap_hyre` because the already-complete 70B run printed a post-OK
`run_answer_cell.sh` shell EOF message. Current `bash -n` checks pass for
`scripts/local/run_answer_cell.sh` and `scripts/local/run_housing_groq_core_queue.sh`.
The remaining 8B `snap_hyre` state-filter row was relaunched explicitly from
the prepared caches as tmux `housing_8b_snap_hyre_statefilter_20260521_091729`
and is now complete, audited, and signed.
The older waiting-only Gemma follow-up/core sessions from
`043157` and `043502` were killed and restarted at `0017` so the live sleepers
used these hardened queue scripts. After canonical q500 finished, the first
exemplar attempt exposed a path resolver bug: `snap_hyre_exemplar` checked the
state-filtered real-passage generation filename, then fell back to a
non-realpassage filename before trying the existing unfiltered real-passage
generation cache. `scripts/local/run_answer_cell.sh` now checks the unfiltered
real-passage generation cache first. The duplicate follow-up queue
`housing_gemma_followup_queue_20260521_0017` and old Gemma core queue
`housing_gemma_core_queue_20260521_0017` were killed; the later Gemma core queue
`housing_gemma_core_queue_20260521_073328` exited before doing API work, and no
Gemma core queue is currently live. `bash -n` now passes on the Housing queue
scripts, including the Gemma audit helper, and `python3 -m py_compile` passes
on the status/provider guard scripts, so the restart path can be relaunched
after the exact same-model OpenRouter route is available again. The older
Groq core queues from `044000` and `0019` were killed/restarted for the same
reason; the latest queue `housing_groq_core_queue_20260521_060520` has now
stopped after the post-OK 70B HyDE wrapper caveat; the remaining 8B
`snap_hyre` row was completed in the explicit tmux session above.

## Latest Live Snapshot

Last manual refresh: 2026-05-21 10:11 CDT.

| Row | Progress | Health |
|---|---:|---|
| `groq-llama70b` `rag_hyde` | complete and signed, 4263/6853 = 62.2% | Strict full-row audits pass: Hit@5 0.3495, MRR@5 0.2260, 0 errors, 0 missing predictions, 0 empty retrieval rows, 0 retrieval/doc/HyRE cache misses, 0 state-filter misses, 0 fallback flags, 0 think tags, 0 near-cap rows; 2 valid same-model answer-format retries. The runner log has a post-OK wrapper EOF caveat after writing the detail log and experiment summary. |
| `groq-llama8b` `snap_hyre` | complete and signed, 4043/6853 = 59.0% | Strict full-row audits pass: Hit@5 0.2956, MRR@5 0.1888, 0 errors, 0 missing predictions, 0 empty retrieval rows, 0 retrieval/doc/HyRE cache misses, 0 state-filter misses, 0 fallback flags, 0 think tags, 0 missing exact final answers; 140 valid same-model answer-format retries with 3-token repairs, 0 retry-near-cap rows, and 3382 CE doc truncation flags treated as a reranker-input caveat. Paired vs `llm_only`: +3.62pp, p=2.63e-07; vs state-filtered `rag_simple`: -3.30pp, p=6.39e-07; vs state-filtered `rag_hyde`: -0.06pp, p=0.946. |
| `or-gemma4-26b` `rag_simple` | 3691/6853 deduped combined prefix+Cloudflare tail/chunks = 53.9% partial in `current_status.md`; 3688 rows are clean and 3 failed rows must be superseded before merge/signoff | Blocked by OpenRouter `403 Key limit exceeded (monthly limit)` at 2026-05-21 06:38 CDT. `NO_SILENT_FALLBACK=1` correctly failed closed, producing 3 errors, 3 missing predictions, 3 empty-retrieval rows, and 3 fallback-guard rows; state-filter/cache/doc-cache misses remain 0 on the 3688 clean retrieval rows. Current partial f_acc including blocked rows is 2445/3691 = 66.2%, gold hit@5 1340/3691 = 36.3%. |
| Gemma q500 canonical `snap_hyre` | complete, audited 500/500 | `scripts/analyze_detail_flags.py` and `scripts/audit_housing_statefilter_detail.py --expected-rows 500 --require-hyre-cache` pass on `logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl`: 315/500 = 63.0%, Hit@1/5/10 0.1640/0.3820/0.4560, MRR@5/10 0.2429/0.2528, 0 errors, 0 missing predictions, 0 state-filter/cache/doc/HyRE misses, 0 fallback/think tags, exact final answer lines, max output 1465 tokens. |
| Gemma q500 `snap_hyre_exemplar` | complete, audited 500/500 | Runner stdout shows `NO_SILENT_FALLBACK=1`, state filter on, strict retrieval/doc cache path, explicit Cloudflare route, and `scripts/local/run_answer_cell.sh` cache resolver patch allowed reuse of `caches/generation/probes/housing_q500_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage.jsonl`. `scripts/analyze_detail_flags.py` and `scripts/audit_housing_statefilter_detail.py --expected-rows 500 --require-hyre-cache` pass: 314/500 = 62.8%, Hit@1/5/10 0.1840/0.4120/0.4960, MRR@5/10 0.2647/0.2760, 0 state-filter/cache/doc/HyRE/final-line/fallback/think issues, max output 1425 tokens. Paired vs canonical q500: -0.2pp, b/c=36/37, p=1.0, CI [-3.6, 3.005]. |

## Required Gates Per Promoted Row

Every promoted HousingQA state-filtered row must have:

- Full N=6853 detail rows after merge, if a merge is needed.
- `NO_SILENT_FALLBACK=1` in runner stdout.
- `housing_state_filter=true` and non-empty per-row `retrieval_where`.
- Strict retrieval-cache replay with zero `retrieval_cache_hit` misses.
- Strict document-cache replay with zero `retrieval_doc_cache_hit` misses.
- HyDE/Snap-HyRE rows have zero `hyre_cache_hit` misses.
- Retrieved/evidence length 5 on every row.
- Exact final `Answer: Yes` or `Answer: No` line on every row.
- Zero errors, missing predictions, parse failures, truthy fallback flags, and
  think tags.
- `scripts/analyze_detail_flags.py` passes.
- `scripts/audit_housing_statefilter_detail.py` passes.
- Retrieval metrics recorded at least for Hit@5, MRR@5, and final accuracy.
- `current_status.md` refreshed after signoff.

## Efficiency Metrics Setup

`scripts/compile_efficiency_metrics.py` now compiles token/performance
efficiency snapshots directly from detail JSONL logs. It reports answer-pass
input/output tokens, actual answer-wrapper calls, logical method calls,
cached-generation calls, average/p50/p95 latency, correct answers per 1M
answer-pass tokens, tokens per correct answer, retrieval Hit@5, and
state/cache/final-line health.

Current generated artifacts:

- `docs/generated/housingqa_statefilter_efficiency_2026-05-21.md`
- `docs/generated/housingqa_statefilter_efficiency_2026-05-21.csv`

Use these as operational diagnostics and paper-prep material. Active/partial
rows in the report remain non-citable until the corresponding full-N row passes
the signoff gates above.

## Exemplar Gate

Prepared q500 caches:

- Canonical `snap_hyre` retrieval/doc cache:
  `caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
  and
  `caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10_doc_cache.jsonl`.
- Exemplar retrieval/doc cache:
  `caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl`
  and
  `caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10_doc_cache.jsonl`.

Retrieval-only gate already passed directionally:

- Canonical q500 `snap_hyre`: Hit@5 0.3820, MRR@5 0.2429.
- Exemplar q500 `snap_hyre_exemplar`: Hit@5 0.4120, MRR@5 0.2647.

The q500 exemplar answer gate completed cleanly. Canonical q500 `snap_hyre`
finished at 315/500 = 63.0%, and q500 `snap_hyre_exemplar` finished at
314/500 = 62.8%; both pass strict state-filter/cache-replay audits.
