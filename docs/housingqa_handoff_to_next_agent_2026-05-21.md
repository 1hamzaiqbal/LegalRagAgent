# HousingQA Handoff To Next Agent - 2026-05-21

Latest update, 2026-05-22 13:32 CDT: the focused HousingQA core state-filter
matrix is now 8/9 signed. The completed/signed `or-gemma4-26b` rows are:

- `rag_simple`: 4531/6853 = 66.1%, Hit@5 0.3695, MRR@5 0.2330, signed at
  `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`.
- `rag_hyde`: 4456/6853 = 65.0%, Hit@5 0.3063, Recall@5 0.2042, MRR@5
  0.1964, signed at
  `logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl`.

The only remaining full-N core row is `or-gemma4-26b` `snap_hyre`. Its
generation, retrieval, and document caches are built and audited. The full-N
state-filter retrieval cache is
`caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
with Hit@5 0.3807, Recall@5 0.2505, and MRR@5 0.2452, which is above the
shared state-filter raw retrieval cache (Hit@5 0.3695, Recall@5 0.2413,
MRR@5 0.2330). The answer row is actively running in
`logs/eval_snap_hyre_or-gemma4-26b_20260522_124028_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-nfull-k5_detail.jsonl`.
`current_status.md` was refreshed at 13:32 CDT and showed 474/6853 rows,
67.1% partial accuracy, zero errors, zero cache misses, zero state-filter
misses, zero fallback/think artifacts, and one logged same-model
answer-format retry. The live stream has continued past that snapshot with no
visible provider, fallback, or truncation errors. Do not launch a duplicate
Gemma core queue while `/tmp/housing_gemma_core_queue.lock` is present.

Latest update, 2026-05-21 13:58 CDT: Gemma `rag_simple` is done and signed.
Use
`logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
and the corresponding row in `docs/signoff_log.md`. Metrics: 4531/6853 =
66.1%, Hit@5 0.3695, MRR@5 0.2330, zero strict health failures, 13
same-model answer-format retries. `current_status.md` now shows HousingQA
`or-gemma4-26b` `rag_simple` as 100% signed and the focused core audit is 7/9.
The only missing core rows are Gemma `rag_hyde` and Gemma `snap_hyre`.

Do not start a duplicate generated-method queue yet. A core-queue lock is
present at `/tmp/housing_gemma_core_queue.lock`, and the Gemma `rag_hyde`
generation cache is actively growing:
`caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl` went from
55 to 95 rows between the 13:57 and 13:58 CDT checks, then to 333 rows at
14:03 CDT, then to 1150 rows at 14:21 CDT. Because this sandbox
cannot reliably see host pids, use cache growth and lock metadata as the active
job signal.
`current_status.md` now shows fresh partial generation caches as active even
when the process is not visible inside the sandbox.

Latest update, 2026-05-21 13:15 CDT: the three 17:34 UTC middle recovery
shards completed cleanly. The tail is now active through three sample-suffixed
logs launched at 18:13 UTC: `s5796-e6148`, `s6148-e6500`, and `s6500-eend`.
`current_status.md` was refreshed at 13:14 CDT and reports
`or-gemma4-26b` Housing `rag_simple` at 85.5% active, 5860/6853 rows, with
zero structural health issues. Do not launch duplicate tail work while these
three logs are growing. After they finish, run:

```bash
python3 scripts/report_housing_gemma_rag_simple_gaps.py
OUT=logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_$(date -u +%Y%m%d_%H%M%S)_detail.jsonl \
  scripts/local/merge_audit_housing_gemma_rag_simple.sh
```

If the gap helper still reports holes after the tail shards stop, use
`RAG_SIMPLE_RANGES="start:end ..."` with
`scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh` only for
those remaining holes.

Latest update, 2026-05-21 12:43 CDT: three sample-suffixed Gemma `rag_simple`
shards from the 17:34 UTC attached recovery are actively growing:
`s3479-e3829`, `s3829-e4180`, and `s4180-e4530`. They cover the first collided
recovery region. `scripts/report_housing_gemma_rag_simple_gaps.py` was added
to report canonical coverage from the state-filter raw retrieval cache. At
12:43 CDT it reported `covered=5072`, `missing=1781`, and `unexpected=0`; the
remaining gaps were the shrinking middle ranges plus the untouched tail
`5849:6853` (`hqa_Utah_49` through `hqa_Wyoming_8829`). The current sandbox
OpenRouter preflight fails closed on DNS, so do not launch replacement Gemma
rows from here. Let the active shard logs finish, then run the gap helper and
only target the remaining missing ranges.

Sample-suffixed detail logs are now first-class inputs for the Housing Gemma
audit path. The patterns in `scripts/audit_housing_statefilter_goal.py`,
`scripts/local/merge_audit_housing_gemma_rag_simple.sh`,
`scripts/local/audit_housing_gemma_core_rows.sh`, and
`scripts/local/finalize_housing_gemma_signoff.sh` were broadened from
`nfull-k5_detail.jsonl` to `nfull-k5*_detail.jsonl`, so collision-safe chunks
are visible to merge/finalization gates.
The resume helper also supports an opt-in targeted continuation:
`RAG_SIMPLE_RANGES="5849:" scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`
or multiple `start:end` ranges copied from the gap helper. Use this only after
active shard logs have stopped or finished, otherwise it can overlap with
in-flight work.

Snapshot refreshed locally at 2026-05-21 10:13 CDT after
`current_status.md` was regenerated at 2026-05-21 10:13:42 CDT. Treat
`docs/signoff_log.md` as the paper citation gate and `current_status.md` as the
operational dashboard.

Latest live checkpoint: at 2026-05-21 12:33 CDT, `current_status.md` had been
refreshed to 2026-05-21 12:32:55 CDT and still showed HousingQA
`or-gemma4-26b` `rag_simple` active from partial detail logs. The first
three-way parallel recovery
used the same tag for all chunks and collided on the same detail-log path. The
unsuffixed 10:14 detail log now preserves the middle span
(`hqa_Ohio_6341` through `hqa_Tennessee_7216`, 1056 rows), but it must not be
treated as a clean three-chunk recovery artifact. `scripts/local/run_answer_cell.sh`
has been patched so future chunked runs include sample spans in the tag, and
`scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh` now
defaults `UV_CACHE_DIR=/tmp/uv-cache` for direct recovery launches.

Two repaired same-model Cloudflare reruns started for the missing first and
third spans:
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s3478-e4530_detail.jsonl`
stalled at 48/1052, and
`logs/eval_rag_simple_or-gemma4-26b_20260521_122107_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5-s5796-eend_detail.jsonl`
stalled at 53/1057. They are clean partials but not merge-ready. A foreground
relaunch at 17:31 UTC failed closed during the OpenRouter preflight smoke with
`Connection error`, before writing new rows. The next agent should rerun the
first and third spans only after API reachability returns, then merge/audit.

## Pasteable Handoff

You are taking over the HousingQA state-filtered finish. Start with
`CLAUDE.md`, `docs/README.md`, `current_status.md`,
`docs/signoff_log.md`, `docs/housingqa_statefilter_goal_checklist_2026-05-21.md`,
and `docs/barexam_housing_core_focus_2026-05-20.md`.

Current HousingQA core status is 6/9 signed for the required state-filtered
core matrix, with the `or-gemma4-26b` `rag_simple` recovery incomplete. The
completed/signed rows are `rag_simple`, `rag_hyde`, and `snap_hyre` for both
`groq-llama8b` and `groq-llama70b`. The incomplete rows are all
`or-gemma4-26b`: `rag_simple` has one preserved middle recovery span but still
needs clean first/third recovery spans, while full-N `rag_hyde` and full-N
`snap_hyre` still need to be built/run after `rag_simple` merges cleanly.
For citation, use the `HousingQA state-filtered` block in `docs/signoff_log.md`
for these six completed rows. Older unfiltered HousingQA signoff rows also
exist in that file and should not be used for the current state-filtered core
matrix.

The q500 Gemma `snap_hyre_exemplar` diagnostic is complete and clean but is not
a replacement for the full-N core rows. It is retrieval-positive and answer
parity: canonical q500 `snap_hyre` is 315/500 = 63.0%, Hit@5 0.3820, MRR@5
0.2429; exemplar q500 is 314/500 = 62.8%, Hit@5 0.4120, MRR@5 0.2647; paired
answer delta is -0.2pp with p=1.0. Scale full-N exemplar only after the three
Gemma core rows are complete and signed.

Live `rag_simple` recovery artifacts:

- `logs/eval_rag_simple_or-gemma4-26b_20260521_101450_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_3478_4530_20260521_151441.out`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_4634_5690_20260521_151441.out`
- `logs/run_housing_statefilter_gemma_rag_simple_resume_5796_end_20260521_151441.out`

At 2026-05-21 10:28 CDT, the live detail log had 388 parsed rows and the
midstream structural audit showed zero JSON errors, row errors, missing final
answer lines, retrieval-cache misses, doc-cache misses, missing state filters,
fallback markers, or think tags.

The exact Gemma route is `or-gemma4-26b` resolving to
`google/gemma-4-26b-a4b-it` through OpenRouter, currently pinned with
`OPENROUTER_PROVIDER_ONLY=Cloudflare`. Do not silently substitute Gemma 3,
Gemini, direct Google, or a different OpenRouter model. Same-model provider
route changes are allowed only if they are explicit in the logs/signoff caveat
and pass the no-silent-fallback/model guards.
As of 2026-05-21 09:15 CDT, a live environment scan found no alternate
approved Google/Gemini/OpenRouter key variable to use for the remaining
canonical row, and `llm_config.py` still maps the direct Google `gemma`
provider to Gemma 3 rather than `google/gemma-4-26b-a4b-it`.
The configured `or-gemma4-26b-free` alias is not currently accepted as a silent
replacement because it resolves to `google/gemma-4-26b-a4b-it:free`, which fails
the canonical exact-model guard for `google/gemma-4-26b-a4b-it`.
The guard script now has an explicit `--allow-openrouter-free-suffix` switch,
also available as `OPENROUTER_ALLOW_FREE_SUFFIX=1`, for validating that suffix
as a deliberate noncanonical route, but default launcher behavior remains strict
and should be treated as the required canonical path.
A non-launching preflight with `PROVIDER=or-gemma4-26b-free`,
`MODEL_LABEL=or-gemma4-26b`, and `OPENROUTER_ALLOW_FREE_SUFFIX=1` passed local
integrity checks and the opt-in model guard at 2026-05-21 08:37 CDT, then
failed closed at the same OpenRouter budget check before any row launch.
Direct tiny completion smokes at 2026-05-21 08:42 CDT show this is not a clean
escape hatch right now: unconstrained `google/gemma-4-26b-a4b-it:free` returned
an upstream Google AI Studio 429, and the Cloudflare-pinned request returned
404 because only `google-ai-studio` is available for the `:free` model.

Before launching anything, run the read-only gate:

```bash
scripts/local/check_housing_gemma_readiness.sh
```

When OpenRouter API/budget reachability is expected to be restored, run the
canonical non-launching network preflight:

```bash
CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh
```

Only after that passes, launch the canonical continuation:

```bash
scripts/local/run_housing_gemma_after_key_reset.sh
```

Optional polling helper if a reset window may happen while no agent is active:

```bash
# dry-run watcher; exits when the route/budget preflight passes
scripts/local/watch_housing_gemma_until_ready.sh

# launch-on-ready watcher; use from tmux/screen, not a short Codex exec
LAUNCH_ON_READY=1 scripts/local/watch_housing_gemma_until_ready.sh
```

The foreground watcher uses a watcher lock while it runs, and the managed
watcher uses `logs/monitors/locks/housing_gemma_budget_watch.lock`. Before each
poll, the patched watcher checks both legacy `/tmp` and repo-local
`logs/monitors/locks/` Housing launch-lock paths, runs the same preflight
wrapper in `PREFLIGHT_ONLY=1` mode, and logs to `logs/monitors/`. It does not
launch rows unless `LAUNCH_ON_READY=1` is set.

`current_status.md` now reports watcher lock metadata including whether the
recorded watcher pid is live. Use
`scripts/local/housing_gemma_budget_watcher.sh status|start|stop` for detached
watcher management; it is non-launching by default unless `LAUNCH_ON_READY=1`
is set explicitly. Stale lock/process mismatches from earlier manual watcher
starts showed `/tmp` locks were not durable enough for this Codex-started
detached process, so managed watcher locks now live under
`logs/monitors/locks/` while the dashboard still checks legacy `/tmp` launch
locks. A managed launch-on-ready watcher started at 2026-05-21 09:22 CDT with
pid `3646233`, lock `logs/monitors/locks/housing_gemma_budget_watch.lock`, and
log `logs/monitors/housing_gemma_budget_watch_20260521_142231.log`. Its first
preflight passed the exact model guard and failed closed at OpenRouter
`limit_remaining=0`; by 09:23 CDT,
`scripts/local/housing_gemma_budget_watcher.sh status` reported the lock stale,
and `scripts/local/housing_gemma_budget_watcher.sh stop` removed it. No Housing
launch locks, answer rows, or
`docs/generated/housing_gemma_signoff_candidates_20260521_142231.md` were
created, and no watcher lock is currently present.

The continuation wrapper and direct Gemma launch helpers now also run
`scripts/check_openrouter_chat_route.py` after their key-budget guards and
before any resume/cache/answer work. This performs a tiny exact-route chat
completion with provider fallbacks disabled. It honors
`OPENROUTER_PROVIDER_ONLY`, `OPENROUTER_PROVIDER_ORDER`, and
`OPENROUTER_PROVIDER_IGNORE`, so a reset window fails before row writes if
OpenRouter can report budget but cannot actually serve the requested Gemma
route under the intended provider controls.

If the canonical route remains blocked and an operator deliberately approves the
noncanonical OpenRouter `:free` suffix as a caveated route, first verify it with
this non-launching preflight:

```bash
PREFLIGHT_ONLY=1 PROVIDER=or-gemma4-26b-free MODEL_LABEL=or-gemma4-26b \
OPENROUTER_ALLOW_FREE_SUFFIX=1 scripts/local/run_housing_gemma_after_key_reset.sh
```

Do not run the actual `:free` continuation unless the paper/result label and
signoff caveat are changed explicitly; it is not the current canonical row. The
latest direct smoke also shows the `:free` route is rate-limited upstream and
not available through Cloudflare, so treat it as unavailable until a new smoke
passes.

That wrapper should resume/supersede the three bad Gemma `rag_simple` spans,
merge/audit/sign `rag_simple`, build Gemma `rag_hyde` and `snap_hyre`
generation/retrieval/doc caches, run their answer rows, audit/sign them, refresh
`current_status.md`, and then run full-N `snap_hyre_exemplar` unless
`RUN_FULL_EXEMPLAR_AFTER_CORE=0` is set.
This sequence was rechecked locally at 2026-05-21 09:54 CDT from the scripts:
`run_housing_gemma_after_key_reset.sh` calls the `rag_simple`
resume/merge-audit helpers, `run_housing_gemma_core_queue.sh`,
`audit_housing_gemma_core_rows.sh`, `finalize_housing_gemma_signoff.sh`, and
then `run_housing_gemma_exemplar_full_after_core.sh` by default. The core queue
calls both `build_and_run_mode rag_hyde hyde_cache hyde` and
`build_and_run_mode snap_hyre hyre_cache snap_hyre`, so both generated Gemma
core rows are covered by the one canonical continuation command after the
preflight clears.

After the continuation, the strict completion gate is:

```bash
scripts/local/verify_housing_statefilter_goal_complete.sh
```

Do not mark the goal complete until that passes. In the current blocked state it
is expected to fail because no full clean Gemma `rag_simple` detail log exists.

## Latest Verified Operational State

- `current_status.md`: 67/78 signed, 67/78 full-row complete, 0/78 active,
  2/78 partial stale, 9/78 not started.
- HousingQA core state-filter gate:
  `python3 scripts/audit_housing_statefilter_goal.py --allow-incomplete`
  reports `complete=6/9`.
- No Housing launch lock directories were present in the 2026-05-21 09:47 CDT
  local lock check:
  `/tmp/housing_gemma_after_key_reset.lock`,
  `/tmp/housing_gemma_rag_simple_resume.lock`,
  `/tmp/housing_gemma_core_queue.lock`,
  `/tmp/housing_gemma_exemplar_full.lock`, and
  `/tmp/housing_gemma_budget_watch.lock`, plus the corresponding
  `logs/monitors/locks/` lock directories.
- No watcher lock is currently present. The prior launch-on-ready watcher
  (`logs/monitors/housing_gemma_budget_watch_20260521_141011.log`, pid
  `3636546`) reached OpenRouter and failed closed with
  `limit_remaining=0 < required 0.01`, then its pid died; the stale
  `/tmp/housing_gemma_budget_watch.lock` was removed at 2026-05-21 09:15 CDT.
  A second launch-on-ready watcher
  (`logs/monitors/housing_gemma_budget_watch_20260521_141921.log`, pid
  `3643843`) did the same at 09:19 CDT; its stale lock was removed at
  09:20 CDT after `ps` showed the pid was gone. A managed repo-local watcher
  (`logs/monitors/housing_gemma_budget_watch_20260521_142231.log`, pid
  `3646233`) did the same at 09:22 CDT; `housing_gemma_budget_watcher.sh stop`
  removed its stale `logs/monitors/locks/` lock at 09:23 CDT. The dashboard now
  reports Housing watcher lock metadata when a watcher lock is present.
- No active Housing/eval runner processes were visible through `ps` in this
  sandbox at the last local check. `tmux ls` was not readable from this sandbox
  (`Operation not permitted`), so use lock dirs plus `ps` here.
- Offline readiness passed at 2026-05-21 09:38 CDT: the focused 9-cell gate is
  still `complete=6/9`, resume offsets verify as
  `3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and
  `5796 -> hqa_Texas_4530`; provider/model check accepts
  `or-gemma4-26b -> google/gemma-4-26b-a4b-it`; the Gemma `rag_simple`
  partial merge gate still fails closed as expected with 3691/6853 rows and
  the three failed rows.
- OpenRouter budget was rechecked directly at 2026-05-21 09:26 CDT with
  `python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01
  --retries 3 --retry-delay 2`. It reached OpenRouter and failed closed with
  `limit_remaining=0 < required 0.01` (`usage=23.912568061`, `limit=20`).
  No answer row was launched and no signoff-candidate file was created.
- OpenRouter budget was rechecked again directly at 2026-05-21 09:39 CDT with
  the same command. In this sandbox it failed closed on OpenRouter DNS
  resolution after bounded retries, before budget lookup. No Gemma row was
  launched.
- The full non-launching readiness wrapper was rerun at 2026-05-21 09:40 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It found no
  Housing launch locks, confirmed the focused gate remains `complete=6/9`,
  verified the three Gemma `rag_simple` resume offsets and exact model label,
  confirmed the current `rag_simple` merge gate still fails closed at 3691/6853,
  and then stopped at the OpenRouter key-budget guard with `limit_remaining=0`.
  No row was launched.
- OpenRouter budget was checked directly again at 2026-05-21 09:43 CDT with
  `python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01
  --retries 3 --retry-delay 2`. It reached OpenRouter and failed closed with
  `limit_remaining=0 < required 0.01` (`usage=23.912568061`, `limit=20`). No
  Gemma row was launched.
- The full non-launching readiness wrapper was rerun at 2026-05-21 10:07 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It found no
  Housing launch locks, confirmed the focused gate remains `complete=6/9`,
  verified the three Gemma `rag_simple` resume offsets and exact model label,
  confirmed the current `rag_simple` merge gate still fails closed at 3691/6853,
  and then failed closed on OpenRouter DNS after bounded retries before budget
  lookup or row launch. It created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_150713.md` file.
- The full non-launching readiness wrapper was rerun at 2026-05-21 10:09 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It found no
  Housing launch locks, confirmed the focused gate remains `complete=6/9`,
  verified the three Gemma `rag_simple` resume offsets and exact model label,
  confirmed the current `rag_simple` merge gate still fails closed at 3691/6853,
  and then failed closed at the OpenRouter key-budget guard with
  `limit_remaining=0` (`usage=23.912568061`, `limit=20`). No row was launched,
  and no `docs/generated/housing_gemma_signoff_candidates_20260521_150919.md`
  file was created.
- The full non-launching readiness wrapper was rerun at 2026-05-21 10:13 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It passed
  the local lock, focused-gate, resume-offset, and exact-model checks, then
  failed closed at the OpenRouter key-budget guard with `limit_remaining=0`
  (`usage=23.912568061`, `limit=20`). No row was launched, and no
  `docs/generated/housing_gemma_signoff_candidates_20260521_151304.md` file was
  created.
- `current_status.md` was refreshed at 2026-05-21 10:13 CDT and still reports
  67/78 signed, 0/78 active, and the same three missing `or-gemma4-26b`
  Housing core rows.
- The strict no-finalizer/no-exemplar completion gate was rerun at 2026-05-21
  09:44 CDT:
  `RUN_FINALIZER=0 REQUIRE_FULL_EXEMPLAR=0 scripts/local/verify_housing_statefilter_goal_complete.sh`.
  It found no Housing launch locks, passed syntax/Python helper checks,
  refreshed `current_status.md`, and failed at the expected focused audit with
  `complete=6/9`; the missing cells are still the three `or-gemma4-26b` core
  rows.
- The canonical continuation wrapper was probed in preflight-only mode at
  2026-05-21 09:45 CDT:
  `PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh`. It
  passed local integrity checks and the exact model guard
  (`or-gemma4-26b -> google/gemma-4-26b-a4b-it`), then failed closed at the
  OpenRouter key-budget guard with `limit_remaining=0`. It skipped the full
  launch lock and created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_144545.md` file.
- The full non-launching readiness wrapper was rerun at 2026-05-21 09:46 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It found no
  launch/watch locks, confirmed the focused gate remains `complete=6/9`,
  verified the three Gemma `rag_simple` resume offsets and exact model label,
  confirmed the `rag_simple` merge gate still fails closed at 3691/6853, and
  then failed closed on OpenRouter DNS before budget lookup or row launch. It
  created no `docs/generated/housing_gemma_signoff_candidates_20260521_144610.md`
  file.
- The focused offline audit was rerun at 2026-05-21 09:47 CDT:
  `python3 scripts/audit_housing_statefilter_goal.py --allow-incomplete`.
  It reconfirmed the six Groq rows are signed, full-N, and clean, and the only
  missing core cells remain Gemma `rag_simple` at 3691/6853, Gemma `rag_hyde`
  at 0/6853, and Gemma `snap_hyre` at 0/6853.
- OpenRouter budget was checked directly again at 2026-05-21 09:48 CDT with
  `python3 scripts/check_openrouter_key_status.py --min-limit-remaining 0.01
  --retries 2 --retry-delay 2`. It reached OpenRouter and failed closed with
  `limit_remaining=0 < required 0.01` (`usage=23.912568061`, `limit=20`). No
  Gemma row was launched.
- Offline readiness was rerun at 2026-05-21 09:49 CDT with
  `scripts/local/check_housing_gemma_readiness.sh`. It found no Housing launch
  locks, reconfirmed `complete=6/9`, verified the three Gemma `rag_simple`
  resume offsets and exact model label without API calls, listed the expected
  missing full-N Gemma HyDE/Snap caches, and confirmed the current Gemma
  `rag_simple` merge gate still fails closed at 3691/6853 with the three failed
  rows. After budget reset, go straight to the network gate and continuation
  wrapper.
- Monitor/process status was checked at 2026-05-21 09:50 CDT:
  `scripts/local/status_monitor.sh status` reports the recurring
  `current_status.md` monitor is running (`pid=3627587`, interval 300s), and a
  lock/process scan found no Housing launch locks and no active Housing
  eval/cache/answer jobs besides that monitor.
- The full non-launching readiness wrapper was rerun again at 2026-05-21
  09:50 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It passed
  offline readiness and exact-model checks, confirmed the expected partial
  merge failure at 3691/6853, then failed closed on OpenRouter DNS before
  budget lookup or row launch. It created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_145054.md` file.
- Local continuation syntax was checked at 2026-05-21 09:52 CDT: `bash -n`
  passes over the Housing Gemma launch/finalize/verify scripts, and
  `python3 -m py_compile` passes over the provider guards, Housing audit
  helpers, status updater, and signoff summarizer. The remaining blocker is not
  local launcher syntax; it is the exact Gemma OpenRouter budget/route.
- The full non-launching readiness wrapper was rerun again at 2026-05-21
  09:53 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It passed
  offline readiness and exact-model checks, confirmed the expected partial
  merge failure at 3691/6853, then failed closed on OpenRouter DNS before
  budget lookup or row launch. It created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_145351.md` file.
- The full non-launching readiness wrapper was rerun again at 2026-05-21
  09:57 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It passed
  offline readiness and exact-model checks, confirmed the expected partial
  merge failure at 3691/6853, then failed closed on OpenRouter DNS before
  budget lookup or row launch. It created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_145701.md` file.
- The full non-launching readiness wrapper was rerun again at 2026-05-21
  10:01 CDT:
  `CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh`. It passed
  offline readiness and exact-model checks, confirmed the expected partial
  merge failure at 3691/6853, then failed closed on OpenRouter DNS before
  budget lookup or row launch. It created no
  `docs/generated/housing_gemma_signoff_candidates_20260521_150104.md` file.
- A masked environment check found no alternate OpenRouter key in the process
  environment, and `.env` contains only `OPENROUTER_API_KEY`. The current
  blocker is account budget on the configured key, not a missing key variable.
- The readiness wrapper was retried with `CHECK_NETWORK=1` at 2026-05-21
  09:38 CDT. It passed offline readiness, resume-offset checks, local integrity
  checks, and exact-model validation, then failed closed on OpenRouter DNS after
  the bounded retries before budget lookup or row launch. The direct budget
  check above shows the alternate safe-fail mode when OpenRouter is reachable:
  `limit_remaining=0`. A direct `getent hosts openrouter.ai api.openrouter.ai`
  check also returned no host records in this sandbox. No signoff-candidate file
  was created for
  `docs/generated/housing_gemma_signoff_candidates_20260521_143811.md`.
- `MAX_ATTEMPTS=1 INTERVAL_SECONDS=30 scripts/local/watch_housing_gemma_until_ready.sh`
  was dry-run at 2026-05-21 09:08 CDT. It stayed non-launching
  (`LAUNCH_ON_READY=0`), passed local exact-model checks, failed closed on the
  same OpenRouter DNS error before budget lookup, exited 20 after one attempt,
  wrote `logs/monitors/housing_gemma_budget_watch_20260521_140823.log`, and
  removed `/tmp/housing_gemma_budget_watch.lock`.
- The six signed Groq Housing core rows were re-audited at 2026-05-21 08:22
  CDT with `scripts/audit_housing_statefilter_detail.py` and all still pass
  current strict checks: 6853/6853 rows, zero wrong provider/mode/dataset rows,
  zero missing predictions/errors, zero state-filter misses, zero retrieval/doc
  cache misses, zero HyRE-cache misses for generated methods, zero bad evidence
  length rows, zero missing exact final answers, zero fallback markers, and zero
  think tags.
- The Gemma q500 canonical/exemplar diagnostic was re-audited at 2026-05-21
  08:25 CDT. Both answer logs still pass strict state-filter/cache/final-line
  checks. Retrieval-cache audits over the matched q500 slice show canonical
  `snap_hyre` Hit@1/5/10 = 0.1640/0.3820/0.4560 and MRR@5/10 =
  0.2429/0.2528; `snap_hyre_exemplar` Hit@1/5/10 =
  0.1840/0.4120/0.4960 and MRR@5/10 = 0.2647/0.2760.
- Offline launch-path validation was rerun at 2026-05-21 08:30 CDT: shell
  syntax checks pass for the Housing Gemma launcher/audit/finalizer scripts,
  Python compile checks pass for the Housing audit/status/provider helpers, and
  `VERIFY_ONLY=1 scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`
  verifies the three failed-row resume offsets plus the exact Gemma model label
  before API preflight.
- The strict completion gate was rerun at 2026-05-21 09:22 CDT:
  `scripts/local/verify_housing_statefilter_goal_complete.sh` found no launch
  locks, passed syntax/Python helper checks, then failed closed in the Gemma
  finalizer with `no full clean state-filter detail log found for
  mode=rag_simple`. This is the expected negative-control result while the
  Gemma `rag_simple` row is still partial.
- The current Gemma `rag_simple` merge gate fails closed as expected:
  merged rows=3691, errors=3, missing predictions=3, empty retrieval=3,
  missing exact final=3, fallback=3, expected rows=6853.
- The Gemma `rag_simple` source-log selection was rechecked at 2026-05-21
  10:01 CDT using the same `housing_state_filter=true` rule as
  `scripts/local/merge_audit_housing_gemma_rag_simple.sh`. Exactly five source
  logs qualify: 88 clean rows, 3282 clean rows, and three shard logs with one
  failed-closed row each. Only `hqa_Nebraska_2941`, `hqa_Ohio_6341`, and
  `hqa_Texas_4530` need same-model superseding before merge/signoff.
- The resume-offset guard was rerun at 2026-05-21 10:03 CDT:
  `VERIFY_ONLY=1 scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh`.
  It verified `3478 -> hqa_Nebraska_2941`, `4634 -> hqa_Ohio_6341`, and
  `5796 -> hqa_Texas_4530`, accepted the exact Gemma model mapping, and exited
  before OpenRouter preflight, launch locks, or row writes.
- Full Gemma generated-method caches are still missing:
  `caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`,
  `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`,
  `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10_doc_cache.jsonl`,
  and the corresponding `snap_hyre` files. A 2026-05-21 08:23 filesystem scan
  found no full-N Gemma `rag_hyde`, `snap_hyre`, or `snap_hyre_exemplar` cache
  artifacts and no new full-N Gemma generated-method detail logs to audit.
  This was rechecked at 2026-05-21 09:59 CDT: the canonical full-N Gemma
  `rag_hyde`, `snap_hyre`, and `snap_hyre_exemplar` generation/retrieval/doc
  cache paths are still absent, and no matching full-N generated-method answer
  detail logs exist.

## Signed HousingQA Core Rows To Cite

Use `docs/signoff_log.md` lines for exact caveats and paired tests.

| Provider | Mode | Final accuracy | Retrieval exposure |
|---|---|---:|---:|
| `groq-llama8b` | `rag_simple` | 4269/6853 = 62.3% | Hit@5 0.3695 / MRR@5 0.2330 |
| `groq-llama8b` | `rag_hyde` | 4047/6853 = 59.1% | Hit@5 0.2695 / MRR@5 0.1688 |
| `groq-llama8b` | `snap_hyre` | 4043/6853 = 59.0% | Hit@5 0.2956 / MRR@5 0.1888 |
| `groq-llama70b` | `rag_simple` | 4258/6853 = 62.1% | Hit@5 0.3695 / MRR@5 0.2330 |
| `groq-llama70b` | `rag_hyde` | 4263/6853 = 62.2% | Hit@5 0.3495 / MRR@5 0.2260 |
| `groq-llama70b` | `snap_hyre` | 4087/6853 = 59.6% | Hit@5 0.2311 / MRR@5 0.1427 |

Interpretation for paper work: HousingQA state filtering is required for fair
main-matrix retrieval. In HousingQA specifically, raw state-filtered RAG is
currently stronger than Snap-HyRE on the completed Groq rows; the positive
Snap-HyRE story here is not "wins HousingQA core." The useful Housing result is
that jurisdiction filtering gives valid retrieval conditions, and the q500
exemplar diagnostic shows a retrieval-exposure lift with answer parity for
Gemma 26B. Stronger positive Snap-HyRE claims should be drawn from the signed
BarExamQA and Legal-Link-EU rows, not overstated from HousingQA.

## Files Another Agent Should Inspect

- `current_status.md` - latest completion and metrics dashboard.
- `docs/signoff_log.md` - only source to cite for paper-facing result claims.
- `docs/housingqa_statefilter_goal_checklist_2026-05-21.md` - exact Housing
  objective-to-artifact checklist and latest blocked-state explanation.
- `docs/barexam_housing_core_focus_2026-05-20.md` - focused operating note for
  the BarExamQA/HousingQA subset.
- `scripts/local/check_housing_gemma_readiness.sh` - read-only local gate.
- `scripts/local/run_housing_gemma_after_key_reset.sh` - one-command Gemma
  continuation after OpenRouter route/budget is available.
- `scripts/local/verify_housing_statefilter_goal_complete.sh` - strict final
  goal gate.
- `scripts/local/audit_housing_gemma_core_rows.sh` and
  `scripts/local/audit_housing_gemma_exemplar_full.sh` - post-run audit gates.
- `scripts/audit_housing_statefilter_goal.py` and
  `scripts/audit_housing_statefilter_detail.py` - focused machine checks.
- `scripts/summarize_housing_statefilter_signoff.py` - generates signoff-ready
  rows only from accepted full clean logs.

## Coordination Notes

- Avoid editing paper files in parallel with the paper-writing agent unless you
  have explicit ownership. Use this handoff plus `current_status.md` and
  `docs/signoff_log.md` as inputs for the paper agent instead.
- Avoid signing partial Gemma rows. The current Gemma `rag_simple` partial has
  three failed-closed rows and must be superseded by same-model reruns before
  merge/signoff.
- Keep `NO_SILENT_FALLBACK=1`, strict cache replay, state-filter verification,
  exact final `Answer: Yes/No` lines, no truthy fallback artifacts, and no
  think-token artifacts as hard gates.
- If OpenRouter network/API is unavailable from the sandbox, do not launch broad
  retries. The intended behavior is fail-closed before row work.
