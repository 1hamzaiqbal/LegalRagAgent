# Local Snap-HyRE Runner Scripts

These scripts mirror the comprehensive Snap-HyRE SLURM helpers for a local
machine with populated `datasets/`, `chroma_db/`, and API keys in `.env`.

On Windows, run these from WSL Ubuntu if possible. Native PowerShell can run the
underlying Python commands, but these helper scripts are Bash-first.

Use them in this order:

1. `run_api_smoke.sh` - provider and harness smoke, no result promotion.
2. `build_retrieval_caches.sh` - raw/golden retrieval caches plus qrel
   alignment and top-k matrix.
3. `build_generation_caches.sh` - HyDE/Snap-HyRE generation caches plus
   retrieval caches from those generated passages.
4. `run_answer_cell.sh` - one dataset/model answer ladder at a time.
5. `build_result_package.sh` - source-gated package status tables and optional
   plots after caches/logs exist.
6. `status_monitor.sh` - lightweight recurring refresh for `current_status.md`
   while answer cells run.

The scripts intentionally default to small or bounded runs where possible.
Set `QUESTIONS=full` only after smokes and cache audits are clean.

Example generation-cache pass:

```bash
PROVIDER=or-gemma4-26b MODEL_LABEL=gemma4-26b QUESTIONS=50 \
  scripts/local/build_generation_caches.sh
```

Cache filenames are scoped by `QUESTIONS`, `SEED`, and optional
`SAMPLE_START`/`SAMPLE_END`, for example
`legalbench_scalr_q50_seed42_gemma4-26b_snap_hyre_k10.jsonl`. Use the same
scope values when building generation caches, retrieval caches, and answer
cells. Override `CACHE_SCOPE` only when you intentionally want a custom name.

Common environment:

```bash
export CHROMA_DB_DIR="$PWD/chroma_db"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_CROSS_ENCODER=0
export LLM_MAX_COMPLETION_TOKENS=2048
```

Keep `DISABLE_CROSS_ENCODER=0` for any retrieval cache or answer row that may
be promoted. Set it to `1` only for a deliberately labeled dense-only speed
smoke.

`run_answer_cell.sh` defaults to `REQUIRE_RETRIEVAL_CACHES=1`, so cacheable
retrieval modes fail instead of silently falling back to live retrieval. Set it
to `0` only for exploratory runs.

To check that a cache will replay through the harness before spending answer
calls:

```bash
uv run python scripts/smoke_retrieval_cache_hydration.py \
  --cache caches/retrieval/full/legalbench_scalr_q50_seed42_raw_question_k10.jsonl \
  --dataset legalbench_scalr --questions 50 --label-prefix simple \
  --retrieval-k 5
```

Required `.env` keys for the planned providers:

```env
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
```

The exact Gemma 4 E4B axis is not an OpenRouter row in this repo. Historical
Gemma 4 E4B rows are vLLM provenance, not a launch requirement for the current
API-only comprehensive package. Use `groq-llama8b` as the small-model API row.
Do not use `or-gemma3n-e4b` as a replacement; that is `google/gemma-3n-e4b-it`.

For the remaining HousingQA Gemma 26B state-filtered rows, use
`scripts/local/check_housing_gemma_readiness.sh` first, then the one-command
continuation `scripts/local/run_housing_gemma_after_key_reset.sh` only after the
network preflight passes. The Gemma launchers fail closed on exact model
identity, OpenRouter budget, and a tiny OpenRouter chat-route smoke via
`scripts/check_expected_provider_model.py`,
`scripts/check_openrouter_key_status.py`, and
`scripts/check_openrouter_chat_route.py`; do not bypass those guards for
paper-facing rows.
The chat-route smoke disables OpenRouter provider fallbacks by default and
honors `OPENROUTER_PROVIDER_ONLY`, `OPENROUTER_PROVIDER_ORDER`, and
`OPENROUTER_PROVIDER_IGNORE` when an operator needs to pin or exclude a
same-model serving route explicitly.
If nobody will be watching the key reset window, run
`scripts/local/watch_housing_gemma_until_ready.sh` in tmux/screen. It is
non-launching by default; set `LAUNCH_ON_READY=1` only when you want it to run
the canonical continuation and final completion gate automatically after the
same preflight succeeds.
Use `scripts/local/housing_gemma_budget_watcher.sh status|start|stop` for
detached watcher management; it records and checks the watcher lock/PID and
removes stale watcher locks on start. The manager is non-launching by default;
set `LAUNCH_ON_READY=1` explicitly only when automatic launch after a successful
preflight is intended.

After any clean batch, rebuild the package status:

```bash
scripts/local/build_result_package.sh
```

To keep the operational dashboard current during long runs:

```bash
scripts/local/status_monitor.sh start
scripts/local/status_monitor.sh status
tail -f logs/monitors/current_status_monitor.log
```
