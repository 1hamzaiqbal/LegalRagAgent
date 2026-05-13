# Snap-HyRE Experiment Runbook - 2026-05-12

## Goal

Produce one clean, fixed-method Snap-HyRE comparison across four legal
benchmarks and three model families. The main claim is retrieval-first:
Snap-HyRE should improve gold-evidence exposure versus raw-question retrieval.
Downstream accuracy lift is desirable, but negative or flat accuracy is still
interpretable if retrieval improves and the answer-conversion failure is shown.

## Canonical Benchmarks

| Dataset | Primary reason to keep | Main reported retrieval metric |
|---|---|---|
| BarExamQA | Legal exam MC task with full-corpus prior results. | Hit@1/5/10 and MRR over gold passage ids. |
| CaseHOLD | Holding-option task that separates retrieval from option mapping. | Hit@1/5/10 and MRR over gold holding ids. |
| LegalBench-SCALR | Legal holding-selection task with populated local/HPC Chroma. | Hit@1/5/10 and MRR where gold ids are available. |
| HousingQA | Statutory yes/no entailment benchmark from the same Zheng line. | Any-gold and all-gold statute retrieval when multiple statutes are labeled. |

HousingQA stays in the first pass because it is legal and related-work aligned.
If yes/no entailment keeps dominating and obscures the retrieval story, replace
it only after a documented smoke/audit says the fixed method table is not
interpretable. State filtering can remain an analysis/control row, not a
headline requirement.

## Canonical Models

| Label | Preferred provider | Fallback | Role |
|---|---|---|---|
| Gemma 4 E4B | HPC vLLM | OpenRouter | small Gemma axis |
| Gemma 4 26B | HPC vLLM | OpenRouter | main Gemma axis |
| Llama 3.3 70B Versatile | Groq | OpenRouter | cross-family large model |

Run provider smokes before queueing full jobs. Do not mix provider rows in the
same table cell unless the provider is explicitly labeled.

## Method Ladder

Use this inherited table order for the main report:

| Row | Harness mode | Retrieval query | Calls | Why it is included |
|---:|---|---|---:|---|
| 0 | `llm_only` | none | 1 | Parametric floor. |
| 1 | `rag_simple` | raw question | 1 | Main retrieval baseline. |
| 2 | `rag_hyde` | question-only hypothetical passage | 2 | Tests HyDE without snap conditioning. |
| 3 | `snap_hyre` | snap-conditioned HyRE passage | 2 | Primary method. Legacy alias: `rag_snap_hyde_2call`. |
| 4 | `golden_passage` | oracle passage injected | 1 | Answer-synthesis diagnostic, not deployable. |
| 5 | `golden_plus_neighbors` | oracle passage plus nearest neighbors | 1 | Tests whether single-gold context is under-specified. |
| 6 | `rag_rewrite` | search-style query rewrite | 2 | Strong non-HyRE query formulation control; keep in the main ablation table for now. |

All retrieval rows in a comparison must use the same `--retrieval-k`.

## Top-k Selection

Run retrieval-first caches at `k=10`, then analyze `k in {1,3,5,10}` without
repeating embedding search. Choose a universal `k` only if macro-average
retrieval exposure is near-best and downstream accuracy does not degrade. If no
single `k` is credible, report the curve and use `k=5` as the predeclared main
table default.

Apply the cached top-k check to deterministic or replayable retrieval methods:
`rag_simple`, `rag_hyde` after query-generation replay is available,
`snap_hyre`, and `golden_plus_neighbors`. `llm_only` and `golden_passage` are
k-invariant controls. Keep `rag_rewrite` in the main ablation table, but treat
it as an uncached second-stage control because rewritten queries are
model-specific.

## Cache Workflow

1. Build or reuse generation caches for generated-query methods:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/build_generation_cache.py \
  --mode snap_hyre \
  --provider <provider> \
  --dataset <dataset> \
  --questions full \
  --out caches/hyre/full/<dataset>_<model>_snap_hyre.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/build_generation_cache.py \
  --mode rag_hyde \
  --provider <provider> \
  --dataset <dataset> \
  --questions full \
  --out caches/hyre/full/<dataset>_<model>_rag_hyde.jsonl
```

`scripts/build_hyre_cache.py` remains available for extracting replay caches
from older detail logs, but the generation-cache builder is the preferred path
for top-k selection before answer sweeps.

2. Build deterministic retrieval caches:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset <dataset> \
  --questions <N-or-full> \
  --query-type raw_question \
  --max-k 10 \
  --out caches/retrieval/<dataset>_raw_question_k10.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset <dataset> \
  --questions <N-or-full> \
  --query-type hyde_cache \
  --hyre-cache-path caches/hyre/full/<dataset>_<model>_rag_hyde.jsonl \
  --max-k 10 \
  --out caches/retrieval/<dataset>_<model>_rag_hyde_k10.jsonl

UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset <dataset> \
  --questions <N-or-full> \
  --query-type hyre_cache \
  --hyre-cache-path caches/hyre/full/<dataset>_<model>_snap_hyre.jsonl \
  --max-k 10 \
  --out caches/retrieval/<dataset>_<model>_snap_hyre_k10.jsonl
```

3. Compile the top-k matrix after per-cache audits:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/compile_retrieval_cache_matrix.py \
  --cache 'caches/retrieval/full/*.jsonl' \
  --ks 1,3,5,10 \
  --min-k 10 \
  --out-md docs/generated/retrieval_cache_matrix.md \
  --out-csv docs/generated/retrieval_cache_matrix.csv
```

The HPC helper for raw-question/golden-neighbor caches is:

```bash
sbatch scripts/hpc/slurm_snap_hyre_retrieval_cache.sh
```

For Snap-HyRE caches, first build the HyRE generation cache, then launch with
`QUERY_TYPES='hyde_cache hyre_cache' HYRE_MODELS='<model-labels>'`.

The HPC helper for generation caches is:

```bash
PROVIDER=<provider> MODEL_LABEL=<model-label> \
sbatch scripts/hpc/slurm_snap_hyre_generation_cache.sh
```

For local vLLM Gemma, add `BACKEND=vllm MODEL=<hf-model-id> PORT=<port>`.

4. Audit before answer generation:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/audit_retrieval_cache.py \
  --cache caches/retrieval/<dataset>_<model>_snap_hyre_k10.jsonl \
  --dataset <dataset> \
  --query-type hyre_cache \
  --min-k 10
```

5. Replay answer generation at chosen `k`:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python eval/eval_harness.py \
  --mode snap_hyre \
  --dataset <dataset> \
  --questions <N-or-full> \
  --provider <provider> \
  --retrieval-k <k> \
  --hyre-cache-path caches/hyre/<dataset>_<model>_snap_hyre.jsonl \
  --retrieval-cache-path caches/retrieval/<dataset>_<model>_snap_hyre_k10.jsonl \
  --tag snap-hyre-comp-v1-<dataset>-<model>-k<k>
```

## Validation Gate

Do not promote a result row unless all of these pass:

- Detail JSONL copied locally.
- `scripts/analyze_detail_flags.py` reports no Tracebacks, auth/rate-limit
  failures, empty-retrieval spikes, runaway outputs, or missing-prediction
  spikes. Single-row caveats are allowed only if explicitly documented.
- Retrieval rows are scored with `scripts/score_retrieval_qrels.py` or
  `scripts/audit_retrieval_cache.py`.
- `docs/signoff_log.md` names the detail log, job/provenance, metric values,
  caveats, and clean/caveated/rejected status.

## Immediate Launch Order

1. Local/HPC corpus check: confirm Chroma collections exist for BarExamQA,
   CaseHOLD, LegalBench-SCALR, and HousingQA on the machine we will use.
2. Provider smoke: confirm Gemma 4 E4B, Gemma 4 26B, and Llama 3.3 70B
   Versatile return parseable answers on one question each.
3. Method smoke: `--questions 5` for each dataset on `rag_simple`,
   `rag_hyde`, `snap_hyre`, `golden_plus_neighbors`, and `rag_rewrite`.
4. Retrieval/top-k cache build: full-dataset k=10 retrieval caches for all
   cacheable retrieval methods. Use these to choose the universal k before
   downstream answer sweeps. Keep `rag_rewrite` out of this first cached
   top-k selection pass.
5. N=50 answer smoke across the three model labels for the fixed ladder at the
   provisional k.
6. N=200 confirmation if the N=50 results show parse/provider stability.
7. Full-corpus deliberate runs for every benchmark/model/method cell, at most
   two or three active jobs at once.

Full corpus means every available row for each benchmark. If provider budget or
rate limits block a full row, mark it as blocked or provisional; do not silently
replace it with a parity cap.

## Smoke Status

Completed locally on 2026-05-12:

- Python compile smoke passed for `eval/eval_config.py`, `eval/eval_harness.py`,
  `rag_utils.py`, `scripts/build_retrieval_cache.py`, and
  `scripts/audit_retrieval_cache.py`.
- CLI smoke confirmed `golden_plus_neighbors`, `--hyre-cache-path`, and
  `--retrieval-cache-path` are visible from `eval/eval_harness.py --help`.
- CLI smoke confirmed both retrieval-cache scripts expose expected arguments.
- Local SCALR raw-question retrieval-cache smoke wrote one row and
  `scripts/audit_retrieval_cache.py` validated it with no duplicate, missing,
  empty, or too-short rows.
- Cached retrieval hydration smoke returned `cache_hit=True` and one SCALR
  passage id without launching an LLM call.
- Unit tests passed: `tests/test_score_retrieval_qrels.py`,
  `tests/test_eval_metrics.py`, `tests/test_formatter.py`, and
  `tests/test_sanitizer.py` (`27 passed`).

Still needs HPC smoke before full launches:

- vLLM smokes for Gemma 4 E4B and Gemma 4 26B using
  `scripts/hpc/slurm_snap_hyre_vllm_smoke.sh`.

Remote collection check on 2026-05-12:

- `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db/chroma.sqlite3`
  has `casehold_holdings`, `housing_statutes`, `legal_passages`,
  `legalbench_scalr_holdings`, and `musique_passages`.
- `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/chroma_db/chroma.sqlite3`
  exists but did not list collections in the read-only check. Prefer syncing
  this branch into the main `LegalRagAgent` HPC checkout or explicitly pointing
  `CHROMA_DB_DIR` at the populated Chroma directory.
- `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive`
  is a clean clone of this branch. It uses local symlinks to the populated
  `datasets` and `chroma_db` directories, excluded via `.git/info/exclude`.

Compute-node cache smoke:

- SLURM job `68366` completed cleanly on 2026-05-12 in 5:55 using
  `scripts/hpc/slurm_snap_hyre_cache_smoke.sh`.
- The smoke validated imports, populated Chroma access, `snap_hyre` runner
  registration, and one raw-question retrieval-cache row for BarExamQA,
  HousingQA, CaseHOLD, and LegalBench-SCALR.
- Each smoke cache passed `scripts/audit_retrieval_cache.py` with zero
  duplicate keys, missing idxs, empty retrieval rows, or rows shorter than
  `min_k=3`.
- Smoke stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/68366.out`.

Provider smoke status:

- API smoke job `68369` was cancelled intentionally after the first Groq
  `llm_only` row showed `predicted_answer=None`. Root cause was the smoke
  script's overly small `LLM_MAX_COMPLETION_TOKENS=256`, which truncated the
  model before a parseable final answer. The row is rejected as a smoke result.
- `scripts/hpc/slurm_snap_hyre_api_smoke.sh` and
  `scripts/hpc/slurm_snap_hyre_vllm_smoke.sh` now default to
  `LLM_MAX_COMPLETION_TOKENS=768` and fail hard if any detail row has missing
  `predicted_answer`.
- Hardened API smoke job `68372` completed cleanly on 2026-05-12 in 6:23.
  It validated `llm_only` and `snap_hyre` for `groq-llama70b` and
  `or-gemma4-26b`; all four rows had `missing_predicted_answer=0`,
  no parse failures, no long-answer rows, and `snap_hyre` had nonempty
  retrieval. Smoke stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/68372.out`.
- New Snap-HyRE runs now use `snap_hyre` as the retrieval trace/cache label.
  Legacy `rag_snap_hyde_2call` remains accepted by `scripts/build_hyre_cache.py`
  for older detail logs.
- Gemma 4 E4B vLLM smoke job `68377` completed cleanly on 2026-05-12 in 16:04
  using `google/gemma-4-E4B-it`. It validated `llm_only` and `snap_hyre` with
  `missing_predicted_answer=0`, no parse failures, no long-answer rows, and
  nonempty Snap-HyRE retrieval. Smoke stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/68377.out`.
- Gemma 4 26B vLLM smoke job `68417` is rejected as an infrastructure failure:
  it landed on an A60 with 47.4 GiB GPU memory and failed during vLLM startup
  with `torch.OutOfMemoryError`. This does not validate or invalidate the
  harness. Retry on H100 or fall back to the already clean OpenRouter
  `or-gemma4-26b` provider smoke. Smoke stdout:
  `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/68417.out`.

## Open Questions for the Team

1. If HousingQA needs replacement, what legal retrieval benchmark has a frozen
   local corpus/qrels and can be run without harness contortion?
2. If `rag_rewrite` becomes competitive, should we later add rewrite-query
   replay caches for a more exact top-k curve, or is second-stage uncached
   control coverage enough?
