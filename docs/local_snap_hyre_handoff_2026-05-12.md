# Local Snap-HyRE Execution Handoff - 2026-05-12

Paste this into the Codex instance running on the Windows local machine that
will own embedding, cache construction, full-corpus API sweeps, monitoring, and
result validation. This machine is not expected to have WUSTL/HPC access.

## Mission

Produce a full-corpus, retrieval-first Snap-HyRE evaluation package across four
legal benchmarks: BarExamQA, HousingQA, Legal-Link-EU, and MASLegalBench. Use
the same fixed method ladder across datasets and report retrieval exposure
metrics alongside downstream answer accuracy. CaseHOLD and LegalBench-SCALR are
historical/superseded for the active main matrix unless explicitly re-added.

Primary claim target:

- Snap-HyRE should be evaluated for retrieval exposure versus standard RAG:
  Hit/Recall@1, Hit/Recall@5, Hit/Recall@10, and MRR@10 where gold ids are
  aligned, with MASLegalBench labeled as a same-source proxy.
- Downstream answer accuracy is the harder transfer target. Report it honestly
  and do not contort dataset-specific harnesses to force a positive result.

Current branch:

```bash
git clone -b snap_hyre_comprehensive https://github.com/shrango/adaptive-plan-and-solve-agent.git LegalRagAgent
cd LegalRagAgent
git status --short --branch
```

Start by reading:

```text
AGENTS.md
CLAUDE.md
docs/README.md
docs/snap_hyre_comprehensive_plan_2026-05-12.md
docs/snap_hyre_experiment_runbook_2026-05-12.md
docs/local_api_mirror_setup_2026-05-12.md
docs/snap_hyre_completion_audit_2026-05-12.md
scripts/local/README.md
```

## API Keys

Put secrets in a local untracked `.env` at the repo root. Do not commit it.
On the original Mac checkout, the source file is:

```text
/Users/hamzaiqbal/grad/LegalRagAgent/.env
```

Copy that file to the other machine out of band, or recreate the same variables
there. Do not add API keys to this repo, a commit, or a shared handoff.

Required for the planned API-first runs:

```env
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
LLM_MAX_COMPLETION_TOKENS=2048
```

Optional:

```env
HF_TOKEN=...
```

Use these providers unless a smoke test proves one is broken:

- `groq-llama8b`: API-only small-model replacement via Groq.
- `or-gemma4-26b`: Gemma 4 26B via OpenRouter.
- `groq-llama70b`: Llama 3.3 70B Versatile via Groq.

Exact historical `google/gemma-4-E4B-it` is not the same as
`or-gemma3n-e4b`. Use the API-only small-model replacement for the
comprehensive package unless the exact E4B endpoint becomes available and we
intentionally add it as a vLLM/provenance row.

## Local Environment

On Windows, use WSL Ubuntu for the main run if available. The checked-in local
helpers are Bash scripts, and the Python/Chroma stack is less fragile under WSL
than native PowerShell. Clone the repo inside the WSL filesystem, not under
`/mnt/c`, if disk space allows.

Use local Chroma and offline HF settings once data and embeddings are present:

```bash
uv sync
export CHROMA_DB_DIR="$PWD/chroma_db"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_CROSS_ENCODER=0
export LLM_MAX_COMPLETION_TOKENS=2048
```

Keep the cross-encoder enabled for any retrieval cache or answer row that may
be promoted. `DISABLE_CROSS_ENCODER=1` is acceptable only for explicitly
labeled dense-only speed smokes.

The Windows machine should not assume WUSTL/HPC access. Populate prerequisites
by downloading datasets and embedding locally, or by copying `datasets/` and
`chroma_db/` from a machine you can directly access. Do not block on WUSTL SSH.

Budget at least 80 GB free for the full local mirror. The last checked populated
mirror sizes were about 33 GB for `chroma_db/` and 4.8 GB for `datasets/`, with
extra room needed for retrieval/generation caches and logs.

## Dataset And Embedding Setup

Download or verify datasets:

```bash
uv run python utils/download_data.py --check || uv run python utils/download_data.py
uv run python utils/download_housingqa.py --check || uv run python utils/download_housingqa.py
uv run python utils/download_new_datasets.py casehold
uv run python utils/download_new_datasets.py legalbench_scalr
```

Build or verify embeddings:

```bash
export CHROMA_DB_DIR="$PWD/chroma_db"
mkdir -p "$CHROMA_DB_DIR"
uv run python utils/fast_embed.py barexam --resume
uv run python utils/fast_embed.py housing --resume
uv run python utils/fast_embed.py casehold --resume
uv run python utils/fast_embed.py legalbench_scalr --resume
uv run python utils/fast_embed.py status
```

For a clean rebuild, omit `--resume`; that deletes/rebuilds the target
collection. Do not rebuild a populated collection unless disk and runtime are
confirmed.

## BarExam Alignment Status

Before promoting BarExam retrieval metrics, run the qrel alignment audit. The
local WSL collection was patched on 2026-05-13 by appending the validation/test
passages from `datasets/barexam_qa/passages/passages.tsv` into
`legal_passages` with the same GTE-large encoder.

Current local status:

- `legal_passages`: 856,835 docs.
- BarExam qrel alignment: 1149/1149 unique gold ids found.
- Full-set BarExam Hit@k/MRR is promotable after rebuilding caches against this
  patched collection.

If another machine still has the train-only 686,324-doc collection, run:

```bash
uv run python utils/augment_barexam_collection.py
uv run python scripts/audit_retrieval_id_alignment.py \
  --dataset barexam --questions full --min-exists 1.0
```

Rerunning the augmentation is idempotent: it checks existing Chroma ids and
only embeds passages still missing from the target collection.

## Retrieval Cache And Top-k Diagnostics

First build deterministic raw-question and golden-neighbor caches. These choose
the universal top-k and provide retrieval-first figures without spending LLM
calls.

Preferred local helper:

```bash
scripts/local/build_retrieval_caches.sh
```

Cache filenames include question scope by default, for example
`barexam_qfull_seed42_raw_question_k10.jsonl` and
`legalbench_scalr_q50_seed42_gemma4-26b_snap_hyre_k10.jsonl`. Keep the same
`QUESTIONS`, `SEED`, `SAMPLE_START`, `SAMPLE_END`, collection, embedding model,
and source filter between cache build and answer replay.

Equivalent explicit command:

```bash
mkdir -p caches/retrieval/full docs/generated

for dataset in barexam housing casehold legalbench_scalr; do
  uv run python scripts/audit_retrieval_id_alignment.py \
    --dataset "$dataset" --questions full --min-exists 0.95 \
    | tee "caches/retrieval/full/retrieval_id_alignment_${dataset}.txt" || true

  for query_type in raw_question golden_neighbors; do
    uv run python scripts/build_retrieval_cache.py \
      --dataset "$dataset" --questions full --query-type "$query_type" \
      --max-k 10 \
      --out "caches/retrieval/full/${dataset}_${query_type}_k10.jsonl"

    uv run python scripts/audit_retrieval_cache.py \
      --cache "caches/retrieval/full/${dataset}_${query_type}_k10.jsonl" \
      --dataset "$dataset" --query-type "$query_type" \
      --min-k 10 --ks 1,3,5,10
  done
done

uv run python scripts/compile_retrieval_cache_matrix.py \
  --cache 'caches/retrieval/full/*.jsonl' \
  --ks 1,3,5,10 --min-k 10 \
  --out-md docs/generated/retrieval_cache_matrix.md \
  --out-csv docs/generated/retrieval_cache_matrix.csv
```

Use the matrix to pick one universal retrieval-k for the main answer table.
Keep top-k curves as an analysis figure.

## Canonical Method Ladder

Main table rows:

```text
llm_only
rag_simple
rag_rewrite
rag_hyde
snap_hyre
golden_passage
golden_plus_neighbors
```

Rules:

- Keep the same selected top-k across comparable retrieval rows.
- Do not use a dataset-specific contorted baseline.
- Treat `snap_hyre` as the method name. `rag_snap_hyde_2call` is only a legacy
  alias for older logs.
- Use retrieval caches for deterministic `rag_simple` and
  `golden_plus_neighbors`.
- Build generation caches for `rag_hyde` and `snap_hyre` before repeated
  answer runs.

## Generation Caches For HyDE And Snap-HyRE

Start with one dataset and one provider at `N=50`; then full corpus.

```bash
provider=or-gemma4-26b
model_label=gemma4-26b
dataset=legalbench_scalr
questions=50

for mode in rag_hyde snap_hyre; do
  uv run python scripts/build_generation_cache.py \
    --mode "$mode" --provider "$provider" --dataset "$dataset" \
    --questions "$questions" --seed 42 \
    --tag "local-gen-${model_label}-${dataset}-${mode}-${questions}" \
    --out "caches/hyre/full/${dataset}_${model_label}_${mode}.jsonl" \
    --resume
done
```

Build retrieval caches from those generations:

```bash
uv run python scripts/build_retrieval_cache.py \
  --dataset "$dataset" --questions "$questions" --query-type hyde_cache \
  --hyre-cache-path "caches/hyre/full/${dataset}_${model_label}_rag_hyde.jsonl" \
  --max-k 10 \
  --out "caches/retrieval/full/${dataset}_${model_label}_rag_hyde_k10.jsonl"

uv run python scripts/build_retrieval_cache.py \
  --dataset "$dataset" --questions "$questions" --query-type hyre_cache \
  --hyre-cache-path "caches/hyre/full/${dataset}_${model_label}_snap_hyre.jsonl" \
  --max-k 10 \
  --out "caches/retrieval/full/${dataset}_${model_label}_snap_hyre_k10.jsonl"
```

Audit both before using them:

```bash
uv run python scripts/audit_retrieval_cache.py \
  --cache "caches/retrieval/full/${dataset}_${model_label}_rag_hyde_k10.jsonl" \
  --dataset "$dataset" --query-type hyde_cache --min-k 10 --ks 1,3,5,10

uv run python scripts/audit_retrieval_cache.py \
  --cache "caches/retrieval/full/${dataset}_${model_label}_snap_hyre_k10.jsonl" \
  --dataset "$dataset" --query-type hyre_cache --min-k 10 --ks 1,3,5,10
```

## API Smoke Gate

Before full runs, smoke each provider with `snap_hyre` and `rag_simple`.

Preferred local helper:

```bash
scripts/local/run_api_smoke.sh
```

Equivalent explicit command:

```bash
for provider in groq-llama8b or-gemma4-26b groq-llama70b; do
  for mode in rag_simple snap_hyre; do
    uv run python eval/eval_harness.py \
      --mode "$mode" --provider "$provider" --dataset legalbench_scalr \
      --questions 2 --retrieval-k 3 \
      --tag "local-api-smoke-${provider}-${mode}"
    latest=$(ls -t logs/eval_${mode}_${provider}_*legalbench_scalr*local-api-smoke*detail.jsonl | head -1)
    uv run python scripts/analyze_detail_flags.py "$latest"
  done
done
```

Do not launch full runs if the smoke log shows auth errors, rate-limit loops,
missing predictions, empty retrieval, runaway output, or parse failures.

## Full Answer Runs

After top-k is selected and caches are validated, run one dataset/provider cell
at a time. Keep at most one or two active local jobs.

Preferred local helper:

```bash
PROVIDER=or-gemma4-26b MODEL_LABEL=gemma4-26b DATASET=legalbench_scalr \
QUESTIONS=full RETRIEVAL_K=5 scripts/local/run_answer_cell.sh
```

Run the generated-query cache helper before `rag_hyde` or `snap_hyre` answer
rows:

```bash
PROVIDER=or-gemma4-26b MODEL_LABEL=gemma4-26b QUESTIONS=full \
scripts/local/build_generation_caches.sh
```

Example:

```bash
provider=or-gemma4-26b
model_label=gemma4-26b
dataset=legalbench_scalr
k=5
questions=full

for mode in llm_only rag_simple rag_rewrite rag_hyde snap_hyre golden_passage golden_plus_neighbors; do
  extra=()
  case "$mode" in
    rag_simple)
      extra+=(--retrieval-cache-path "caches/retrieval/full/${dataset}_raw_question_k10.jsonl")
      ;;
    rag_hyde)
      extra+=(--hyre-cache-path "caches/hyre/full/${dataset}_${model_label}_rag_hyde.jsonl")
      extra+=(--retrieval-cache-path "caches/retrieval/full/${dataset}_${model_label}_rag_hyde_k10.jsonl")
      ;;
    snap_hyre)
      extra+=(--hyre-cache-path "caches/hyre/full/${dataset}_${model_label}_snap_hyre.jsonl")
      extra+=(--retrieval-cache-path "caches/retrieval/full/${dataset}_${model_label}_snap_hyre_k10.jsonl")
      ;;
    golden_plus_neighbors)
      extra+=(--retrieval-cache-path "caches/retrieval/full/${dataset}_golden_neighbors_k10.jsonl")
      ;;
  esac

  EVAL_TRACE_CALLS=1 EVAL_TRACE_EVENTS=1 \
  uv run python eval/eval_harness.py \
    --mode "$mode" --provider "$provider" --dataset "$dataset" \
    --questions "$questions" --seed 42 --retrieval-k "$k" \
    --tag "local-snap-hyre-${model_label}-${dataset}-${mode}-full-k${k}" \
    "${extra[@]}"

  latest=$(ls -t logs/eval_${mode}_${provider}_*_${dataset}_*local-snap-hyre*detail.jsonl | head -1)
  uv run python scripts/analyze_detail_flags.py "$latest"
done
```

## Monitoring And Promotion Rules

Every active run needs a periodic health check:

```bash
tail -n 80 logs/<active-stdout-or-log>
rg -n "Traceback|RateLimit|rate.limit|auth|401|403|429|timeout|CUDA|ECC|empty retrieval|missing prediction|parse|runaway" logs/
```

Promote a row only if all are true:

- Detail JSONL exists and has the expected row count.
- `scripts/analyze_detail_flags.py` passes or the caveat is written down.
- Retrieval-cache audit exists for rows with retrieval exposure claims.
- Qrel alignment is valid for Hit@k/MRR claims.
- The row is added to `docs/signoff_log.md`.
- Any generated table cites source logs, not stale narrative notes.

Rejected or caveated rows should be recorded explicitly instead of silently
replaced.

## Suggested Run Order

1. Pull branch and set `.env`.
2. Verify disk, datasets, Chroma path, and API smoke.
3. Embed or copy `legalbench_scalr` and `casehold`; smoke retrieval cache.
4. Embed or copy `barexam` and `housing`.
5. Build raw-question and golden-neighbor retrieval caches for all four.
6. Run qrel alignment and decide BarExam retrieval reporting policy.
7. Pick a universal top-k from `docs/generated/retrieval_cache_matrix.md`.
8. For each provider, build `rag_hyde` and `snap_hyre` generation caches one
   dataset at a time.
9. Run `N=50` answer ladder for one dataset/provider, validate logs, then scale
   to full corpus.
10. Expand to all three providers only after the first full cell is clean.
11. Regenerate tables/plots and update `docs/signoff_log.md`.
12. Commit and push after every clean batch of promoted rows.

Rebuild source-gated package status after each clean batch:

```bash
scripts/local/build_result_package.sh
```

## What Not To Do

- Do not queue many long jobs without a health monitor.
- Do not report BarExam retrieval metrics over unaligned qrels.
- Do not call pending partial rows full-corpus rows.
- Do not introduce dataset-specific harness changes just to rescue downstream
  accuracy.
- Do not promote old diagnostic-controller/adaptive-route artifacts into this
  branch narrative.

The intended outcome is a clean, simple package: one Snap-HyRE method, shared
comparison rows, shared top-k discipline, verified retrieval exposure metrics,
downstream accuracy where it transfers, and clear caveats where it does not.
