# Snap-HyRE Comprehensive Plan - 2026-05-12

## Mission

Build a clean, source-gated Snap-HyRE result package across four legal
benchmarks. The paper claim should be straightforward:

> A single snap-conditioned HyRE retrieval method can be applied consistently
> across legal QA benchmarks and should be judged first by retrieval exposure
> metrics, then by downstream answer accuracy.

This branch deliberately moves away from a bottleneck-aware adaptive controller
as the main contribution. Dataset-specific analysis remains useful, but the
headline method should not be a per-dataset routing policy.

## Current Branch State

- Branch: `snap_hyre_comprehensive`
- Remote: `shrango/snap_hyre_comprehensive`
- Starting point: prior `final_snap_hyre` meeting branch
- Cleanup rule: archive stale diagnostic/adaptive working notes; do not delete
  provenance.
- Paper source to build from: `/Users/hamzaiqbal/Downloads/main.tex`

## Benchmarks

Use four legal benchmarks unless a benchmark becomes structurally incompatible
with the fixed-method story.

| Benchmark | Keep? | Why | Main risk |
|---|---:|---|---|
| BarExamQA | yes | Full-corpus prior evidence and direct related-work comparability. | Reasoning-heavy; retrieval lift may not convert to accuracy. |
| HousingQA | yes, replacement-eligible | Statutory retrieval and yes/no legal entailment; related-work benchmark. | Yes/no format and jurisdiction filtering can make it feel unlike the other MC tasks. |
| CaseHOLD | yes | Holding-option task; exposes retrieval-vs-answer conversion. | Gold retrieval may improve faster than answer accuracy. |
| LegalBench-SCALR | yes | Legal holding-selection task with local Chroma coverage and full-ish runs. | Candidate depth saturates quickly; Snap-HyRE may only match baseline. |

LegalSearchQA should stay related work or a separate current-law web-search
diagnostic for now. It has only 50 questions and no frozen local corpus/qrels,
so it is not a controlled fixed-corpus Snap-HyRE benchmark without extra
snapshotting work.

## Models

Run the same method ladder across:

| Model label | Provider path | Role |
|---|---|---|
| Gemma 4 E4B | HPC vLLM only for exact-model rows | small/cheap open model axis |
| Gemma 4 26B | OpenRouter `or-gemma4-26b` | main Gemma result |
| Llama 3.3 70B Versatile | Groq `groq-llama70b`, OpenRouter fallback | cross-family larger model |

Before full launches, run a provider smoke and one N=5 harness smoke per
dataset/method family. Do not launch broad sweeps if any provider returns
auth/rate-limit/model-format errors.

Default future launches should use API providers for answer/generation sweeps.
The vLLM path is now a fallback for exact Gemma 4 E4B coverage or API outages,
not the default execution path. OpenRouter's current model list exposes Gemma 4
26B/31B but not a true Gemma 4 E4B endpoint; `or-gemma3n-e4b` and
`or-gemma3-4b` are acceptable small-Gemma API controls only if labeled as such,
not as Gemma 4 E4B.

## Fixed Method Ladder

The main table should look like an inherited ablation ladder. All retrieval
methods in a comparison must use the same chosen `k`.

| Row | Method | Calls | Purpose |
|---:|---|---:|---|
| 0 | `llm_only` | 1 | Parametric floor; no retrieval. |
| 1 | `rag_simple` | 1 | Raw-question retrieval baseline. |
| 2 | `rag_hyde` | 2 | HyDE without snap reasoning. |
| 3 | `snap_hyre` | 2 | Primary Snap-HyRE method: snap plus HyRE passage in one call, then final answer. Legacy alias: `rag_snap_hyde_2call`. |
| 4 | `golden_passage` | 1 | Gold-context diagnostic control, not deployable. |
| 5 | `golden_plus_neighbors` | 1 | Gold passage plus 4 retrieved neighbors; implement only after retrieval cache exists. |
| 6 | `rag_rewrite` | 2 | Strong non-HyRE query-generation control in the main ablation table; uncached second-stage control because rewritten queries are model-specific. |

Keep `snap_only_in_final` as a small ablation if needed, but it should not be a
main row unless the final report needs to separate "reasoning in context" from
"reasoning-conditioned retrieval."

## Primary Metrics

Every promoted row should have these fields:

- `N`, dataset, model/provider, seed/slice, method, `k`, detail log.
- Downstream accuracy.
- Retrieval Hit/Recall@1, @5, @10 where gold ids exist.
- MRR@10 where rank is available.
- Gold retrieved but answer wrong.
- Gold missing but answer correct.
- Accuracy conditional on gold retrieved / not retrieved.
- Average LLM calls, output tokens, latency.
- Health status: clean, caveated, rejected.

For HousingQA, report both "any gold statute retrieved" and, where possible,
"all gold statutes retrieved" because multiple statutes may support one answer.

## Top-k Plan

Do not pick `k=5` by habit. First run retrieval-only/top-k diagnostics at
`k in {1, 3, 5, 10}` for `rag_simple`, `rag_hyde`, and `snap_hyre`.
Run `rag_rewrite` after the cached top-k choice as an uncached second-stage
control rather than making it part of the first universal-k selection.

Selection rule:

1. Choose a universal `k` if one value has the best or near-best macro-average
   retrieval exposure without major downstream degradation.
2. If the universal `k` is tied, prefer lower cost and less context noise.
3. If no universal `k` is credible, report the top-k curve as analysis and use
   a predeclared default, likely `k=5`, for the main method table.

Plots to generate:

- Macro-average retrieval Hit@k by method.
- Dataset-level Hit@k curves for `rag_simple` vs Snap-HyRE.
- Downstream accuracy vs k, same methods.
- Calls/tokens vs downstream accuracy for the main method ladder.

## Caching Requirements

The harness already supports Snap-HyRE generation replay through
`scripts/build_hyre_cache.py` and `eval/eval_harness.py --hyre-cache-path`.

Implemented on this branch:

1. Deterministic retrieval-id cache for raw-question retrieval.
2. Deterministic retrieval-id cache for Snap-HyRE retrieval when a HyRE
   generation cache is supplied.
3. Deterministic retrieval-id cache for golden-passage-neighbor retrieval.
4. Harness loader flag `--retrieval-cache-path` that can hydrate
   `evidence_store` from passage ids and slice top-k without running embedding
   search again.
5. Cache audit script that verifies row labels, corpus collection, embedding
   model, max_k, retrieved ids, and missing ids before answer generation.

Active commands:

- Build raw-question cache:
  `scripts/build_retrieval_cache.py --dataset barexam --questions 200 --query-type raw_question --max-k 10 --out caches/retrieval/barexam_raw_question_k10.jsonl`
- Build Snap-HyRE retrieval cache after a HyRE generation cache exists:
  `scripts/build_retrieval_cache.py --dataset barexam --questions 200 --query-type hyre_cache --hyre-cache-path caches/hyre/barexam_snap_hyre.jsonl --max-k 10 --out caches/retrieval/barexam_snap_hyre_k10.jsonl`
- Audit cache:
  `scripts/audit_retrieval_cache.py --cache caches/retrieval/barexam_snap_hyre_k10.jsonl --dataset barexam --query-type hyre_cache --min-k 10`
- Replay answer generation with cached retrieval:
  `uv run python eval/eval_harness.py --mode snap_hyre --dataset barexam --questions 200 --provider openrouter-gemma4-26b --retrieval-k 5 --hyre-cache-path caches/hyre/barexam_snap_hyre.jsonl --retrieval-cache-path caches/retrieval/barexam_snap_hyre_k10.jsonl`

Suggested cache schema:

```json
{
  "label": "qa_Criminal Law_123",
  "idx": "123",
  "dataset": "barexam",
  "query_type": "raw_question",
  "label_prefix": "simple",
  "collection": "legal_passages",
  "embedding_model": "thenlper/gte-large",
  "where": {},
  "max_k": 10,
  "retrieved_ids": ["..."],
  "scores": [0.0],
  "gold_ids": ["..."],
  "source_text_hash": "..."
}
```

## Golden Passage Paradox

Treat "golden passage worse than LLM-only" as an analysis question, not a
blocker. The working hypotheses are:

- The labeled gold passage may be sufficient for retrieval evaluation but not
  sufficient for answer selection.
- A single gold passage can distract or anchor the model away from a better
  parametric answer.
- MC answer extraction and option mapping can dominate context quality.
- Gold plus nearest neighbors may restore missing legal context.

Run paired audits only on rows where `golden_passage` is wrong and `llm_only`
or Snap-HyRE is correct.

## Launch Discipline

No six-job pileups. Use at most two or three active jobs at a time:

1. One retrieval-cache/top-k job.
2. One answer-generation job.
3. One optional provider/model smoke.

Every job must have:

- Unique tag: `snap-hyre-comp-v1-<dataset>-<model>-<n>-k<k>-<method>`.
- Stdout tail check for Tracebacks, auth/rate limits, CUDA/ECC, timeouts,
  empty retrieval, runaway output, and missing predictions.
- Local copied detail logs before promotion.
- `scripts/analyze_detail_flags.py` pass.
- Retrieval scoring pass through `scripts/score_retrieval_qrels.py` or a
  dataset-specific equivalent.
- Signoff entry before any table cites the row.

## One-week Execution Plan

1. **Branch hygiene and planning**: finish this cleanup, archive diagnostic
   controller notes, and write the new runbook.
2. **Harness robustness**: implement retrieval caches and `golden_plus_neighbors`.
3. **N=50/N=100 calibration**: test prompt stability, provider formatting, and
   top-k curves across all four benchmarks.
4. **N=200 confirmation**: select universal `k`, confirm the fixed method ladder
   with Gemma 26B and one Llama run.
5. **Full-corpus runs**: launch every available row for every
   benchmark/model/method cell using cached retrieval where possible.
6. **Report package**: produce main ablation table, retrieval table, top-k
   figures, model-transfer table, and concise failure analysis.

## Open Decisions

- If HousingQA needs replacement, what frozen legal retrieval benchmark can
  replace it without changing the harness contract?
- Whether to add a new internal retrieval label for `snap_hyre` or keep
  `snap_hyde_2call` as the retrieval-cache label for backward compatibility.
- If `rag_rewrite` becomes competitive enough to merit a later cached
  rewrite-query replay path.
