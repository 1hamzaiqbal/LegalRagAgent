# Snap-HyRE Prelaunch Readiness - 2026-05-13

Purpose: preserve the current prelaunch state before full comprehensive runs.
The north star is a simple fixed Snap-HyRE method, reported primarily as a
retrieval-exposure improvement and secondarily as downstream answer accuracy.

## Current Model Grid

Use API providers for the main grid:

| Label | Provider model | Role |
|---|---|---|
| `or-ministral-8b` | `mistralai/ministral-8b-2512` | small API-only model row |
| `or-gemma4-26b` | `google/gemma-4-26b-a4b-it` | default/top-k decision model |
| `groq-llama70b` | `llama-3.3-70b-versatile` | Llama 70B row |

Historical `google/gemma-4-E4B-it` was a local/cluster vLLM checkpoint. It is
not a launch requirement for the current API-only grid. Do not substitute
`or-gemma3n-e4b` as if it were the same model.

## Strictness Gates

- `NO_SILENT_FALLBACK=1` is required.
- OpenRouter requests keep the requested model id fixed and disable model
  fallback. Same-model provider routing is acceptable only when explicit and
  logged; silent model/method/cache substitution is invalid.
- Unknown providers fail closed.
- Local OpenRouter runners pace calls by default:
  `LLM_CALL_MIN_INTERVAL_SEC=2.0`,
  `LLM_CALL_RATE_LIMIT_COOLDOWN_SEC=8.0`.
- Pacing is operational only; it does not change prompts, model labels,
  retrieval, or token caps.

## Cache Process Flow

1. Build deterministic retrieval caches for model-invariant methods:
   `raw_question` and `golden_neighbors`.
2. Build generation caches for generated-query methods:
   `rag_hyde` and `snap_hyre`.
3. Build retrieval-id caches from those generated passages.
4. Audit qrel alignment and retrieval-cache health.
5. Compile retrieval matrices at `k in {1,3,5,10}`.
6. Run answer cells using `--retrieval-cache-path` and, for generated methods,
   `--hyre-cache-path`.
7. Promote a row only after detail-log health checks pass:
   no errors, no missing predictions, no parse failures, no fallback markers,
   no required cache misses, and no long/truncated-looking outputs.

This lets top-1/top-5/top-10 answer experiments reuse the same passage-id
caches. For top-1, use the first cached passage id; for top-5/top-10, use the
first five or ten ids in order.

## Current Retrieval Signal

The q100 retrieval matrices are clean. For `or-gemma4-26b` generated caches:

| Method | Macro Hit@5 | Macro Hit@10 | Macro MRR@5 | Macro MRR@10 |
|---|---:|---:|---:|---:|
| `rag_simple` raw question | 0.1775 | 0.2150 | 0.1135 | 0.1184 |
| `rag_hyde` | 0.4050 | 0.4575 | 0.3078 | 0.3152 |
| `snap_hyre` | 0.3700 | 0.4125 | 0.2956 | 0.3010 |

Retrieval headline should be phrased carefully: generated queries strongly
improve retrieval exposure over raw question on average, while `rag_hyde` is
currently above `snap_hyre` on the q100 macro retrieval matrix. Snap-HyRE has
dataset-specific wins, especially LegalBench-SCALR at k=5/k=10.

## Live q100 Slice Sanity

Gemma 26B, sample rows 0:5 from the q100 seed-42 sample, strict replay caches:

| Dataset | k | `rag_simple` | `rag_hyde` | `snap_hyre` |
|---|---:|---:|---:|---:|
| BarExamQA | 5 | 5/5 | 5/5 | 5/5 |
| BarExamQA | 10 | 5/5 | 5/5 | 5/5 |
| HousingQA | 5 | 4/5 | 4/5 | 3/5 |
| HousingQA | 10 | 4/5 | 3/5 | 2/5 |
| CaseHOLD | 5 | 3/5 | 4/5 | 3/5 |
| CaseHOLD | 10 | 3/5 | 4/5 | 4/5 |
| LegalBench-SCALR | 5 | 5/5 | 5/5 | 5/5 |
| LegalBench-SCALR | 10 | 4/5 | 5/5 | 5/5 |

Interpretation: the answer path is structurally ready, but the tiny slice is
not a top-k winner. It does show that HousingQA can distort the fixed-method
story, and that k=10 can help some generated-query rows while hurting some raw
RAG rows.

## Prompt/Context Audit

A dry cache-backed runner with LLM calls monkeypatched verified all q100
datasets and modes at k=10:

- 100/100 retrieval-cache hits for every dataset/mode.
- 100/100 HyRE generation-cache hits for `rag_hyde` and `snap_hyre`.
- No parse/fallback/cache failures.
- Worst approximate user prompt size was HousingQA, about 18-19k tokens by
  character heuristic. Other datasets were far smaller.

HousingQA is the main prompt-length and format caveat. It is runnable, but it
should not be allowed to contort the shared method definition.

## LegalSearchQA Feasibility

LegalSearchQA is worth a separate feasibility note, not a main-run replacement
yet. The Hugging Face dataset card lists 50 test rows, multiple-choice QA,
fields for question/choices/answer/rationale/domain/category/difficulty, and
`source_url` / `source_name` metadata. It is tagged for
retrieval-augmented-generation, but it does not ship a local frozen corpus in
the same form as the current Chroma-backed benchmarks.

Implication: LegalSearchQA may be useful as an appendix or feasibility probe
after building a frozen corpus from the cited source URLs. It should not block
the current four-benchmark comprehensive grid unless we explicitly decide to
spend time on corpus construction and source-page stability.

Source checked: https://huggingface.co/datasets/boqiny/LegalSearchQA

## Launch Recommendation

Ready to launch deliberate answer sweeps, with this order:

1. Use `RETRIEVAL_K=5` as the shared main-grid answer setting. The 2026-05-14
   prelaunch probe found no downstream reason to promote k=10 before the
   comprehensive runs.
2. Run the full corpus main table one provider and one dataset at a time.
3. Keep golden-passage and golden-plus-neighbors as analysis controls, not as
   main method rows.

Do not queue many jobs at once. Keep the monitoring loop tight and treat failed
rows as failed, not silently substitutable.

## Choice-Aware Probe Addendum

See `docs/choice_aware_retrieval_probe_2026-05-13.md` for probe-only q20
retrieval diagnostics on `or-gemma4-26b`.

Practical takeaways:

- Keep `rag_simple` raw stem/context-only; do not let choices leak into this
  baseline retrieval query.
- Distinguish generated-query baselines as blind vs choice-aware when analyzing
  CaseHOLD/SCALR. Candidate-aware generated HyDE is strong on holding tasks and
  should not be conflated with raw RAG.
- `snap_choice_hyre` is promising enough for q50 retrieval-only CaseHOLD/SCALR
  checks, but not yet canonical for the comprehensive answer grid.
- `snap_hyre_anchor` did not improve Hit@5 in the q20 checks when it reused the
  same generated Snap-HyRE passage; leave it out unless larger sweeps show a
  consistent MRR/Hit@10 benefit.
- OpenRouter `DekaLLM` returned an upstream 401 during BarExam probing.
  `NO_SILENT_FALLBACK=1` correctly hard-stopped the row. Use
  `OPENROUTER_PROVIDER_IGNORE=dekallm` for OpenRouter retries if that connector
  remains flaky; fallback stays disabled.

## q50 Choice-Aware Follow-Up

See `docs/choice_aware_retrieval_q50_2026-05-14.md` for the q50 SCALR and
CaseHOLD retrieval-only follow-up.

Health gates passed on both combined q50 logs: 300/300 expected rows, zero
errors, zero parse failures, zero fallback rows, zero answer-artifact rows, zero
empty retrieval rows, zero think-tag rows, and 50/50 qrel alignment.

Key metrics:

| Dataset | Best Hit@5 row | Snap-HyRE Hit@5 | Raw RAG Hit@5 |
|---|---:|---:|---:|
| LegalBench-SCALR | `snap_hyre` 0.76 | 0.76 | 0.56 |
| CaseHOLD | `rag_hyde_choice` 0.66 | 0.58 | 0.24 |

Launch implication: keep `snap_hyre` in the main ladder and keep
blind-vs-choice HyDE as retrieval analysis, especially for holding-option
datasets. Do not promote `snap_choice_hyre` to the comprehensive answer grid
unless a downstream q20/q50 answer slice shows a real accuracy gain.

## Top-K Prelaunch Follow-Up

See `docs/top_k_prelaunch_probe_2026-05-14.md`.

The fine-grained q100 retrieval curve was compiled from cached top-10 passage
ids for k=1 through k=10. Macro Hit@k rises from 0.3175 at k=5 to 0.3617 at
k=10, but macro MRR only rises from 0.2389 to 0.2449. On the default-model
BarExam q100 downstream check, k=10 did not beat k=5:

| Mode | k=5 | k=10 |
|---|---:|---:|
| `rag_simple` | 83/100 | 81/100 |
| `rag_hyde` | 87/100 | 84/100 |

The launch default is therefore k=5. Report k=1..10 retrieval curves for
analysis, and reserve k=10 downstream answer runs for later targeted ablation
if the main grid needs it.

Operational guardrail from this probe: answer cells now fail closed when
`LLM_MAX_COMPLETION_TOKENS` is below `EVAL_MIN_COMPLETION_TOKENS` (default
2048), so stale `.env` caps cannot silently create truncation-prone runs.

Current provider smoke on 2026-05-14 passed for all three launch labels on a
q1 LegalBench-SCALR `llm_only` call: `or-ministral-8b`, `or-gemma4-26b`, and
`groq-llama70b`.
