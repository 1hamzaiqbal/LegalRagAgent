# Adaptive HyRE Frontier N=200 Analysis (2026-05-10)

This note records the clean explicit-selector N=200 run and the main follow-up
interpretation. Use it with `docs/adaptive_hyre_frontier_n200_latest.md` and
`docs/adaptive_hyre_final_frontier_n200_latest.md`.

## What Landed

The callable `adaptive_snap_hyre_frontier` selector was run on all four legal
datasets at N=200 with `or-gemma4-26b`.

| Dataset | Selector route | Accuracy | Audit | Main comparison |
|---|---|---:|---|---|
| barexam | `frontier_barexam_v2` | 84.0% | PASS after one-row repair | +4.0pp vs `rag_simple`, p=0.2005 |
| housing | `frontier_housing_diverse` | 62.0% | PASS | +1.5pp vs `rag_state_filter`, -1.0pp vs `snap_hyre_state` |
| casehold | `frontier_casehold_diverse` | 70.5% | PASS | -2.5pp vs `rag_simple`, p=0.4244 |
| legalbench_scalr | `frontier_scalr_plain_snap_hyde` | 76.5% | PASS | +0.5pp vs `rag_snap_hyde_2call`, p=1.0000 |

Barexam needed a repair because row `qa_nan_mbe_535` produced a malformed
Snap/HyRE parse even though the final answer was correct. Repair job `67279`
reran sampled row `[187:188]`, predicted `A`, and passed audit. The repaired
full 200-row log is now the selector Barexam source.

## Selector Versus Component Frontier

The explicit selector is clean, but it is not the strongest per-dataset table.
The stronger component frontier remains:

| Dataset | Strongest clean component | Accuracy |
|---|---|---:|
| barexam | `adaptive_snap_hyre_v2` | 86.0% |
| housing | `adaptive_snap_hyre_diverse` | 63.5% |
| casehold | `adaptive_snap_hyre_diverse` | 73.5% |
| legalbench_scalr | `adaptive_snap_hyre_frontier` / plain two-call route | 76.5% |

The selector reuses the intended route families, but repeated online HyRE
generation is stochastic. On the same 200 sampled labels:

| Comparison | Prior component correct | Selector correct | Selector wins | Selector losses | Same predicted answer |
|---|---:|---:|---:|---:|---:|
| Barexam v2 -> selector | 172 | 168 | 6 | 10 | 181/200 |
| Housing diverse -> selector | 127 | 124 | 9 | 12 | 179/200 |
| CaseHOLD diverse -> selector | 147 | 141 | 6 | 12 | 174/200 |
| SCALR two-call -> selector | 152 | 153 | 10 | 9 | 177/200 |

This is not a route-registration failure. It is a stability problem in the
generated HyRE/query/answer path: most generated retrieval queries differ across
runs, and 10-26 labels flip predictions depending on dataset.

## Current Interpretation

The defensible claim is that adaptive HyRE can execute cleanly across legal
benchmarks and can match or improve the best known two-call policy on several
tasks, but the literal online selector is still sensitive to generation
variance. For reporting, use the component frontier as the strongest evidence
and use the selector run as deployment validation.

For research continuation, the next adaptive method should reduce stochastic
variance rather than add another broad route. Good targeted options:

1. Deterministic HyRE cache: generate each HyRE passage once per sample, persist
   it, and evaluate downstream answer policies against a fixed retrieval view.
2. Cheap verifier/arbitrator: when the model is likely unstable, compare the
   raw-question answer and HyRE answer with one additional judge/rerank call.
3. Dataset-typed confidence rule: route Housing residuals to a yes/no
   consistency check, and route CaseHOLD residuals to option-level reranking
   instead of another free-form final answer.

Do not claim the selector is universally best. Claim that the frontier evidence
identifies reusable bottleneck-specific interventions, and that selector-level
deployment is now clean but needs stability/arbitration to preserve the best
component result on CaseHOLD and Housing.
