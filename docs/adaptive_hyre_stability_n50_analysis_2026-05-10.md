# Adaptive HyRE Stability N=50 Analysis (2026-05-10)

This note records the first stability-arbitration probe after the clean N=200
frontier selector run.

## Method

`adaptive_snap_hyre_stability` runs the audited frontier selector and a
dataset-specific control. If both answers agree, it keeps the frontier answer.
If they disagree or one side is unparsable, it spends one extra arbitration call.

Controls:

| Dataset | Control |
|---|---|
| barexam | `adaptive_snap_hyre_v2` |
| housing | `snap_hyre_state` |
| casehold | `adaptive_snap_hyre_diverse` |
| legalbench_scalr | `rag_snap_hyde_2call` |

## N=50 Results

All four stability logs pass strict detail-log health checks.

| Dataset | Stability acc | Avg calls | Arbitration rows | Arbitration acc | Agreement acc |
|---|---:|---:|---:|---:|---:|
| barexam | 41/50 = 82.0% | 4.14 | 7/50 | 4/7 | 37/43 |
| housing | 31/50 = 62.0% | 4.26 | 13/50 | 6/13 | 25/37 |
| casehold | 36/50 = 72.0% | 4.12 | 6/50 | 3/6 | 33/44 |
| legalbench_scalr | 38/50 = 76.0% | 4.10 | 5/50 | 2/5 | 36/45 |

Paired comparisons against the cheaper frontier/control rows are mostly
neutral:

| Dataset | Comparison | Delta | p |
|---|---|---:|---:|
| barexam | `adaptive_snap_hyre_frontier` -> stability | +2.0pp | 1.0000 |
| housing | `adaptive_snap_hyre_frontier` -> stability | +0.0pp | 1.0000 |
| casehold | `adaptive_snap_hyre_frontier` -> stability | +2.0pp | 1.0000 |
| legalbench_scalr | `adaptive_snap_hyre_frontier` -> stability | -2.0pp | 1.0000 |

## Interpretation

This is a useful negative/cost result. The stability layer is clean and behaves
as designed, but it roughly doubles LLM calls and does not reliably improve
accuracy at N=50. Arbitration fires on only 10-26% of rows, and arbitration
itself is not consistently more accurate than agreement rows.

Do not scale this exact method to N=200 as the next default. The more promising
direction is to stabilize the HyRE input before retrieval or use a narrower
verifier only for rows with stronger uncertainty signals. Candidate next steps:

1. Cache/reuse one HyRE passage per row to separate generation variance from
   retrieval and answer variance.
2. Add a cheap confidence signal before arbitration, instead of arbitrating any
   disagreement.
3. For Housing, target final yes/no calibration directly; the broad arbitrator
   often keeps the same wrong polarity.
4. For CaseHOLD, target option reranking rather than free-form arbitration.
