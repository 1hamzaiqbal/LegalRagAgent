# July 2026 local/EIT completion audit — 2026-07-17

## Purpose and scope

This audit reconciles the experiments performed around 2026-07-02 with the
files that remained locally and on EIT. It is a provenance and interpretation
gate, not a new paper table. The durable research synthesis is
[`wiki/snapshots/research-state-2026-07-17.md`](../wiki/snapshots/research-state-2026-07-17.md).

## Ledger repair

- `logs/experiments.jsonl` contains **671 valid JSON rows** after merging the
  20 July-only rows back into the 651-row historical ledger and de-duplicating
  exact rows.
- All 50 tracked detail logs that had appeared deleted were restored from Git.
- The larger July logs are represented by the 71-entry SHA/size/path manifest
  at `evidence/july_2026/manifests/local_july_detail_logs.tsv`.
- A pre-cleanup Git bundle, patch, untracked-file tarball, and checksums exist at
  `/Users/hamzaiqbal/grad/LegalRagAgent_recovery_20260717`.

## EIT job adjudication

| Job | Semantic outcome | Durable evidence | Cite status |
|---|---|---|---|
| 93598 | Failed: installed Transformers did not recognize `qwen3_5`, despite the wrapper printing `ALL DONE` | `evidence/july_2026/eit_job_logs/judge_lane_93598.out` | Failure provenance only |
| 93606 | Partial: zero-shot 63/399 completed, training OOMed and the missing adapter could not be scored | `judge_lane_93606.out` | Diagnostic only |
| 93629 | Cancelled after partial training; no final checkpoint/result | `judge_lane_93629.out` | Not citable |
| 93632 | Completed specialist BarExam judge: 82/399, Hit@5 0.2055 | `judge_lane_93632.out`, `judge_results/barexam_specialist_9b_20260702.json` | Citable with pool/label caveats |
| 93656 | Failed offline model-cache resolution | `judge_lane_93656.out` | Failure provenance only |
| 93658 | Cancelled | `judge_lane_93658.out` | Not citable |
| 93660 | Completed mixed legal judge: BarExam 88/399; Housing 277/500 | `judge_lane_93660.out`, `judge_results/mixed_legal_9b_20260702.json` | Citable with pool/label caveats |
| 93770 | Allocation policy trained; the EIT script did not itself perform the later paired policy evaluation | `judge_lane_93770.out`, local rung-2 analysis and wiki page | Cite the local analysis, not this stdout as an evaluated result |
| 93773 | Incomplete first OPD smoke: cold-start/readiness window expired | `opd_smoke_93773.out` | Failure provenance only |
| 93802 | End-to-end OPD plumbing passed: teacher ready, three finite student steps, checkpoint exists | `opd_smoke_93802.out` | Infrastructure validation only; no scientific performance claim |

## Stable result objects

The compact JSON files in `evidence/july_2026/judge_results/` preserve the
result cells needed to reason about the selector and label-semantics findings.
They are not all interchangeable:

- BarExam and Housing legal labels directly encode gold-passage membership in
  the candidate pool.
- SciDocs citation-proxy supervision underperformed zero-shot judgment; this is
  evidence that label semantics constrain what the judge learns, not a generic
  failure of trained reranking.
- FiQA shows the zero-shot judge already exceeds the CE baseline and that the
  trained row need not improve when headroom or label quality is limited.
- Mixed legal training retained both legal domains, but these are reranking
  results on constructed pools, not downstream answer-accuracy results.

## Result boundary

The July program supports four conclusions:

1. A trained small judge can expose gold evidence substantially better than
   the fixed CE reranker on the tested legal pools.
2. Retrieval exposure does not automatically convert into reader accuracy;
   the conversion depends on reader capability and the evidence regime.
3. Cheap external per-question allocation did not beat fixed arms, while the
   internalized policy learned coarse regime behavior but not the key
   `do-not-retrieve` decision for strong readers.
4. The OPD software path works end-to-end, but job 93802 does **not** validate
   OPD as a learning method. After the SDAR read, ungated OPD is explicitly a
   plumbing baseline, not the next scale experiment.

Anything stronger must return to the row-level detail logs, paired-answer
analysis, and the signoff log rather than citing this prose alone.
