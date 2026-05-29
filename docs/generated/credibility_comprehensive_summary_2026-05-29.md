# Comprehensive Credibility Battery Summary - 2026-05-29

No `paper/` files were edited.

## Phase Verdicts

| Phase | Status | Main read |
|---|---|---|
| A++ full independent judges | **blocked/provisional** | OpenRouter key quota is exhausted (`limit_remaining=0`), so full-N Claude/GPT/Qwen judging cannot run yet. The earlier q200 independent GPT-4o cache remains supportive but not full-N/two-judge. |
| C++ full-corpus BM25 + multi-retriever | **partly closed** | Full-corpus Tantivy BM25 completed for all seven datasets. SCOPE gold-affinity-delta correlation survives under BM25: mean SCOPE Spearman `0.354`, close to original gte+CE `0.342`. The requested E5/BGE third-dense retriever is still pending, so the three-retriever claim is not fully closed. |
| D no-gold OOD QPP predictor | **closed as useful negative** | Held-out-generator Kendall tau is only `0.090`; held-out-dataset tau is `0.052`; calibration up to 1000 held-out-generator labels does not approach the `|tau| >= 0.5` reliability bar. |

## What Is Closed

- Full-corpus BM25 replication is no longer a sample-only result. It ran on BarExamQA, HousingQA state-filtered, SciFact, NFCorpus, FiQA, TREC-COVID, and SciDocs.
- The mechanism is not obviously tied to the original gte+CE retrieval stack. Under full-corpus BM25, SCOPE gold-affinity delta still predicts retrieval gain with mean Spearman `0.354` across the seven datasets.
- The no-gold selective routing story should be framed as a negative result for now. The learned QPP router is far below the Datta-style reliability bar out of distribution, even with labeled calibration.

## What Remains

- A++ still needs two full-N independent judge passes once provider quota is available. The current hard blocker is external key limit, not analysis code.
- C++ still needs the third dense retriever (`intfloat/e5-large-v2` or BGE) over full corpora before claiming three-retriever generality. The E5 model is cached locally, but full E5 retrieval rows were not generated in this pass.
- The exact requested `5 train / 2 held-out datasets` OOD split is impossible in the current four-generator cache because the four-generator breadth exists only for five BEIR datasets. The report uses leave-one-dataset-out plus leave-two-datasets-out proxies.

## Claim Guidance

Strong supported language:

- The retrieval mechanism survives a full-corpus sparse-retrieval replication: SCOPE's movement toward gold evidence predicts retrieval gains under both the original dense+CE stack and BM25.
- Per-query no-gold routing is not ready as a reliable deployment gate; the available QPP signals are weak out of distribution.

Do not claim yet:

- Do not claim full-N two-independent-judge factuality falsification.
- Do not claim three-retriever generality until E5/BGE retrieval is completed.
- Do not claim a deployable QPP router; the current result is a useful negative.

## Artifacts

- A++ report: `docs/generated/credibility_A_full_independent_judges_2026-05-29.md`
- C++ report: `docs/generated/credibility_C_three_retrievers_full_2026-05-29.md`
- C++ row points: `docs/generated/credibility_C_three_retrievers_full_2026-05-29_points.jsonl`
- D report: `docs/generated/credibility_D_ood_predictor_2026-05-29.md`
- D row points: `docs/generated/credibility_D_ood_predictor_2026-05-29_points.jsonl`
- Shared script: `scripts/analyze_credibility_comprehensive.py`

## Commits

- `f01ff17` - A++ provider-quota report
- `75dabfc` - C++ full-corpus BM25 report
- `aaf1c1a` - D OOD QPP predictor report

