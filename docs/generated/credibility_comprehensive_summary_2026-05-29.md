# Comprehensive Credibility Battery Summary - 2026-05-29

No `paper/` files were edited.

## Phase Verdicts

| Phase | Status | Main read |
|---|---|---|
| A++ full independent judges | **single-judge GPT-4o landed; Claude pending** | GPT-4o re-judging now covers full BarExamQA, full SciFact/NFCorpus/FiQA/TREC-COVID, and a budget-truncated SciDocs q520 slice. The single-judge verdict supports Phase A: pooled factuality AUC `0.548`, geometry AUC `0.823`, joint AUC `0.826`, marginal lift `+0.003`; q200 Gemma-vs-GPT-4o IRR is Spearman `0.681`, kappa@0.5 `0.614`. This is not yet the two-independent-judge closeout. |
| C++ full-corpus BM25 + multi-retriever | **closed** | Full-corpus Tantivy BM25 completed for all seven datasets. SCOPE gold-affinity-delta correlation survives under BM25: mean SCOPE Spearman `0.354`, close to original gte+CE `0.342`. E5-large-v2 now covers all seven datasets with mean SCOPE Spearman `0.387`, above the `>=0.3` close criterion. HousingQA state-filtered E5 is retrieval-positive for SCOPE (`+7.5pp` Hit@5 over raw), although HyDE is stronger there (`+11.0pp`). |
| D no-gold OOD QPP predictor | **closed as useful negative** | Held-out-generator Kendall tau is only `0.090`; held-out-dataset tau is `0.052`; calibration up to 1000 held-out-generator labels does not approach the `|tau| >= 0.5` reliability bar. |
| E mid-regime pool threshold | **closed as diagnostic** | Lower-k evaluation over existing caches fills the 20-30% raw-retrieval void. Raw+SCOPE pooling improves raw on strict mid-regime points: SciDocs Hit@1 and Housing state-filtered Hit@2/Hit@3. |

## What Is Closed

- Full-corpus BM25 replication is no longer a sample-only result. It ran on BarExamQA, HousingQA state-filtered, SciFact, NFCorpus, FiQA, TREC-COVID, and SciDocs.
- The first independent factuality re-judge is no longer quota-blocked. GPT-4o single-judge results meet the pre-stated survival gate: pooled retrieval-hurt factuality AUC is below `0.65`, and the marginal AUC lift after geometry is below `+0.03`.
- The mechanism is not obviously tied to the original gte+CE retrieval stack. Under full-corpus BM25, SCOPE gold-affinity delta still predicts retrieval gain with mean Spearman `0.354` across the seven datasets.
- The third-retriever test is closed on all seven E5 corpora. E5 SCOPE gold-affinity delta predicts E5 retrieval gain with mean Spearman `0.387`, which clears the requested `>=0.3` threshold. BarExamQA is positive under E5: raw Hit@5 `0.5%`, HyDE `11.2%`, SCOPE `11.7%`, with SCOPE Spearman `0.344`. HousingQA state-filtered is also SCOPE-positive under E5: raw `39.5%`, SCOPE `47.0%`, with SCOPE Spearman `0.454`; HyDE is higher at `50.5%`.
- The mid-regime gap is no longer empty. In the strict 20-30% raw band, raw+SCOPE pooling improves raw on SciDocs Hit@1 (`22.2% -> 23.2%`) and Housing state-filtered Hit@2/Hit@3 (`23.9% -> 25.9%`; `29.3% -> 32.3%`).
- The no-gold selective routing story should be framed as a negative result for now. The learned QPP router is far below the Datta-style reliability bar out of distribution, even with labeled calibration.

## What Remains

- A++ still needs the second independent Claude/Sonnet judge before claiming a two-judge factuality closeout. The GPT-4o wave spent `$21.96` and stopped before HousingQA and full SciDocs to respect the explicit `$25` task cap.
- The exact requested `5 train / 2 held-out datasets` OOD split is impossible in the current four-generator cache because the four-generator breadth exists only for five BEIR datasets. The report uses leave-one-dataset-out plus leave-two-datasets-out proxies.

## Claim Guidance

Strong supported language:

- The retrieval mechanism survives a full-corpus sparse-retrieval replication: SCOPE's movement toward gold evidence predicts retrieval gains under both the original dense+CE stack and BM25.
- A single independent GPT-4o factuality judge corroborates the Phase A geometry-over-factuality reading on the completed A++ wave.
- Across the full seven-dataset E5 pass, including BarExamQA and HousingQA state-filtered, the same mechanism signal remains positive under a separate dense retriever.
- Raw+SCOPE pooling is a useful risk-control fusion in mid/high raw regimes, but it should be presented as regime-dependent.
- Per-query no-gold routing is not ready as a reliable deployment gate; the available QPP signals are weak out of distribution.

Do not claim yet:

- Do not claim full-N two-independent-judge factuality falsification; Claude/Sonnet remains pending, and this GPT-4o wave is budget-truncated on SciDocs and excludes HousingQA.
- Do not claim SCOPE is the best E5 expansion on HousingQA; HyDE has higher E5 Hit@5 there.
- Do not claim a deployable QPP router; the current result is a useful negative.
- Do not claim raw+SCOPE pooling universally beats canonical SCOPE. It is weak relative to SCOPE on BarExamQA and CaseHOLD, where SCOPE alone supplies the main retrieval lift.

## Artifacts

- A++ report: `docs/generated/credibility_A_full_independent_judges_2026-05-29.md`
- A++ GPT-4o single-judge report: `docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31.md`
- A++ GPT-4o single-judge cache: `docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl`
- A++ GPT-4o single-judge row points: `docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31_points.jsonl`
- C++ report: `docs/generated/credibility_C_three_retrievers_full_2026-05-29.md`
- C++ row points: `docs/generated/credibility_C_three_retrievers_full_2026-05-29_points.jsonl`
- C++ E5 addendum: `docs/generated/credibility_C_e5_addendum_2026-05-29.md`
- C++ E5 row points: `docs/generated/credibility_C_e5_addendum_2026-05-29_points.jsonl`
- D report: `docs/generated/credibility_D_ood_predictor_2026-05-29.md`
- D row points: `docs/generated/credibility_D_ood_predictor_2026-05-29_points.jsonl`
- E report: `docs/generated/credibility_E_midregime_2026-05-29.md`
- E row points: `docs/generated/credibility_E_midregime_2026-05-29_points.jsonl`
- Shared script: `scripts/analyze_credibility_comprehensive.py`

## Commits

- `f01ff17` - A++ provider-quota report
- `75dabfc` - C++ full-corpus BM25 report
- `aaf1c1a` - D OOD QPP predictor report
- `bc073e9` - C++ E5 retriever addendum
- `eb2742e` - E mid-regime analysis
- `919941f` - BarExamQA legal E5 retrieval
