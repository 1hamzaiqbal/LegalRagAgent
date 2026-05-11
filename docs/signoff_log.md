# Sign-off Log — verified results approved for paper/meeting citation

## Update 2026-05-11 ~meeting package

Change reason: consolidated the May 11 diagnostic-adaptation meeting package,
validated the repaired CaseHOLD direct option-table held-out run, and generated
source-gated figures for the inherited ablation/controller story.

Last updated: 2026-05-11
Branch: `codex/final-report-snap-hyde`

### Delta since 2026-05-10 diagnostic controller package

1. ✅ **Meeting package ready for the diagnostic-adaptation frame**:
   `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md` consolidates the
   four legal benchmarks, inherited calibration/held-out ablation tables,
   bottleneck summary, controller narrative, and generated figure pack. Use it
   as the first meeting entrypoint for the May 11 discussion.
2. ✅ **CaseHOLD direct option-table route is no longer implementation-blocked**:
   SLURM job `67744` completed with exit `0:0`. The repaired
   `adaptive_snap_hyre_option_table` route runs on the held-out CaseHOLD rows
   200-249 and passes `analyze_detail_flags.py` plus
   `audit_adaptive_hyre_logs.py`.
3. ⚠️ **CaseHOLD direct option-table is a clean negative design point**:
   `adaptive_snap_hyre_option_table` is 35/50 = 70.0% with 2.00 calls. It is a
   small lift over `rag_simple` on the same rows (34/50 = 68.0%; +2pp,
   b/c=2/1, p=1.0000), but below `rag_rewrite` (38/50 = 76.0%; -6pp,
   b/c=1/4, p=0.3750) and `adaptive_snap_hyre_diverse` (39/50 = 78.0%; -8pp,
   b/c=2/6, p=0.2891). Cite this as evidence that answer-option conversion
   remains a distinct bottleneck, not as a positive route.
4. ✅ **Figure pack generated from source-gated summaries**:
   `scripts/build_meeting_package_figures.py` emits figures 12-16 under
   `docs/presentation/figures/`. Figures 12-14 use the diagnostic controller
   portfolio and held-out query/controller JSON files; figures 15-16 are
   scripted diagrams derived from the meeting-prep source claims and linked
   result docs.
5. ⚠️ **Snap-only ladder controls are locally auditable across all four legal benchmarks**:
   BarExam `snap_only_in_final` job `67773` is 171/200 = 85.5% with 2.00 calls,
   errors 0, and one missing prediction. HousingQA `snap_only_in_final` job
   `67775` is 110/200 = 55.0% with 2.00 calls, errors 0, and one missing
   prediction. CaseHOLD `snap_only_in_final` job `67867` is 145/200 = 72.5%
   with 2.00 calls, errors 0, no missing predictions, and no long-answer rows;
   it supersedes health-caveated job `67777`. LegalBench-SCALR
   `snap_only_in_final` job `67779` is 145/200 = 72.5% with 2.00 calls, errors
   0, and no missing predictions. All four detail logs were copied from the
   cluster and summarized with `scripts/analyze_detail_flags.py`; empty
   retrieval payloads are expected for `snap_only_in_final`.
6. ✅ **Retrieval-bearing blocker is repaired enough for evaluated jobs**:
   `rag_utils.py` reinitializes the GTE remote-code `position_ids` buffer.
   Direct embedding smoke `67820` produced finite unit-norm query embeddings,
   and `rag_hyde` smoke `67821` completed 5/5. The N=200 HyRE-only and fixed
   Snap-HyRE rows below have since landed under the normal gates; SCALR
   HyRE-only uses capped rerun `67864` with a postprocess-wrapper caveat.
7. ✅ **BarExam HyRE-only landed as a modest positive retrieval control**:
   `rag_hyde` job `67825` completed with exit `0:0` at 164/200 = 82.0%,
   average calls 2.00, errors 0, missing predictions 0, and empty retrieval 0.
   It improves over baseline retrieval (80.0%) but trails snap-only reasoning
   (85.5%) and the stronger fixed Snap-HyRE v2 route (86.0%), so cite it as
   evidence for routing between rewrite/Snap-HyRE rather than generic HyRE-only.
8. ⚠️ **HousingQA HyRE-only landed as a negative control**:
   `rag_hyde` job `67826` completed with exit `0:0` at 100/200 = 50.0%,
   average calls 2.00, errors 0, missing predictions 0, and empty retrieval 0.
   Cite as evidence against generic HyRE retrieval for HousingQA; the stronger
   route remains state-filter retrieval plus verifier.
9. ⚠️ **CaseHOLD HyRE-only landed as a weak/negative retrieval control**:
   `rag_hyde` job `67827` completed with exit `0:0` at 143/200 = 71.5%,
   average calls 2.00, errors 0, one missing prediction, and empty retrieval 0.
   It trails the current N=200 `rag_simple` baseline (73.0%), snap-only
   control (72.5%), and diverse HyRE-family row (73.5%), so generic HyRE-only
   does not resolve the CaseHOLD answer-option conversion bottleneck.
10. ❌ **SCALR HyRE-only uncapped completed but is not a clean report row**:
   `rag_hyde` job `67828` completed with exit `0:0` at 142/200 = 71.0%, but
   `analyze_detail_flags.py` flags one runaway final answer with 267,458 chars
   and 70,593 output tokens. Do not cite this as a clean method result.
11. ⚠️ **SCALR capped HyRE-only landed as wrapper-caveated evidence**:
   `rag_hyde` job `67864` completed the eval loop at 148/200 = 74.0%, average
   calls 2.00, errors 0, missing predictions 1, empty retrieval 0, and no
   long-answer rows. The SLURM job state is `FAILED` because the wrapper tried
   to run missing `scripts/postprocess_adaptive_hyre_sweep.py` after writing the
   detail log. Cite as detail-log clean but wrapper-caveated; it matches the
   SCALR baseline and trails fixed Snap-HyRE/controller rows.
12. ⚠️ **HousingQA fixed Snap-HyRE landed as a negative control**:
   `rag_snap_hyde_2call` job `67830` completed with exit `0:0` at 103/200 =
   51.5%, average calls 2.00, errors 0, missing predictions 0, empty retrieval
   0, and no long-answer rows. It is below snap-only, state-filter retrieval,
   snap-HyRE state retrieval, and the verifier route.
13. ⚠️ **CaseHOLD fixed Snap-HyRE landed as weak/negative**:
   `rag_snap_hyde_2call` job `67831` completed with exit `0:0` at 144/200 =
   72.0%, average calls 2.00, errors 0, missing predictions 0, empty retrieval
   0, and no long-answer rows. It trails baseline retrieval, clean snap-only,
   and diverse HyRE-family rows.
14. ✅ **BarExam fixed Snap-HyRE landed**:
   `rag_snap_hyde_2call` job `67829` completed with exit `0:0` at 169/200 =
   84.5%, average calls 2.00, errors 0, one missing prediction, empty retrieval
   0, and no long-answer rows. It beats baseline retrieval (80.0%) and
   HyRE-only (82.0%), but trails snap-only (85.5%) and adaptive Snap-HyRE v2
   (86.0%).
15. ✅ **Groq Llama 70B held-out sanity mostly landed**:
   clean rows: BarExam `rag_simple` 38/50 = 76.0%, BarExam
   `adaptive_snap_hyre_v2` 36/50 = 72.0%, HousingQA `rag_state_filter` 22/50 =
   44.0%, HousingQA verifier 30/50 = 60.0%, CaseHOLD `rag_simple` 33/50 =
   66.0%, SCALR `rag_simple` 41/50 = 82.0%, and SCALR frontier 44/50 = 88.0%.
   CaseHOLD diverse HyRE is rejected as clean model-coverage evidence because
   it has errors 2, empty retrieval 2, and missing predictions 2.
16. ❌ **Full-SCALR probe `67863` is not a promoted result**:
   job `67863` wrote the full-SCALR `rag_simple` detail log at 424/571 =
   74.3%, average calls 1.00, errors 0, missing predictions 0, and empty
   retrieval 0, but `analyze_detail_flags.py` flags three long-answer rows
   with max 233,166 final-answer chars / 73,151 output tokens. Do not cite as a
   clean full-corpus baseline. The paired frontier half then produced a
   232,797-character answer at row 296 and was cancelled before writing a clean
   detail log. Capped replacement `67897` is running with
   `LLM_MAX_COMPLETION_TOKENS=4096`; do not cite it unless both modes finish and
   pass validation.
17. ✅ **CaseHOLD capped snap-only replacement `67867` is clean**:
   `67866` is rejected because it was cancelled at 71/200 after row 12 produced
   a 157,678-character answer and `pred=None`. After patching OpenRouter caps
   through `extra_body.max_tokens`, replacement `67867` completed with exit
   `0:0` at 145/200 = 72.5%, average calls 2.00, errors 0, missing predictions
   0, and no long-answer rows. Use `67867`, not health-caveated `67777`, for
   the snap-only ladder.
18. ✅ **Capped SCALR baseline half of `67897` is clean, paired frontier health-gated**:
   the completed `rag_simple` mode was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1218_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 419/571 = 73.4%, average
   calls 1.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 4,405, and no long-answer rows. Cite as a verified baseline-half log
   only; the paired frontier half is health-gated below.
19. ✅ **CaseHOLD N=500 baseline mode of `67913` is clean, paired modes pending**:
   the completed `rag_simple` mode was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1334_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 359/500 = 71.8%, average
   calls 1.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 2,725, and no long-answer rows. Cite only as a verified
   baseline-mode log until `rag_rewrite` and `adaptive_snap_hyre_diverse`
   finish and validate.
20. ❌ **Capped SCALR frontier half of `67897` completed but is health-gated**:
   the paired `adaptive_snap_hyre_frontier` detail log was copied locally from
   `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl`
   and reaches 417/571 = 73.0%, average calls 2.00, errors 0, parse failures 0,
   and empty retrieval 0. It is not a clean report row: one row has no
   predicted answer, `scripts/audit_adaptive_hyre_logs.py` fails with
   `missing_prediction=1`, and `scripts/analyze_detail_flags.py` flags one
   long-answer row with max final-answer chars 20,480 / max output tokens
   8,454. Do not promote `67897` as a clean paired full-SCALR result.
21. ✅ **SCALR N=571 query-rewrite retry `67915` completed cleanly**:
   `rag_rewrite` was copied locally from
   `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1542_legalbench_scalr_meeting-n500-canonical-r2-or-gemma4-26b-legalbench_scalr-n571-k5-rag_rewrite_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 423/571 = 74.1%, average
   calls 2.00, errors 0, missing predictions 0, empty retrieval 0, max output
   tokens 4,005, and no long-answer rows. This is the clean N>=500 SCALR
   rewrite control after invalid CUDA/ECC attempt `67914`.
22. ⚠️ **BarExam N=500 baseline mode of `67911` completed with one missing prediction**:
   `rag_simple` was copied locally from
   `logs/eval_rag_simple_or-gemma4-26b_20260511_1538_barexam_meeting-n500-canonical-or-gemma4-26b-barexam-n500-k5-rag_simple_detail.jsonl`
   and passes `scripts/analyze_detail_flags.py` at 400/500 = 80.0%, average
   calls 1.00, errors 0, empty retrieval 0, max output tokens 2,260, and no
   long-answer rows, but has one missing prediction. Cite as a verified
   baseline-mode log with that caveat until `rag_rewrite` and
   `adaptive_snap_hyre_v2` finish and validate.

### New source paths

- Meeting package: `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`.
- Package audit: `docs/meeting_package_audit_2026-05-11.md`.
- Snap-only summary: `docs/snap_only_controls_2026-05-11.json`.
- CaseHOLD direct option-table result:
  `docs/casehold_option_table_direct_heldout_2026-05-11.md`.
- Figure generator: `scripts/build_meeting_package_figures.py`.
- Figure outputs:
  `docs/presentation/figures/12_diagnostic_adaptation_calibration_ablation.png`,
  `docs/presentation/figures/13_diagnostic_adaptation_heldout_ablation.png`,
  `docs/presentation/figures/14_diagnostic_controller_macro_lift.png`,
  `docs/presentation/figures/15_bottleneck_diagnostic_route_map.png`,
  `docs/presentation/figures/16_method_ladder_flowchart.png`.
- Snap-only detail logs:
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0346_barexam_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0259_housing_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0943_casehold_meeting-capped-snap-casehold-v2-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`,
  `logs/eval_snap_only_in_final_or-gemma4-26b_20260511_0411_legalbench_scalr_meeting-missing-ladder-retry-or-gemma4-26b-n200-k5-snap_only_in_final_detail.jsonl`.
- BarExam HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0526_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- HousingQA HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0443_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- CaseHOLD HyRE-only detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0511_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- SCALR HyRE-only rejected detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0559_legalbench_scalr_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_hyde_detail.jsonl`.
- SCALR HyRE-only capped detail log:
  `logs/eval_rag_hyde_or-gemma4-26b_20260511_0734_detail.jsonl`.
- Full-SCALR `rag_simple` health-gated detail log:
  `logs/eval_rag_simple_or-gemma4-26b_20260511_0731_legalbench_scalr_meeting-full-scalr-sanity-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`.
- Full-SCALR cancelled stdout:
  `logs/slurm_67863_full_scalr_cancelled.out`.
- HousingQA fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0559_housing_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- CaseHOLD fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0602_casehold_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- BarExam fixed Snap-HyRE detail log:
  `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260511_0626_barexam_meeting-missing-retrieval-fixed-or-gemma4-26b-n200-k5-rag_snap_hyde_2call_detail.jsonl`.
- Groq held-out sanity detail logs:
  `logs/eval_rag_simple_groq-llama70b_20260511_0604_barexam_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_v2_groq-llama70b_20260511_0605_barexam_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_v2_detail.jsonl`,
  `logs/eval_rag_state_filter_groq-llama70b_20260511_0622_housing_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_state_filter_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_housing_verifier_groq-llama70b_20260511_0624_housing_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_housing_verifier_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260511_0610_casehold_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_diverse_groq-llama70b_20260511_0617_casehold_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_diverse_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260511_0622_legalbench_scalr_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-rag_simple_detail.jsonl`,
  `logs/eval_adaptive_snap_hyre_frontier_groq-llama70b_20260511_0626_legalbench_scalr_meeting-groq-heldout-fixed-groq-llama70b-q250-start200-end250-k5-adaptive_snap_hyre_frontier_detail.jsonl`.
- N>=500 scale-up baseline-mode detail logs:
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1218_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-rag_simple_detail.jsonl`,
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1538_barexam_meeting-n500-canonical-or-gemma4-26b-barexam-n500-k5-rag_simple_detail.jsonl`,
  `logs/eval_rag_simple_or-gemma4-26b_20260511_1334_casehold_meeting-n500-canonical-or-gemma4-26b-casehold-n500-k5-rag_simple_detail.jsonl`.
- N>=500 clean rewrite detail logs:
  `logs/eval_rag_rewrite_or-gemma4-26b_20260511_1542_legalbench_scalr_meeting-n500-canonical-r2-or-gemma4-26b-legalbench_scalr-n571-k5-rag_rewrite_detail.jsonl`.
- N>=500 rejected/health-gated adaptive detail logs:
  `logs/eval_adaptive_snap_hyre_frontier_or-gemma4-26b_20260511_1513_legalbench_scalr_meeting-full-scalr-capped-or-gemma4-26b-n571-k5-adaptive_snap_hyre_frontier_detail.jsonl`.

## Update 2026-05-01 ~meeting prep

Change reason: pulled completed cluster jobs `58282` and `58283`, fixed the
Housing state-filter metadata casing bug, and consolidated the meeting story in
`docs/meeting_state_2026-05-01.md`.

Last updated: 2026-05-01
Branch: `codex/final-report-snap-hyde`

### Delta since 2026-04-30 overlay

1. ✅ **CaseHOLD repaired pair landed**: after rebuilding `casehold_holdings`,
   `rag_simple` is 69.5% and `rag_snap_hyde_2call` is 72.0%, +2.5pp,
   b/c=16/11, McNemar p=0.4421, 95% bootstrap CI [-2.5, +7.5] pp. Gold
   retrieval is now meaningful for this pair: 16.0% -> 47.0%. Cite as
   "better gold retrieval does not yet translate into reliable answer lift,"
   not as a positive method result.
2. ⚠️ **CaseHOLD repaired top-k and HyDE follow-ups landed**: k=1 is 64.5%,
   k=5 is 69.5%, k=10 is 68.0%, and `rag_hyde` is 72.0%. The k=1 -> k=5
   depth delta is +5.0pp with McNemar p=0.0525; k=5 -> k=10 is flat/negative
   (-1.5pp, p=0.6072). Cite as a diagnostic option-conversion bottleneck,
   not as a reliable answer lift.
3. ❌ **Housing state-filter job `58282` is invalid as a method result**:
   both k=5 and k=10 runs were tagged `_FAILED-EMPTY-RETRIEVAL`, with 200/200
   rows having no retrieved evidence. The logged accuracies, 53.5% and 55.0%,
   are parametric behavior and must not be cited as state-filtered retrieval.
4. ✅ **Housing state-filter blocker unblocked for N=200 diagnostics**:
   `_housing_state_where(...)` now lowercases question states to match the
   lowercase statute metadata in `datasets/housing_qa/statutes.csv`. The fixed
   cluster run `58799` landed a clean k=5 row: 123/200 (61.5%), 0/200 empty
   retrieval, 81/200 gold retrieved. It beats generic top-5 by +8.0pp
   (b/c=36/20, p=0.0440) and is directionally above generic top-10 by +3.5pp
   (p=0.4350). The chunked k=10 rerun `58937` landed cleanly at 125/200
   (62.5%), 0/200 empty retrieval, 98/200 gold retrieved. It beats generic
   top-5 by +9.0pp (b/c=34/16, p=0.0153), is directionally above generic top-10
   by +4.5pp (p=0.3057), and is only +1.0pp above state-filter k=5 (p=0.8145).
   Cite as metadata-filtering signal, not as deeper-is-always-better.
5. 🧭 **Meeting framing**: top-k sensitivity should be called a cheap
   retrieval-policy stress test or first-pass bottleneck signal. It directly
   probes retrieval-depth/candidate-set sensitivity; query formulation,
   evidence use, metadata filtering, and option anchoring require the broader
   diagnostic matrix.

### New source paths

- Meeting synthesis: `docs/meeting_state_2026-05-01.md`.
- CaseHOLD repaired rerun: `docs/casehold_repaired_rerun_2026-05-01.md`.
- Housing state-filter followup: `docs/housing_state_filter_followup_2026-05-01.md`.
- Pulled detail logs:
  `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`,
  `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl`,
  `logs/eval_rag_simple_groq-llama70b_20260501_1440_detail.jsonl`,
  `logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1649_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260430_1720_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260501_1406_detail.jsonl`,
  `logs/eval_rag_state_filter_or-gemma4-26b_20260501_k10_merged_detail.jsonl`.

## Update 2026-04-30 ~15:30 CDT

Change reason: second adversarial report pass found that the class report had
outpaced this signoff layer. This update is a source-gated overlay for the
2026-04-28/30 bottleneck-taxonomy pivot; older sections are preserved below for
traceability.

Last updated: 2026-04-30 ~15:30 CDT
Branch: `codex/evidence-ledger-router`, HEAD after report refresh: `b59cb62`

### Current quick reference

1. ✅ **MuSiQue Llama 70B `snap_hyde_2call` is the current N=200 method
   vehicle**: `rag_simple` 27.5% -> `snap_hyde_2call` 37.0%, +9.5pp,
   b/c=33/14, McNemar p=0.007943, 95% bootstrap CI [+3.0, +16.0] pp.
   Cite as paired N=200, not full-corpus.
2. ✅ **MuSiQue top-k collapse is the cleanest retrieval-depth diagnostic**:
   `rag_simple` top-5 27.5% -> top-1 13.0%, -14.5pp, b/c=3/32,
   p=4.177e-07, CI [-20.0, -9.5] pp.
3. ✅ **BarExam top-k is depth-flat on the N=200 diagnostic slice**:
   `rag_simple` top-5 82.5% -> top-1 83.0%, +0.5pp, b/c=18/17, p=1.0.
   Keep separate from the full-corpus BarExam method result.
4. ✅ **LegalBench-SCALR is candidate-depth limited then saturated**:
   top-5 77.0% -> top-1 59.5%, -17.5pp, b/c=3/38, p=1.048e-08;
   top-10 ties top-5 at 77.0%, b/c=8/8, p=1.0.
5. ⚠️ **HousingQA is a directional statutory depth signal**:
   top-1 50.5% -> top-10 58.0%, +7.5pp, b/c=38/23, p=0.0722.
   Cite with state-metadata and low gold-hit caveats.
6. ⚠️ **CaseHOLD is answer-flat under current logs, not retrieval-recall
   evidence**: top-5 72.0% -> top-1 70.5%, -1.5pp, b/c=10/13, p=0.6776;
   two-call 69.5%, -2.5pp, b/c=14/19, p=0.4869. Old gold-hit is 0/200 due to
   instrumentation; wait for repaired Chroma rerun before retrieval claims.
7. ✅ **MuSiQue `golden_passage` control confirms context utility is
   task-dependent**: `golden_passage` reaches 56.5% EM, beating `rag_simple`
   by +29.0pp (b/c=64/6, p=2.44e-13) and `snap_hyde_2call` by +19.5pp
   (b/c=47/8, p=8.07e-08). This is a privileged-context diagnostic, not a
   deployable method.

### Current source paths

- Bottleneck matrix and paired deltas: `docs/evidence_matrix_2026-04-30.md`.
- Housing metrics and metadata caveats:
  `docs/housing_speculative_metrics_2026-04-30.md` and
  `docs/housing_metadata_depth_audit_2026-04-30.md`.
- CaseHOLD instrumentation caveat and repair:
  `docs/casehold_flatness_audit_2026-04-30.md` and
  `docs/casehold_gold_mapping_repair_2026-04-30.md`.
- SCALR depth audit: `docs/scalr_depth_disagreement_2026-04-30.md`.
- MuSiQue golden control: `docs/musique_golden_passage_2026-04-30.md`.
- Full-corpus BarExam method matrix remains Section A below.

## Update 2026-04-27 ~12:30 CDT

Change reason: added the 2026-04-27 ~12:30 CDT McNemar results for Llama planning methods and BarExam cross-domain mhd. That McNemar section gives paired statistics but no separate audit IDs, so new audit cells cite the 12:30 McNemar source rather than inventing IDs.

Last updated: 2026-04-27 ~12:30 CDT
Branch: hpc-setup, HEAD: a50f67a

This log lists results that have:
1. Landed cleanly (no preflight failure, no harness crash)
2. Passed per-entry confound audit (codex sampled records: no MAJOR truncation, leakage, fallback, empty-retrieval, or format issues)
3. Been reviewed by architect (Claude Opus) for paper-defensibility
4. Have a direct path to detail log + commit SHA + audit doc

**Sign-off levels:**
- ✅ **APPROVED** — cite freely, paper-grade
- ⚠️ **APPROVED-WITH-CAVEAT** — cite with the documented caveat
- ⏸ **PENDING** — landed but awaiting audit
- ❌ **REJECTED** — known confound, do not cite

## Quick reference: top 5 cite-able findings for the paper (Tier 2 MuSiQue = N=200 paired; full-corpus replicate pending)

1. ✅ BarExam snap+HyDE is the Tier 3 legal-MC winner: Gemma 4 26B-A4B 78.08% → 81.17% (+3.09pp) and Gemma 4 E4B 58.49% → 62.18% (+3.69pp). **Architecture note**: ~76-83% of `rag_snap_hyde` final preds match `snap_letter` (BY DESIGN architecture — the mode combines snap reasoning + HyDE retrieval; snap reasoning dominates because Gemma 4 has strong legal MC priors). HyDE provides marginal lift and sometimes conflicting evidence; when pred==snap, EM=88.7%, while pred≠snap is 45.7%. Frame this as mechanism understanding.
2. ✅ Llama 70b MuSiQue `multi_hyde_diverse` is the superseded pre-pivot Tier 2 N=200 paired multi-hop headline: 27.5% → 35.5%, +8pp, McNemar p=0.0195; *pending full-corpus replicate*. The current 2026-04-30/05-01 meeting vehicle is `snap_hyde_2call` plus the bottleneck taxonomy.
3. ⚠️ Llama 70b MuSiQue `iterative_planning_table` is cite-able as N=200 paired TRENDING-SIG, not fully significant: 27.5% → 36.0%, +8.5pp, p=0.0533; *pending full-corpus replicate*.
4. ✅ Gemma 3 27B MuSiQue mhd is a cite-able N=200 paired negative cross-family check: 28.5% → 31.0%, +2.5pp, p=0.5901 NULL; *full-corpus replicate would solidify*.
5. ⚠️ Llama 70b MuSiQue `subagent_rag` N=200 paired -12pp p=0.0007 SIG NEGATIVE. **Implementation caveat**: 200/200 records triggered gap-routing (100% rate; over-aggressive); 59/200=29.5% finals are "Unknown/Not found" vs 12.5–15% in other methods. With our gap-routing implementation, `subagent_rag` systematically over-abstains on multi-hop and produces a real -12pp finding; reframing the prompt could likely close part of this gap. Do not generalize beyond this implementation.

## Audit lineage (2026-04-27 ~14:30 CDT, comprehensive per-log Haiku audit)

Per-log audit reports under `docs/audits/`:
- `2026-04-27_barexam_26b_audit.md` — 8 logs × N=1195. Initial Haiku review raised a snap-dominance concern; architect-verified as BY DESIGN architecture (the mode combines snap+HyDE, snap dominates because Gemma 4 has strong legal priors, and HyDE acts as marginal evidence). 7/8 CLEAN, 1 architecture-clarified.
- `2026-04-27_barexam_e4b_audit.md` — 8 logs × N=1195. All ✅ CLEAN.
- `2026-04-27_llama70b_musique_audit.md` — 8 logs × N=200. All data CLEAN; subagent_rag flagged for implementation quirk (100% gap-routing trigger → over-abstention) — caveat documented in Top 5 #5.
- `2026-04-27_other_tier2_audit.md` — 6 logs × N=200. All ✅ CLEAN.

---

## Section A — Tier 3 / Full corpus

### A.1 BarExam Gemma 4 26B-A4B method matrix at N=1195

| Mode | EM | Audit | Sign-off | Caveat |
|---|---:|---|---|---|
| `rag_simple` | 78.08% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | 2/15 sampled records had null pred + empty retrieval; 933/1195 = 78.08% holds |
| `rag_snap_hyde` | 81.17% | MINOR | ✅ APPROVED | low BarExam exact-gold retrieval (generic to dataset) |
| `snap_only_in_final` | 80.59% | CLEAN | ✅ APPROVED | — |
| `rag_hyde` | 78.91% | MINOR | ✅ APPROVED | low BarExam exact-gold retrieval |
| `subagent_rag` | 78.16% | MINOR | ✅ APPROVED | 8 records empty retrieval in full scan; sample clean |
| `subagent_hybrid` | 74.23% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | rescore note: raw stored 74.14%, audit re-scored to 74.23% |
| `llm_only` | 79.75% | CLEAN | ✅ APPROVED | — |
| `golden_passage` | 78.66% | CLEAN | ✅ APPROVED | — |

**Source-of-truth**: `docs/audit_log.md` (post-fix re-scored from detail logs; experiments.jsonl rows are pre-fix and stale).
**Detail logs**: `logs/eval_*_cluster-vllm_2026042{5,6}_*_detail.jsonl` (see `docs/compiled_results.md` Section 1.2).
**Result commits**: `8bbf0e7` (audit), `ed15eb7` (extractor).

### A.2 BarExam Gemma 4 E4B method matrix at N=1195

| Mode | EM | Audit | Sign-off | Caveat |
|---|---:|---|---|---|
| `rag_simple` | 58.49% | MINOR | ⚠️ APPROVED-WITH-CAVEAT | low exact-gold retrieval; no sampled parser issue |
| `rag_hyde` | 60.59% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `rag_snap_hyde` | 62.18% | MINOR | ✅ APPROVED | one raw null parsed prediction in full scan; sample clean |
| `snap_hyde_report` | 60.75% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `snap_only_in_final` | 57.82% | CLEAN | ✅ APPROVED | — |
| `subagent_hybrid` | 58.83% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `subagent_hyde` | 60.17% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |
| `subagent_rag` | 60.92% | MINOR | ✅ APPROVED | low exact-gold retrieval (generic to dataset) |

**Detail logs**: `logs/eval_*_cluster-vllm_20260426_*_detail.jsonl` (E4B); see `docs/compiled_results.md` Section 1.3.

### A.3 BarExam cross-size headline (PAPER STORY)

**`rag_snap_hyde` lifts BarExam EM at both Gemma 4 sizes:**
- Gemma 4 26B-A4B: +3.09pp (78.08% → 81.17%), b/c=124/87, McNemar p=0.0130,
  95% bootstrap CI [+0.67, +5.44] pp.
- Gemma 4 E4B: +3.68pp (58.49% → 62.18%), b/c=172/128, McNemar p=0.0129,
  95% bootstrap CI [+0.92, +6.53] pp.

**Sign-off**: ✅ APPROVED (Tier 3, cross-size confirmed; both sizes have post-fix detail-log/audit support, with the caveats listed above).

---

## Section B — Tier 2 / N=200 paired McNemar

### B.1 Llama 70b MuSiQue method matrix (PAPER HEADLINE + TRENDING)

| Mode | EM | Δ | McNemar p | Audit | Sign-off |
|---|---:|---:|---:|---|---|
| `rag_simple` | 27.5% | — | — | N=200 paired; CLEAN | ✅ APPROVED (baseline) |
| **`iterative_planning_table`** | **36.0%** | **+8.5pp** | **0.0533** | N=200 paired; McNemar 12:30 | **✅ APPROVED — TRENDING-SIG (*pending full-corpus replicate*)** |
| **`multi_hyde_diverse`** | **35.5%** | **+8pp** | **0.0195** | N=200 paired; CLEAN | **✅ APPROVED — superseded pre-pivot headline (*pending full-corpus replicate*)** |
| `rag_multi_query` | 29.0% | +1.5pp | 0.728 | N=200 paired; CLEAN | ✅ APPROVED (mechanism decomposition) |
| `rag_snap_hyde` | 24.0% | -3.5pp | 0.36 | N=200 paired; CLEAN | ✅ APPROVED (cross-domain neg evidence) |
| `iter_hyde` | 24.5% | -3.0pp | 0.47 | N=200 paired; CLEAN | ✅ APPROVED (multi-round neutral at large) |
| `advisor_planning_table` | 23.0% | -4.5pp | 0.222 | N=200 paired; McNemar 12:30 | ✅ APPROVED — NS but informative negative |
| **`subagent_rag`** | **15.5%** | **-12.0pp** | **0.0007** | N=200 paired; CLEAN | **✅ APPROVED — sig negative** |

**Detail logs**: `logs/eval_*_groq-llama70b_20260427_{0952,1010,1019,1036,1044,1112,1208,1216}_detail.jsonl`.
**Source-of-truth**: `docs/mcnemar_2026-04-27.md`.

### B.2 Mechanism decomposition (Llama 70b N=200 paired)

**mhd's +8pp lift decomposes into:**
- HyDE-style answer-bearing passages: ~6.5pp (mhd minus multi_query)
- Query diversity alone: +1.5pp NS (multi_query alone)

**Sign-off**: ✅ APPROVED (Tier 2 N=200 paired; HyDE-style is dominant ~80% contributor; *full-corpus replicate would solidify*).

### B.3 Cross-family negative finding (N=200 paired)

**mhd × Gemma 3 27B N=200 paired = 31.0%, +2.5pp, p=0.5901 NULL**

**Sign-off**: ✅ APPROVED (negative finding) — Tier 2 N=200 paired NULL on Gemma 3 27B; the cross-family lift on dense models is NOT universal; *full-corpus replicate would solidify*.

### B.4 BarExam cross-domain (paired N=200)

| Method / model | Comparator | Result | McNemar p | Sign-off |
|---|---|---:|---:|---|
| `multi_hyde_diverse` × Gemma 4 26B-A4B | N=200 paired first-200 `rag_simple` = 84.5% | 82.0%, -2.5pp | 0.499 | ⏸ SOURCE-PENDING — do not cite as landed |

**Source status**: source-pending in `docs/mcnemar_2026-04-27.md`; keep provisional until the SLURM 55107 detail log lands locally.

---

## Section C — Tier 1 / direction-only (NOT paper-grade alone)

### C.1 Friend/foe attribution-bias probe

| Model | N | Outcome changes | Audit | Sign-off |
|---|---:|---:|---|---|
| Gemma 3 27B | 30 | 4/30 = 13.3% | CLEAN | ⚠️ APPROVED-WITH-CAVEAT (N=30 directional only) |
| Llama 70b | 30 | 6/30 = 20.0% | CLEAN | ⚠️ APPROVED-WITH-CAVEAT (N=30 directional only) |

**Sign-off**: ⚠️ APPROVED-WITH-CAVEAT — cite as "real mechanism detected but limited effect size at N=30". For paper claim, scale to N=100+.

### C.2 iter_hyde × small-model negative direction

| Model | iter_hyde EM | rag_simple comparator | Δ | Sign-off |
|---|---:|---:|---:|---|
| Gemma 3 27B (N=30) | 6.7% | 22% (N=100) | -15pp | ⚠️ DIRECTION-ONLY |
| Llama 4 Scout (N=30) | 16.7% | 30% (N=100) | -13pp | ⚠️ DIRECTION-ONLY |
| Qwen3 30B MoE (N=30) | 6.7% | 24% (N=100) | -17pp | ⚠️ DIRECTION-ONLY |

**Sign-off**: ⚠️ DIRECTION-ONLY (N=30 small samples; direction is consistent but cite as "trend not test").

### C.3 Llama 70b iter_hyde Tier 2 (lift to APPROVED)

iter_hyde × Llama 70b N=200 = -3pp p=0.47 NS (audit CLEAN).

**Sign-off**: ✅ APPROVED — multi-round HyDE doesn't help large dense (statistically null).

---

## Section D — In flight (will sign off when landed + audited)

| Run | Status | Spot-check verdict | Expected sign-off |
|---|---|---|---|
| SLURM 55107 BarExam mhd+iter_hyde × Gemma 4 26B-A4B N=200 | still running; operator snapshot says mhd 82% done and iter_hyde at q106+/200 (78.3% partial PASS rate) | LEGIT by operator spot-check, but source detail log not present locally | Expected ✅ APPROVED only after landing + source log |
| `qwen_full` mhd-pair × Qwen3 30B MoE × N=2400 MuSiQue | RUNNING ~q1058/2400 (rag_simple = 26.1%, slow but progressing) | LEGIT by operator spot-check, but source log not present locally | Tier 2.5 partial only until full run + audit land |

### Section D — KILLED jobs (cannot be relied on as Tier 2/3 results)

| Run | Status at kill | Reason killed | Citation guidance |
|---|---|---|---|
| `gemma4_full` mhd-pair × Gemma 4 26B-A4B × N=2400 MuSiQue (or-gemma4-26b API) | KILLED 2026-04-27 14:00 CDT at q431/2400 (rag_simple partial = 30.9%) | Hung 73+ min on q432 due to OR-served Gemma 4 26B runaway-loop generation (one 91k-char looped answer at q431 took 601s; subsequent query never returned) | ⚠️ Tier 2.5 partial — citeable ONLY as "Gemma 4 26B-A4B `rag_simple` MuSiQue N=431 = 30.9% (partial, OR-Gemma serving cut short by runaway loops)". Do NOT cite as Tier 3. |
| `iterative_planning_table` × Gemma 27B N=200 (or-gemma27b) | KILLED 2026-04-27 14:00 CDT at q29/200 | Same OR-Gemma issue — one query took 2405s = 40 min. Projected ETA was 10+ hours. | ❌ DO NOT CITE — N=29 is below Tier 0 threshold and contains a 40-min outlier. |

### Section D' — OR-served Gemma serving issue (methodology finding)

**Discovered 2026-04-27 ~13:00 CDT:** OpenRouter-served Gemma models (Gemma 4 26B-A4B and Gemma 3 27B) exhibit pathological **runaway-loop generations** on iterative or multi-step prompts. Symptoms:
- Single queries occasionally take 600s, 1200s, 2400s instead of normal 5-30s
- Answer text contains repetitive looping (e.g., "Lou Boudreau (no), it is Lou Boudreau (no)..." for 91k chars)
- ~2% of `rag_simple` (single-call) MuSiQue queries echoed `[your answer here]` placeholder and looped
- Effect compounds in iterative/multi-call modes: `iter_planning_table × Gemma 27B` was projected to take 10+ hours for N=200

**Mitigation**: Use cluster vLLM (Gemma 4 26B-A4B served locally via vLLM nightly + transformers 5.5.0) instead of OR API for Gemma. SLURM 55107 confirms cluster vLLM is clean — same model, no leakage, normal latencies.

**Implication for meeting**: Whenever cited Gemma results were collected via OR, prefer cluster vLLM equivalents. The `gemma4_full` partial result (q431=30.9% rag_simple Gemma 4 26B MuSiQue) should be treated as a noisy lower-bound, not a Tier 2/3 cite-able number.

---

## Section E — Sign-off process

1. Run lands cleanly → enters PENDING
2. Codex per-entry audit (sample 5-10 records) → CLEAN / MINOR / MAJOR
3. Architect reviews audit + cross-checks sources → ✅ APPROVED / ⚠️ APPROVED-WITH-CAVEAT / ❌ REJECTED
4. Entry added here with date/time + commit SHA + paths
5. Compiled_results.md is the detailed reference; this log is the cite-or-not gate

**Architect**: Claude Opus 4.7 (1M context), this session.
**Audit principal**: codex CLI 0.126.0-alpha.4 with `~/.codex/config.toml` defaults.

## Section F — Historical N≥200 runs retroactively audited

(Audited 2026-04-27 ~12:00 CDT, 3-record spot-check per row)

| Tag | Mode | Provider | N | EM | T? | E? | Th? | ER? | Sign-off |
|---|---|---|---|---|---|---|---|---|---|
| `captain-llama70b-musique-mhd-n200` | `multi_hyde_diverse` | `groq-llama70b` | 200 | 35.5% | N | N | N | N | ✅ APPROVED |
| `mhd-pair-gemma27b-n200-power` | `multi_hyde_diverse` | `or-gemma27b` | 200 | 31.0% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-snap-hyde-n200` | `rag_snap_hyde` | `groq-llama70b` | 200 | 24.0% | N | N | N | N | ✅ APPROVED |
| `26b-seed99-repeat` | `rag_snap_hyde` | `custom` | 1195 | 75.4% | N | N | N | N | ✅ APPROVED |
| `e4b-n200-postfix-v2` | `rag_snap_hyde` | `custom` | 200 | 67.5% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `rag-multi-query-scout-n200` | `rag_multi_query` | `groq-scout` | 200 | 30.5% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-multi-query-n200` | `rag_multi_query` | `groq-llama70b` | 200 | 29.0% | N | N | N | N | ✅ APPROVED |
| `rag-multi-query-gemma27b-n200` | `rag_multi_query` | `or-gemma27b` | 200 | 28.5% | N | N | N | N | ✅ APPROVED |
| `rag-simple-scout-n200` | `rag_simple` | `groq-scout` | 200 | 30.0% | N | N | N | N | ✅ APPROVED |
| `mhd-pair-gemma27b-n200-power` | `rag_simple` | `or-gemma27b` | 200 | 28.5% | N | N | N | N | ✅ APPROVED |
| `captain-llama70b-musique-rag-simple-n200` | `rag_simple` | `groq-llama70b` | 200 | 27.5% | N | N | N | N | ✅ APPROVED |
| `e4b-seed99-repeat` | `rag_simple` | `custom` | 1195 | 55.7% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `e4b-n200-prompt-fix` | `rag_simple` | `custom` | 200 | 61.5% | N/A | N/A | N/A | N/A | ⏸ PENDING |
| `e4b-n200-postfix-v2` | `rag_simple` | `custom` | 200 | 61.0% | N | N | N | N | ✅ APPROVED |
| `e4b-n200-postfix-v2` | `rag_hyde` | `custom` | 200 | 61.5% | N | N | N | N | ✅ APPROVED |

T? = Truncation, E? = Empty pred, Th? = <think> leak, ER? = Empty retrieval

## Section G — Historical runs INVALIDATED (do not cite)
- Pre-fix BarExam (timestamps before 2026-04-22): `26b-seed99-repeat` (2026-04-21T21:15:16Z, `rag_simple`); `26b-baseline-ceiling` (2026-04-21T21:58:57Z, `golden_passage`); `31b-full-matrix` (2026-04-21T22:09:55Z, `rag_hyde`); `26b-subagent-1` (2026-04-21T22:26:13Z, `subagent_rag`); `26b-subagent-2` (2026-04-21T22:30:08Z, `subagent_hybrid`); `e2b-full-matrix-redo` (2026-04-21T22:58:01Z, `rag_hyde`); `26b-seed99-repeat` (2026-04-21T23:33:23Z, `rag_hyde`); `26b-full-matrix` (2026-04-21T23:39:52Z, `snap_only_in_final`)
- Empty-retrieval contaminated: `api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL` (2026-04-27T03:42:40Z, `advisor_planning_table`)
- Smoke/test runs: `api-smoke` (2026-04-26T04:44:05Z, `llm_only`); `api-musique-smoke2` (2026-04-26T04:54:25Z, `llm_only`); `api-musique-ptable-smoke` (2026-04-26T22:20:07Z, `planning_table`); `api-smoke-groq-qwen` (2026-04-27T00:23:27Z, `llm_only`); `api-smoke-groq-llama70b` (2026-04-27T00:23:27Z, `llm_only`)

---

## Section F — Historical runs (retroactively audited 2026-04-27)

Scope: top paper-relevant historical rows from `logs/experiments.jsonl`, excluding rows already covered in Sections A/B/C. For rows with detail logs, codex checked first 2 + middle 1 + last 2 records for truncation, empty predictions, `<think>` leakage, snap-letter echo, fallbacks, and empty retrieval; obvious full-log quality counters were also checked. Missing detail log means `PENDING`.

### F.1 BarExam Gemma 4 26B-A4B historical (post-fix era)

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `20260421_2149 / 26b-subagent-2` | `snap_hyde_report` | 1195 | 76.57% | detail log missing (`logs/eval_snap_hyde_report_cluster-vllm_20260421_2149_detail.jsonl`) | ⏸ PENDING |
| `20260421_2150 / 26b-subagent-1` | `subagent_hyde` | 1195 | 76.57% | detail log missing (`logs/eval_subagent_hyde_cluster-vllm_20260421_2150_detail.jsonl`) | ⏸ PENDING |
| `20260421_2234 / 26b-seed99-repeat` | `rag_snap_hyde` | 1195 | 75.40% | 5-row spot clean; no empty pred, no snap-stage echo, no fallback, no empty retrieval; superseded by Section A current 26B matrix | ⚠️ APPROVED-WITH-CAVEAT |

### F.2 BarExam Gemma 4 E4B historical

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `20260421_2000 / e4b-seed99-repeat` | `rag_simple` | 1195 | 55.73% | detail log missing (`logs/eval_rag_simple_cluster-vllm_20260421_2000_detail.jsonl`) | ⏸ PENDING |
| `20260421_2239 / e4b-n200-postfix-v2` | `rag_simple` | 200 | 61.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `20260421_2312 / e4b-n200-postfix-v2` | `rag_hyde` | 200 | 61.50% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `20260421_2331 / p1a-full-rerun` | `snap_only_in_final` | 1195 | 54.81% | detail log missing (`logs/eval_snap_only_in_final_cluster-vllm_20260421_2331_detail.jsonl`) | ⏸ PENDING |
| `20260422_0007 / e4b-n200-postfix-v2` | `rag_snap_hyde` | 200 | 67.50% | detail log missing (`logs/eval_rag_snap_hyde_cluster-vllm_20260422_0007_detail.jsonl`) | ⏸ PENDING |

### F.3 MuSiQue historical (Llama 70b, Gemma 27B, Scout, Qwen; N≥100 only)

| Tag | Mode | N | EM | Audit | Sign-off |
|---|---|---:|---:|---|---|
| `api-musique-rag-simple-llama-n100` | `rag_simple` / Llama 70b | 100 | 21.00% | 5-row spot clean; `audit_log.md` paired-advisor check re-scored 21/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-advisor-llama-n100` | `advisor_planning_table` / Llama 70b | 100 | 23.00% | 5-row spot clean; `audit_log.md` says CLEAN but not statistically significant vs rag_simple | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-mhd-llama-n100` | `multi_hyde_diverse` / Llama 70b | 100 | 33.00% | 5-row spot clean; `audit_log.md` cross-family N=100 audit confirmed 33/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-rag-simple-gemma27b-n100` | `rag_simple` / Gemma 3 27B | 100 | 22.00% | 5-row spot clean; `audit_log.md` confirmed 22/100 | ⚠️ APPROVED-WITH-CAVEAT |
| `api-musique-mhd-gemma27b-n100` | `multi_hyde_diverse` / Gemma 3 27B | 100 | 30.00% | 5-row spot clean; `audit_log.md` confirmed 30/100, p=0.134 trend vs rag_simple | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-scout-n100` | `rag_simple` / Scout | 100 | 30.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-scout-n100` | `multi_hyde_diverse` / Scout | 100 | 29.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-snap-hyde-llama-musique-n100` | `rag_snap_hyde` / Llama 70b | 100 | 21.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 echo/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-llama-musique-n100` | `rag_multi_query` / Llama 70b | 100 | 25.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-scout-musique-n100` | `rag_multi_query` / Scout | 100 | 25.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-multi-query-scout-n200` | `rag_multi_query` / Scout | 200 | 30.50% | sample clean, but full log has 1 placeholder-echo prediction (`[your answer here]`) counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-qwen-n100` | `rag_simple` / Qwen3 30B MoE | 100 | 24.00% | sample clean, but full log has 1 blank final answer / empty prediction counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `mhd-pair-qwen-n100` | `multi_hyde_diverse` / Qwen3 30B MoE | 100 | 28.00% | sample clean, but full log has 1 generate-empty error and 2 empty predictions counted wrong | ⚠️ APPROVED-WITH-CAVEAT |
| `rag-simple-scout-n200` | `rag_simple` / Scout | 200 | 30.00% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |
| `rag-multi-query-gemma27b-n200` | `rag_multi_query` / Gemma 3 27B | 200 | 28.50% | 5-row spot clean; 0 errors, 0 empty preds, 0 leakage/fallback/empty retrieval | ✅ APPROVED |

### F.4 BarExam other models (Qwen3 30B MoE, Llama 70b, Scout, Gemma 27B)

No BarExam Llama 70b N≥200 row was found in the Apr. 20-26 historical slice; the clean N=100 cross-family API rows are signed below as support-only results.

| Tag | Model | Mode | N | EM | Audit | Sign-off |
|---|---|---|---:|---:|---|---|
| `api-cross-scout-n100` | Llama 4 Scout 17B | `llm_only` | 100 | 67.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-llama70b-n100` | Llama 3.3 70B | `llm_only` | 100 | 81.00% | 5-row spot clean; `audit_log.md` cross-family check says CLEAN | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-qwen3-32b-n100` | Qwen3 32B dense | `llm_only` | 100 | 68.00% | `audit_log.md` found 13/100 truncated mid-`<think>` with `predicted_answer=None`; sample reproduced 1 empty pred | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-gemma3-27b` | Gemma 3 27B | `llm_only` | 100 | 68.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |
| `api-cross-qwen3-30b-moe-n100` | Qwen3 30B MoE | `llm_only` | 100 | 70.00% | 5-row spot clean; 0 errors, 0 empty preds | ⚠️ APPROVED-WITH-CAVEAT |

## Section G — Historical runs INVALIDATED (do not cite)

### G.1 Pre-fix BarExam rows (formatter/retrieval-query bug window)

Current BarExam citations must use the post-fix source-of-truth values in `docs/audit_log.md` / Sections A and F. These `logs/experiments.jsonl` rows are retained only as historical references because they landed before the `3d5ff05` retrieval-query fix or in the immediate pre-2026-04-22 bug window:

- `20260420_2349_rag_snap_hyde_cluster-vllm_leak-fix-validation` (`leak-fix-validation`, N=30)
- `20260421_0055_rag_simple_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0203_rag_hyde_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0359_rag_snap_hyde_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0405_rag_simple_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0458_rag_hyde_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0526_snap_only_in_final_cluster-vllm_mini-eval-leak-fix` (`mini-eval-leak-fix`, N=200)
- `20260421_0632_rag_snap_hyde_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0746_snap_only_in_final_cluster-vllm_31b-n200-matrix` (`31b-n200-matrix`, N=200)
- `20260421_0802_rag_simple_cluster-vllm_e2b-full-matrix` (`e2b-full-matrix`, N=1195)
- `20260421_0812_rag_simple_cluster-vllm_p1a-full-rerun` (`p1a-full-rerun`, N=1195)
- `20260421_0857_rag_simple_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1112_rag_hyde_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1203_rag_simple_cluster-vllm_31b-full-matrix` (`31b-full-matrix`, N=1195)
- `20260421_1402_rag_snap_hyde_cluster-vllm_p1b-full-rerun` (`p1b-full-rerun`, N=1195)
- `20260421_1449_rag_hyde_cluster-vllm_p1a-full-rerun` (`p1a-full-rerun`, N=1195)
- `20260421_1501_llm_only_cluster-vllm_26b-baseline-ceiling` (`26b-baseline-ceiling`, N=1195)
- `20260421_1515_rag_snap_hyde_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1615_rag_simple_cluster-vllm_26b-seed99-repeat` (`26b-seed99-repeat`, N=1195)
- `20260421_1658_golden_passage_cluster-vllm_26b-baseline-ceiling` (`26b-baseline-ceiling`, N=1195)
- `20260421_1709_rag_hyde_cluster-vllm_31b-full-matrix` (`31b-full-matrix`, N=1195)
- `20260421_1726_subagent_rag_cluster-vllm_26b-subagent-1` (`26b-subagent-1`, N=1195)
- `20260421_1730_subagent_hybrid_cluster-vllm_26b-subagent-2` (`26b-subagent-2`, N=1195)
- `20260421_1758_rag_hyde_cluster-vllm_e2b-full-matrix-redo` (`e2b-full-matrix-redo`, N=1195)
- `20260421_1833_rag_hyde_cluster-vllm_26b-seed99-repeat` (`26b-seed99-repeat`, N=1195)
- `20260421_1839_snap_only_in_final_cluster-vllm_26b-full-matrix` (`26b-full-matrix`, N=1195)
- `20260421_1902_rag_simple_cluster-vllm_e4b-n200-prompt-fix` (`e4b-n200-prompt-fix`, N=200)

### G.2 Empty-retrieval contaminated runs from local Mac

- `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50` (`api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL`, N=50) — `legal_passages` collection was empty locally; 50/50 rows had empty retrieval.

### G.3 Smoke / test runs

- `20260421_0229_rag_hyde_cluster-vllm_smoke-31b`
- `20260425_2344_llm_only_or-gemma4-26b_api-smoke`
- `20260425_2354_llm_only_or-gemma4-26b_api-musique-smoke2`
- `20260426_1720_planning_table_or-gemma4-26b_api-musique-ptable-smoke`
- `20260426_1923_llm_only_groq-qwen_api-smoke-groq-qwen`
- `20260426_1923_llm_only_groq-llama70b_api-smoke-groq-llama70b`
- `20260426_1923_llm_only_groq-kimi_api-smoke-groq-kimi`
- `20260426_1923_llm_only_groq-scout_api-smoke-groq-scout`
- `20260426_1925_llm_only_groq-kimi_api-smoke-groq-kimi-v2`
- `20260426_1925_llm_only_groq-scout_api-smoke-groq-scout-v2`
- `20260426_1925_llm_only_groq-llama70b_api-smoke-groq-llama70b-v2`
- `20260426_1925_llm_only_groq-qwen_api-smoke-groq-qwen-v2`
- `20260426_1935_llm_only_groq-qwen_api-smoke-qwen-thinkfix`
- `20260426_2044_rag_multi_query_or-gemma4-26b_api-musique-multiquery-smoke`
- `20260426_2203_iterative_planning_table_or-gemma4-26b_api-musique-iter-ptable-smoke`
- `20260426_2206_advisor_planning_table_or-gemma4-26b_api-musique-advisor-smoke`
- `20260426_2246_multi_hyde_diverse_or-gemma4-26b_api-musique-multi-hyde-div-gemma-smoke`
- `20260426_2258_multi_hyde_diverse_or-gemma4-26b_api-musique-multi-hyde-div-gemma-smoke2`
- `20260427_0012_iter_hyde_groq-llama70b_api-musique-iter-hyde-llama-smoke`
- `20260427_0134_friend_foe_attribution_or-gemma27b_friend-foe-smoke`
- `20260427_0300_iter_hyde_or-gemma27b_bug-fix-smoke`
- `20260427_0301_multi_hyde_diverse_or-gemma27b_bug-fix-smoke`

### G.4 Zero-call API failures

- `20260426_1917_llm_only_groq-llama70b_api-cross-llama70b` (`api-cross-llama70b`, N=100) — summary has 0 correct, 0 avg LLM calls, 0 input/output tokens.
- `20260426_1917_llm_only_deepseek_api-cross-deepseek` (`api-cross-deepseek`, N=100) — summary has 0 correct, 0 avg LLM calls, 0 input/output tokens.

---

## Section H — Top-1 retrieval-depth ablation (audited 2026-04-28)

Scope: Llama 70B Groq x MuSiQue x N=200 paired top-1 vs top-5 retrieval-depth ablation, seed=42. The `--retrieval-k` CLI flag landed in commit `b286279`; audit doc `docs/audits/2026-04-28_top1_ablation_audit.md` verifies all top-1 rows have exactly one `evidence_store` item and one `retrieved_ids` item, with exact 200-row `idx` intersections against the top-5 baselines.

| Method | Top-5 detail log | Top-1 detail log | Paired N | Top-5 EM | Top-1 EM | Delta | McNemar p | Audit | Sign-off |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` | 200 | 27.5% | 13.0% | -14.5pp | 4.176981747e-07 | MINOR; `retrieval_k=1` proof clean, but 23/200 abstention-like predictions and one runaway/truncated output; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `rag_multi_query` | `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl` | `logs/eval_rag_multi_query_groq-llama70b_20260428_0029_detail.jsonl` | 200 | 29.0% | 14.0% | -15.0pp | 5.299581744e-06 | MINOR; `retrieval_k=1` proof clean, but 25/200 abstention-like predictions and one runaway/truncated output; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `rag_snap_hyde` | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | `logs/eval_rag_snap_hyde_groq-llama70b_20260428_0025_detail.jsonl` | 200 | 24.0% | 14.0% | -10.0pp | 0.001193242962 | MINOR; `retrieval_k=1` proof clean, no obvious final truncation, but 27/200 abstention-like predictions; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |
| `multi_hyde_diverse` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260428_0019_detail.jsonl` | 200 | 35.5% | 19.0% | -16.5pp | 5.417768989e-07 | MINOR; `retrieval_k=1` proof clean, no obvious final truncation, but 20/200 abstention-like predictions; see `docs/audits/2026-04-28_top1_ablation_audit.md` | ⚠️ APPROVED-WITH-CAVEAT |

Citation guidance: cite as a clean retrieval-depth ablation with caveat that top-1 is an under-context stress test and materially increases abstention-like predictions. Do not frame the lower top-1 EM as a harness/retrieval-k failure; the audit proves top-1 retrieval was applied on 800/800 top-1 rows.
