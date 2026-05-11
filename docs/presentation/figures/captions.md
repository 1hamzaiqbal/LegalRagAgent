# Figure captions + context

All figures live in `docs/presentation/figures/` as PNGs. Click any figure path in VSCode to open it inline.

## `12_diagnostic_adaptation_calibration_ablation.png`

**May 11 calibration ablation — Gemma 4 26B legal-only N=200 portfolio.**
The table compares matched baseline routes, snap-only reasoning, legal query
rewrite, a preselected HyRE-family route, and diagnostic controller routes. The
controller is the strongest macro row at 77.9% while averaging 1.30 LLM calls
per question; query rewrite includes mixed-N controls outside BarExam.

Source data: `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json`
and `docs/snap_only_controls_2026-05-11.json`.

## `13_diagnostic_adaptation_heldout_ablation.png`

**May 11 held-out check — rows 200-249 across the four legal benchmarks.**
Matched baselines average 71.5%, legal query rewrite averages 75.5%, and the
selected diagnostic routes average 77.5% at 1.54 LLM calls per question. The
lift is concentrated in HousingQA verifier routing and CaseHOLD diverse HyRE;
BarExam and SCALR remain route-policy refinement cases.

Source data: `docs/heldout_controller_eval_2026-05-10.json` and
`docs/heldout_query_rewrite_2026-05-10.json`.

## `14_diagnostic_controller_macro_lift.png`

**Macro accuracy versus average LLM calls.** The controller lifts calibration
macro accuracy over the matched baseline and preselected HyRE-family row while
spending fewer average calls than fixed two-call methods, because it can route
some questions through cheaper replay/verifier policies.

Source data: `docs/diagnostic_controller_portfolio_comparison_2026-05-10.json`.

## `15_bottleneck_diagnostic_route_map.png`

**Evidence signal to bottleneck to policy route.** The route map is the compact
meeting narrative: BarExam points to rewrite-vs-HyRE selection, HousingQA to
state-filter plus verifier, CaseHOLD to option-conversion work, and SCALR to
disagreement arbitration.

Source: scripted from `docs/meeting_prep_2026-05-11_diagnostic_adaptation.md`
and the linked source-gated result docs.

## `16_method_ladder_flowchart.png`

**Inherited method ladder for the ablation table.** The flowchart shows how the
main controls differ: baseline RAG, snap-only, HyRE/HyDE-only retrieval,
Snap-HyRE, and diagnostic routing. Use it to explain that the comparison is not
a broad prompt sweep; each row adds a specific reasoning or routing mechanism.

Source: scripted from `scripts/build_meeting_package_figures.py`.

## `01_llama70b_method_matrix.png`

**Llama 3.3 70b dense x MuSiQue N=200, pre-2call method matrix.** 8 method bars sorted by mode name. Two methods lift over the rag_simple baseline: `multi_hyde_diverse` +8pp p=0.020 SIG (green) and `iterative_planning_table` +8.5pp p=0.053 TRENDING (yellow). `subagent_rag` -12pp p=0.0007 SIG NEG (red). `snap_hyde_2call` is the newer 2026-04-30 method vehicle and should be discussed alongside this figure. The blue dashed line is the `rag_simple` baseline at 27.5%.

Source data: `docs/signoff_log.md` Section B.1 + `docs/mcnemar_2026-04-27.md`.

Historical note: this figure predates the 2026-04-28 `snap_hyde_2call` pivot.
Use `08_musique_snap_hyde_2call_pivot.png` for the current class-report
MuSiQue headline.

## `08_musique_snap_hyde_2call_pivot.png`

**Llama 3.3 70b × MuSiQue N=200 — current snap-HyDE 2-call pivot.**
`snap_hyde_2call` reaches 37.0% EM (+9.5pp, p=0.0079), above
`multi_hyde_diverse` at 35.5% (+8.0pp, p=0.0195) and
`iterative_planning_table` at 36.0% (+8.5pp, p=0.0533). The baseline is
`rag_simple` at 27.5%; current `subagent_rag` is a significant negative at
15.5% (-12.0pp, p=0.0007). Treat `snap_hyde_2call` as the best point estimate
and lowest-cost lifting arm here, not as statistically separated from MHD or
iterative planning.

Source data: `docs/snap_hyde_2call_2026-04-28.md` and `docs/signoff_log.md`
Section B.1.

## `09_bottleneck_rag_graphical_abstract.png`

**Bottleneck-typed RAG graphical abstract.** AI-generated process schematic
used in the class report to replace the earlier text-only pivot box. It shows
legal documents and model priors feeding controlled retrieval interventions,
then splitting into successful and failed evidence-use paths. It is conceptual,
not a quantitative result.

Source: generated with the local `imagegen` skill on 2026-04-30.

## `10_musique_current_budget_frontier.png`

**Current budget frontier — Llama 70B × MuSiQue N=200.** Scatter plot of EM
against average total tokens per question, with point size/color tracking LLM
calls and method family, plus a call-budget panel. Includes the current
`snap_hyde_2call` pivot: 37.0% EM, 2.00 calls/q, 1,135 input + 249 output
tokens/q. It replaces the older `07_cost_vs_accuracy.png` in the class report.

Source data: `docs/evidence_matrix_2026-04-30.md` and current MuSiQue detail
logs listed there.

## `11_barexam_golden_snap_mechanism.png`

**BarExam golden-passage mechanism — Gemma 4 26B-A4B N=1195.** Bars compare
`rag_simple`, `golden_passage`, `llm_only`, `snap_only_in_final`, and
`rag_snap_hyde`; the right panel shows the paired `llm_only` vs
`golden_passage` flip asymmetry (96 correct answers hurt vs 83 wrong answers
helped). Use this to explain why the single gold-passage control is not an
oracle ceiling.

Source data: `docs/golden_paradox_audit_2026-04-27.md` and
`docs/methods_vs_golden_audit_2026-04-27.md`.

## `02_barexam_cross_size.png`

**BarExam Tier 3 (full corpus N=1195) — `rag_snap_hyde` wins on both Gemma 4 sizes.** Gemma 4 26B-A4B: 78.08% → 81.17% (+3.09pp). Gemma 4 E4B: 58.49% → 62.18% (+3.69pp). The lift is consistent cross-size, which is what makes this a paper-grade Tier 3 finding rather than single-model noise.

Source data: `docs/audit_log.md` (post-fix re-scored) + `docs/signoff_log.md` Section A.

## `03_mechanism_decomposition.png`

**Mechanism — Llama 70b MuSiQue N=200.** rag_simple 27.5% → rag_multi_query 29.0% (+1.5pp NS, query diversity alone) → multi_hyde_diverse 35.5% (+8pp SIG). The +8pp lift decomposes into ~+1.5pp from diversity and ~+6.5pp from HyDE-style answer-bearing passages. **HyDE-style passages do ~80% of the work; diversity alone is non-significant.**

Source data: `docs/mcnemar_2026-04-27.md` Section B.2.

## `04_cross_domain_specificity.png`

**Methods are domain-specific.** The BarExam winner (`rag_snap_hyde`) is +3.09pp on its native domain (BarExam) but -3.5pp NS on MuSiQue. The MuSiQue winner (`multi_hyde_diverse`) is +8pp SIG on its native domain but -2.5pp NS on BarExam (paired N=200 against the Tier 3 baseline's first 200 records). Conclusion: there is no universal RAG trick — different tasks expose different bottlenecks.

Source data: `docs/signoff_log.md` Section B.4 + `docs/mcnemar_2026-04-27.md` Update 12:30.

## `05_cross_family_check.png`

**Cross-family check — mhd lift is NOT yet universal at Tier 2.** Llama 3.3 70b dense: +8pp p=0.020 SIG. Gemma 3 27B dense: +2.5pp p=0.59 NULL. The mhd lift held on Llama at full Tier 2 confirmation but did NOT replicate on Gemma 3 27B. Gemma 4 26B and Qwen3 30B MoE full-corpus runs are in flight (will land after meeting).

Source data: `docs/mcnemar_2026-04-27.md` Sections B.1 + B.3.

## `07_cost_vs_accuracy.png`

**Cost vs accuracy — Llama 70b MuSiQue N=200.** 4-panel figure:
- **A. LLM calls per question**: `iterative_planning_table` and `iter_hyde` use 6.3-6.8 calls/q (6-7× the simple methods). `mhd` uses 2 calls/q.
- **B. Input tokens per question**: `iter_hyde` is 4× the average (3,619 input tokens/q). Most other methods are 800-2,500.
- **C. Output tokens per question**: `iter_hyde` generates 822 tokens/q vs 84 for `rag_simple`. mhd generates 417/q.
- **D. Efficiency frontier**: scatter of EM vs total tokens/q. **`mhd` is the cheapest of the lifters** (top-left = better). `iterative_planning_table` matches mhd's EM but at ~4× the token cost. `iter_hyde` is the most expensive method and lifts none.

Source data: `logs/experiments.jsonl` rows tagged `captain-llama70b-musique-*-n200`.

## `06_barexam_26b_full_matrix.png`

**BarExam Gemma 4 26B-A4B Tier 3 method matrix (full corpus N=1195), sorted by EM.** Winner: `rag_snap_hyde` at 81.17% (+3.09pp). Runner-up: `snap_only_in_final` at 80.59% (+2.51pp). `llm_only` at 79.75% is a strong no-retrieval reference point. `subagent_hybrid` at 74.23% is the only method that materially hurts (-3.85pp).

Source data: `docs/audit_log.md` (post-fix re-scored from cluster vLLM detail logs) + `docs/signoff_log.md` Section A.1.

---

## Audit + caveats summary (cross-references)

- **`rag_snap_hyde` snap-dominance** (76-83% pred==snap on 26B): BY DESIGN, not contamination. See `docs/audits/2026-04-27_barexam_26b_audit.md` and signoff Top 5 #1.
- **`subagent_rag` -12pp + 100% gap-routing trigger**: implementation caveat. See `docs/audits/2026-04-27_llama70b_musique_audit.md` and signoff Top 5 #5.
- **OR-Gemma serving runaway loops** (gemma4_full + iter_ptable Gemma killed): cluster vLLM unaffected. See signoff Section D'.
- **Tier system**: N<200 directional only. See signoff Section A header.
