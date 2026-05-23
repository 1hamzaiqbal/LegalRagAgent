# Final Paper Reported-Data Lineage

This document maps the numeric result claims in `paper/main.pdf` and
`paper/FINALFINALVERSION.zip` to the JSONL files that generated them.
Paths are described relative to the repository root; links in this copy are
written from the `paper/after_report/` directory.

Canonical paper artifacts checked:

- PDF: [`main.pdf`](../main.pdf)
- Source package: [`FINALFINALVERSION.zip`](../FINALFINALVERSION.zip)
- Final source package used for inspection:
  [`../FINALFINALVERSION.zip`](../FINALFINALVERSION.zip)
- Key final table and CSV artifacts are copied into
  [`tables/`](tables/) for self-contained review.

The stale comparison artifact `paper/archive/icml_submission.pdf` is not the source of truth for the
numbers below. It contains stale narrative around HousingQA parity,
Legal-Link-EU exemplar diagnostics, and older coverage wording.

## Source Conventions

- Answer accuracy comes from detail logs under `logs/**/*.jsonl`.
- Retrieval exposure comes from retrieval caches under `caches/retrieval/**/*.jsonl`.
- Usage metrics come from the same detail logs as answer accuracy, aggregated
  in [`tables/current_usage_metrics.csv`](tables/current_usage_metrics.csv).
- Table/figure CSVs in the source package are paper-facing derived views, not
  replacements for the JSONL files.
- Percentages in the paper are rounded to one decimal point.

## Main Answer Accuracy: Table 1

All Table 1 percentages are exact-answer accuracy from the detail logs below.
The displayed averages are descriptive macro-averages over the visible cells,
so they do not have separate JSONL files.

| Dataset | Model | Method | Paper value | Source JSONL |
|---|---|---:|---:|---|
| BarExamQA | Llama 3.1 8B | LLM | 685/1195 = 57.3% | [`../logs/eval_llm_only_groq-llama8b_20260518_205159_barexam_local-snap-hyre-groq-llama8b-barexam-llm_only-nfull-k5_detail.jsonl`](../../logs/eval_llm_only_groq-llama8b_20260518_205159_barexam_local-snap-hyre-groq-llama8b-barexam-llm_only-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.1 8B | Raw question RAG | 651/1195 = 54.5% | [`../logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.1 8B | HyDE | 670/1195 = 56.1% | [`../logs/eval_rag_hyde_groq-llama8b_20260518_225428_barexam_local-snap-hyre-groq-llama8b-barexam-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_groq-llama8b_20260518_225428_barexam_local-snap-hyre-groq-llama8b-barexam-rag_hyde-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.1 8B | Snap-HyRE | 680/1195 = 56.9% | [`../logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl`](../../logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.1 8B | Gold Evidence | 717/1195 = 60.0% | [`../logs/eval_golden_passage_groq-llama8b_20260518_213011_barexam_local-snap-hyre-groq-llama8b-barexam-golden_passage-nfull-k5_detail.jsonl`](../../logs/eval_golden_passage_groq-llama8b_20260518_213011_barexam_local-snap-hyre-groq-llama8b-barexam-golden_passage-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | LLM | 966/1195 = 80.8% | [`../logs/merged/eval_llm_only_or-gemma4-26b_cloudflare_tail_20260516_barexam_nfull_k5_detail.jsonl`](../../logs/merged/eval_llm_only_or-gemma4-26b_cloudflare_tail_20260516_barexam_nfull_k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | Raw question RAG | 932/1195 = 78.0% | [`../logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | HyDE | 959/1195 = 80.3% | [`../logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | Snap-HyRE | 980/1195 = 82.0% | [`../logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`](../../logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | Gold Evidence | 939/1195 = 78.6% | [`../logs/eval_golden_passage_or-gemma4-26b_20260516_200935_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_passage-nfull-k5_detail.jsonl`](../../logs/eval_golden_passage_or-gemma4-26b_20260516_200935_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_passage-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | LLM | 940/1195 = 78.7% | [`../logs/eval_llm_only_groq-llama70b_20260515_191548_barexam_local-snap-hyre-groq-llama70b-barexam-llm_only-nfull-k5_detail.jsonl`](../../logs/eval_llm_only_groq-llama70b_20260515_191548_barexam_local-snap-hyre-groq-llama70b-barexam-llm_only-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | Raw question RAG | 891/1195 = 74.6% | [`../logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | HyDE | 958/1195 = 80.2% | [`../logs/eval_rag_hyde_groq-llama70b_20260515_222654_barexam_local-snap-hyre-groq-llama70b-barexam-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_groq-llama70b_20260515_222654_barexam_local-snap-hyre-groq-llama70b-barexam-rag_hyde-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | Snap-HyRE | 953/1195 = 79.7% | [`../logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl`](../../logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | Gold Evidence | 946/1195 = 79.2% | [`../logs/eval_golden_passage_groq-llama70b_20260515_210741_barexam_local-snap-hyre-groq-llama70b-barexam-golden_passage-nfull-k5_detail.jsonl`](../../logs/eval_golden_passage_groq-llama70b_20260515_210741_barexam_local-snap-hyre-groq-llama70b-barexam-golden_passage-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.1 8B | LLM | 3795/6853 = 55.4% | [`../logs/eval_llm_only_groq-llama8b_20260519_082209_housing_local-snap-hyre-groq-llama8b-housing-llm_only-nfull-k5_detail.jsonl`](../../logs/eval_llm_only_groq-llama8b_20260519_082209_housing_local-snap-hyre-groq-llama8b-housing-llm_only-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.1 8B | Raw question RAG | 4269/6853 = 62.3% | [`../logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.1 8B | HyDE | 4047/6853 = 59.1% | [`../logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.1 8B | Snap-HyRE | 4043/6853 = 59.0% | [`../logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl`](../../logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.1 8B | Gold Evidence | 4408/6853 = 64.3% | [`../logs/eval_golden_passage_groq-llama8b_20260519_140614_housing_local-snap-hyre-groq-llama8b-housing-golden_passage-nfull-k5_detail.jsonl`](../../logs/eval_golden_passage_groq-llama8b_20260519_140614_housing_local-snap-hyre-groq-llama8b-housing-golden_passage-nfull-k5_detail.jsonl) |
| HousingQA | Gemma 4 26B | Raw question RAG | 4531/6853 = 66.1% | [`../logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`](../../logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl) |
| HousingQA | Gemma 4 26B | HyDE | 4456/6853 = 65.0% | [`../logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_or-gemma4-26b_20260521_174454_housing_local-snap-hyre-or-gemma4-26b-housing-rag_hyde-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.3 70B | LLM | 3067/6853 = 44.8% | [`../logs/eval_llm_only_groq-llama70b_20260516_203552_housing_local-snap-hyre-groq-llama70b-housing-llm_only-nfull-k5_detail.jsonl`](../../logs/eval_llm_only_groq-llama70b_20260516_203552_housing_local-snap-hyre-groq-llama70b-housing-llm_only-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.3 70B | Raw question RAG | 4258/6853 = 62.1% | [`../logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.3 70B | HyDE | 4263/6853 = 62.2% | [`../logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl) |
| HousingQA | Llama 3.3 70B | Snap-HyRE | 4087/6853 = 59.6% | [`../logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`](../../logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl) |
| HousingQA | Llama 3.3 70B | Gold Evidence | 4611/6853 = 67.3% | [`../logs/eval_golden_passage_groq-llama70b_20260518_061249_housing_local-snap-hyre-groq-llama70b-housing-golden_passage-nfull-k5_detail.jsonl`](../../logs/eval_golden_passage_groq-llama70b_20260518_061249_housing_local-snap-hyre-groq-llama70b-housing-golden_passage-nfull-k5_detail.jsonl) |

Derived Table 1 values:

- BarExamQA Snap-HyRE answer deltas over raw RAG: +2.4, +4.0, +5.1 pp from the three BarExamQA raw/Snap-HyRE pairs above.
- BarExamQA averages: LLM 72.3, Raw 69.0, HyDE 72.2, Snap-HyRE 72.9, Gold Evidence 72.6.
- HousingQA averages over available cells: LLM 50.1, Raw 63.5, HyDE 62.1, Snap-HyRE 59.3, Gold Evidence 65.8.
- HousingQA complete-Llama Snap-HyRE deficits versus raw RAG: -3.3 pp (8B) and -2.5 pp (70B).

## Evidence Exposure: Table 2

Table 2 reports Gemma 4 26B retrieval exposure at `k=5`. HousingQA uses the
state-filtered cache for every method.

| Dataset | Method | Paper value | Source JSONL |
|---|---|---:|---|
| BarExamQA | Raw question RAG | Hit@5 1.4%, MRR@5 0.7% | [`../caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl) |
| BarExamQA | HyDE | Hit@5 11.4%, MRR@5 5.4% | [`../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl) |
| BarExamQA | Snap-HyRE | Hit@5 12.1%, MRR@5 6.0% | [`../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl) |
| HousingQA | Raw question RAG | Hit@5 36.9%, MRR@5 23.3% | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl) |
| HousingQA | HyDE | Hit@5 30.6%, MRR@5 19.6% | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl) |
| HousingQA | Snap-HyRE | Hit@5 38.1%, MRR@5 24.5% | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl) |

The source rows for these same values are also summarized in
[`tables/current_figure_metrics.csv`](tables/current_figure_metrics.csv).

## Worked Example: Table 3

The worked example is `mbe_1175` under Gemma 4 26B. Its reported predictions
come from these detail logs:

| Method | Paper value | Source JSONL |
|---|---:|---|
| Raw question RAG | predicted D, gold answer B, incorrect, gold not retrieved | [`../logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`](../../logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl) |
| HyDE | predicted D, gold answer B, incorrect, gold not retrieved | [`../logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl`](../../logs/eval_rag_hyde_or-gemma4-26b_20260517_040742_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_hyde-nfull-k5_detail.jsonl) |
| Snap-HyRE | predicted B, gold answer B, correct, gold retrieved | [`../logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`](../../logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl) |

## Answer-Stage Token Efficiency: Table 4

Table 4 is aggregated over five logged cells per method where token counts are
available. The per-cell source list is
[`tables/current_usage_metrics.csv`](tables/current_usage_metrics.csv).

| Method | Paper value | JSONL source set |
|---|---:|---|
| LLM | In/q 131, Out/q 237, Correct/M 1487.2 | BarExamQA 8B/26B/70B + HousingQA 8B/70B LLM rows in the Table 1 ledger |
| Raw RAG | In/q 2295, Out/q 339, Correct/M 241.5 | BarExamQA 8B/26B/70B + HousingQA 8B/70B raw-RAG rows in the Table 1 ledger |
| HyDE | In/q 2237, Out/q 340, Correct/M 244.5 | BarExamQA 8B/26B/70B + HousingQA 8B/70B HyDE rows in the Table 1 ledger |
| Snap-HyRE | In/q 2062, Out/q 338, Correct/M 258.8 | BarExamQA 8B/26B/70B + HousingQA 8B/70B Snap-HyRE rows in the Table 1 ledger |
| Gold Evidence | In/q 1447, Out/q 215, Correct/M 404.5 | BarExamQA 8B/26B/70B + HousingQA 8B/70B Gold Evidence rows in the Table 1 ledger |

The earlier stale PDF had `Snap-HyRE = 2001 / 376 / 268.3` over four cells.
The final paper uses the five-cell aggregate after adding the HousingQA 70B
Snap-HyRE token row from
[`../logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl`](../../logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl).

## Exemplar Probe: Table 5 and Section 5.6

These are `N=500` Gemma 4 26B retrieval probes. The stale Legal-Link-EU
diagnostic from the older paper draft is intentionally omitted from the final
main table and removed from the final source-package exemplar CSV.

| Dataset | Method | Paper value | Source JSONL |
|---|---|---:|---|
| BarExamQA | canonical Snap-HyRE | Hit@5 13.0%, MRR@5 6.4% | [`../caches/retrieval/probes/barexam_q500_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`](../../caches/retrieval/probes/barexam_q500_seed42_or-gemma4-26b_snap_hyre_k10.jsonl) |
| BarExamQA | + Exemplar | Hit@5 13.6%, MRR@5 6.3% | [`../caches/retrieval/probes/barexam_q500_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl`](../../caches/retrieval/probes/barexam_q500_seed42_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl) |
| BarExamQA | raw reference in prose/source CSV | Hit@5 1.6%, MRR@5 0.8% | [`../caches/retrieval/probes/barexam_q500_seed42_raw_question_k10.jsonl`](../../caches/retrieval/probes/barexam_q500_seed42_raw_question_k10.jsonl) |
| HousingQA | canonical Snap-HyRE | Hit@5 38.2%, MRR@5 24.3% | [`../caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`](../../caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl) |
| HousingQA | + Exemplar | Hit@5 41.2%, MRR@5 26.5% | [`../caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl`](../../caches/retrieval/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10.jsonl) |
| HousingQA | raw reference in prose | Hit@5 36.9%, MRR@5 23.3% | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl) |

Paired answer slice reported in Section 5.6 and Appendix A.4:

| Dataset | Method | Paper value | Source JSONL |
|---|---|---:|---|
| HousingQA `N=500` | canonical Snap-HyRE | 315/500 = 63.0% | [`../logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl`](../../logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl) |
| HousingQA `N=500` | + Exemplar | 314/500 = 62.8% | [`../logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260521_023301_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre_exemplar-n500-k5_detail.jsonl`](../../logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260521_023301_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre_exemplar-n500-k5_detail.jsonl) |

The reported `-0.20pp, p=1.0` paired comparison is recorded in
[`../docs/signoff_log.md`](../../docs/signoff_log.md), line for
`HousingQA state-filtered q500 diagnostic`.

## Appendix Answer Tables: Tables 8, 9, 10, and 11

The appendix answer tables reuse the same detail logs as Table 1 plus the
following control rows.

| Dataset | Model | Control | Paper value | Source JSONL |
|---|---|---:|---:|---|
| BarExamQA | Llama 3.1 8B | Rewrite | 685/1195 = 57.3% | [`../logs/eval_rag_rewrite_groq-llama8b_20260518_233753_barexam_local-snap-hyre-groq-llama8b-barexam-rag_rewrite-nfull-k5_detail.jsonl`](../../logs/eval_rag_rewrite_groq-llama8b_20260518_233753_barexam_local-snap-hyre-groq-llama8b-barexam-rag_rewrite-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | Rewrite | 964/1195 = 80.7% | [`../logs/eval_rag_rewrite_or-gemma4-26b_20260517_124940_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_rewrite-nfull-k5_detail.jsonl`](../../logs/eval_rag_rewrite_or-gemma4-26b_20260517_124940_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_rewrite-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | Rewrite | 923/1195 = 77.2% | [`../logs/eval_rag_rewrite_groq-llama70b_20260515_234357_barexam_local-snap-hyre-groq-llama70b-barexam-rag_rewrite-nfull-k5_detail.jsonl`](../../logs/eval_rag_rewrite_groq-llama70b_20260515_234357_barexam_local-snap-hyre-groq-llama70b-barexam-rag_rewrite-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.1 8B | Gold + neighbors | 738/1195 = 61.8% | [`../logs/eval_golden_plus_neighbors_groq-llama8b_20260518_215012_barexam_local-snap-hyre-groq-llama8b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl`](../../logs/eval_golden_plus_neighbors_groq-llama8b_20260518_215012_barexam_local-snap-hyre-groq-llama8b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl) |
| BarExamQA | Gemma 4 26B | Gold + neighbors | 964/1195 = 80.7% | [`../logs/eval_golden_plus_neighbors_or-gemma4-26b_20260516_233647_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl`](../../logs/eval_golden_plus_neighbors_or-gemma4-26b_20260516_233647_barexam_local-snap-hyre-or-gemma4-26b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl) |
| BarExamQA | Llama 3.3 70B | Gold + neighbors | 930/1195 = 77.8% | [`../logs/eval_golden_plus_neighbors_groq-llama70b_20260515_202827_barexam_local-snap-hyre-groq-llama70b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl`](../../logs/eval_golden_plus_neighbors_groq-llama70b_20260515_202827_barexam_local-snap-hyre-groq-llama70b-barexam-golden_plus_neighbors-nfull-k5_detail.jsonl) |

Derived appendix answer numbers:

- Table 8 deltas are macro-averages over the rows above and the Table 1 rows:
  BarExamQA Snap vs raw +3.8, HyDE vs Snap -0.7, Gold vs raw +3.6;
  HousingQA Snap vs raw -2.9, HyDE vs Snap +1.4, Gold vs raw +3.6;
  pooled Snap vs raw +1.1, HyDE vs Snap +0.1, Gold vs raw +3.6.
- Table 9 `Best main non-gold evidence` labels are row maxima over LLM, Raw,
  HyDE, Snap-HyRE, and Rewrite where Rewrite is reported.
- Table 10 `Gold vs raw pp` values are computed from the raw-RAG and Gold
  Evidence rows in Table 1.
- Table 11 `Delta pp` values are computed from Gold Evidence and Gold +
  neighbors rows above.

## Top-k Retrieval Curves and Appendix Tables 12-13

The plotted top-k curves in Figure 3 and the full Appendix Table 13 are backed
by [`tables/topk_retrieval_metrics.csv`](tables/topk_retrieval_metrics.csv).
Every row in that CSV has a `source_path` column naming the JSONL cache used.

Direct cache mapping for each plotted/table row:

| Dataset/scope | Method | Paper values | Source JSONL |
|---|---|---:|---|
| BarExamQA shared | Raw question RAG | n=1195; Hit@3 0.9, Hit@5 1.4, Hit@10 2.2, MRR@10 0.8 | [`../caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl) |
| BarExamQA Llama 3.1 8B | HyDE | n=1195; Hit@3 5.5, Hit@5 8.3, Hit@10 13.5, MRR@10 5.2 | [`../caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl) |
| BarExamQA Llama 3.1 8B | Snap-HyRE | n=1195; Hit@3 6.3, Hit@5 9.5, Hit@10 14.8, MRR@10 5.4 | [`../caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl) |
| BarExamQA Gemma 4 26B | HyDE | n=1195; Hit@3 7.1, Hit@5 11.4, Hit@10 19.1, MRR@10 6.4 | [`../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl) |
| BarExamQA Gemma 4 26B | Snap-HyRE | n=1195; Hit@3 7.9, Hit@5 12.1, Hit@10 18.7, MRR@10 6.9 | [`../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl) |
| BarExamQA Llama 3.3 70B | HyDE | n=1195; Hit@3 6.5, Hit@5 10.5, Hit@10 17.6, MRR@10 6.1 | [`../caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl) |
| BarExamQA Llama 3.3 70B | Snap-HyRE | n=1195; Hit@3 7.2, Hit@5 11.0, Hit@10 18.5, MRR@10 6.6 | [`../caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl) |
| HousingQA state-filtered shared | Raw question RAG | n=6853; Hit@3 29.5, Hit@5 36.9, Hit@10 48.1, MRR@10 24.8 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl) |
| HousingQA Llama 3.1 8B | HyDE | n=6853; Hit@3 21.6, Hit@5 27.0, Hit@10 34.5, MRR@10 17.9 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl) |
| HousingQA Llama 3.1 8B | Snap-HyRE | n=6853; Hit@3 23.4, Hit@5 29.6, Hit@10 37.0, MRR@10 19.9 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl) |
| HousingQA Llama 3.3 70B | HyDE | n=6853; Hit@3 27.9, Hit@5 34.9, Hit@10 42.6, MRR@10 23.6 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl) |
| HousingQA Llama 3.3 70B | Snap-HyRE | n=6853; Hit@3 18.1, Hit@5 23.1, Hit@10 31.3, MRR@10 15.3 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl) |

Mean rows in Table 13 and Figure 3 are computed from the model-specific caches
listed above:

- BarExamQA HyDE mean over 3 models: Hit@3 6.4, Hit@5 10.0, Hit@10 16.7, MRR@10 5.9.
- BarExamQA Snap-HyRE mean over 3 models: Hit@3 7.1, Hit@5 10.9, Hit@10 17.3, MRR@10 6.3.
- HousingQA HyDE mean over 2 complete models: Hit@3 24.7, Hit@5 30.9, Hit@10 38.5, MRR@10 20.7.
- HousingQA Snap-HyRE mean over 2 complete models: Hit@3 20.7, Hit@5 26.3, Hit@10 34.1, MRR@10 17.6.

Appendix Table 12 uses:

| Corpus scope | Paper values | Source JSONL |
|---|---:|---|
| HousingQA national raw corpus | Hit@1 0.8, Hit@3 1.9, Hit@5 2.8, Hit@10 5.1, MRR@10 1.8 | [`../caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_raw_question_k10.jsonl) |
| HousingQA jurisdiction state filter | Hit@1 15.5, Hit@3 29.5, Hit@5 36.9, Hit@10 48.1, MRR@10 24.8 | [`../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`](../../caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl) |

## Non-result Numeric Text

These values are configuration or dataset descriptors rather than model-result
measurements, but they are still grounded in the JSONL/cache row counts above:

- BarExamQA `N=1195`: all full BarExamQA detail logs and caches listed above.
- HousingQA `N=6853`: all full HousingQA detail logs and caches listed above.
- Exemplar probes `N=500`: the probe caches and HousingQA paired answer logs
  listed in the Table 5 section.
- Retrieval cutoffs `k=5` and `k=10`: encoded in cache filenames and table
  generation (`*_k10.jsonl` contains the full top-10 ranked list; Table 2 uses
  top five from those lists).
- Model sizes 8B, 26B, and 70B: route/model descriptors, not result metrics.

## Known Stale-Source Hazards

- `paper/archive/icml_submission.pdf` contains stale claims; use
  `paper/main.pdf` and `paper/FINALFINALVERSION.zip`.
- `paper/archive/snap_hyre_2025_05_18/current_audited_rows.csv` includes a stale row
  where the HousingQA Gemma exemplar diagnostic is labeled as `snap_hyre`.
  The final paper does not use that row as a full-N answer result; the final
  figure metrics relabel it as `snap_hyre_exemplar_q500`.
- The final paper intentionally leaves HousingQA Gemma 26B LLM, Snap-HyRE
  answer accuracy, and Gold Evidence cells blank in Table 1.
