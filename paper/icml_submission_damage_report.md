# `icml_submission.pdf` Damage Report

This report compares the stale comparison PDF
[`icml_submission.pdf`](icml_submission.pdf) against the corrected upload PDF
[`main.pdf`](main.pdf). It focuses on result-claim damage: claims that were
wrong, stale, overbroad, or likely produced by a merge between older paper
sources and newer tables.

For per-number JSONL lineage, see
[`reported_data_lineage.md`](reported_data_lineage.md).

## Breadcrumbs

| Artifact | Status | Pages | SHA-256 |
|---|---:|---:|---|
| `paper/icml_submission.pdf` | stale comparison target | 10 | `8c78723c488ec7a2fd12abc584c3befdf15d1022b9137421cb3251dbc5cc43e0` |
| `paper/main.pdf` | committed final PDF, byte-identical to PDF inside `FINALFINALVERSION.zip` | 11 | `641fd97574f961bceab83fdd3ed3f8cc0d07e9ce326fb6f672cc7788eaeb57ae` |
| `paper/FINALFINALVERSION.zip` | committed final source package | n/a | `522d5ef70c24f3dbca29acbe4625a560e95f73625797219c245f65bfd76f7813` |

No literal merge-conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) were found
in either PDF text or in the final unpacked source. The damage is semantic:
old prose and appendix text survived around newer tables.

## Executive Summary

The stale PDF has the right broad paper shape and much of the right table data,
but it contains several claim-level problems:

1. It calls HousingQA a parity case even though Table 1 shows Snap-HyRE below
   raw-question RAG on complete answer rows.
2. It overstates the exemplar diagnostic as broadly lifting retrieval above
   raw baselines or across all tested benchmarks.
3. It leaves Legal-Link-EU in the main exemplar story, where the numbers show
   raw search is far stronger than generated-query search.
4. It states the Gold Evidence comparison too strongly for Llama 3.3 70B.
5. It reports stale token-efficiency numbers for Snap-HyRE.
6. It says Figure 3 HousingQA averages include Gemma 26B even though the final
   figure caption should only describe complete full-cache generated-query
   rows for 8B and 70B.
7. It has stale appendix coverage wording (`31 signed cells`) and weaker
   reproducibility/provider disclosure.

The final PDF fixes those issues without changing the core answer-accuracy
matrix. The actual data changes are concentrated in Table 4's token-efficiency
aggregate and Table 5's exemplar scope.

## Claim-Level Damage and Corrections

| Area | Stale `icml_submission.pdf` claim | Problem | Final `main.pdf` correction |
|---|---|---|---|
| Abstract: Gold Evidence | "matches or exceeds Gold Evidence ... on Gemma 4 26B and Llama 3.3 70B" | Gemma 26B clearly exceeds Gold Evidence: 82.0 vs 78.6. Llama 70B is only 79.7 vs 79.2, a +0.5pp descriptive gap that should not be sold as a strong exceedance. | "exceeds a single labeled-gold-passage control on Gemma 4 26B, and approximately matches it on Llama 3.3 70B." |
| Abstract: HousingQA | "trades a small answer margin for a measurable retrieval lift on the largest open-weight model" plus exemplar "lifts retrieval above the raw baseline at every size" | The answer side is not parity: complete Llama rows favor raw RAG by 3.3pp and 2.5pp. The exemplar probe is Gemma 26B `N=500`, not every size. | Raw RAG is stated as the stronger answer baseline; Snap-HyRE trails raw by 2.5-3.3pp on complete Llama rows; Gemma 26B evidence probe is described as prompt conditioning improving top-five exposure. |
| Introduction | "Snap-HyRE delivers answer parity rather than lift" on HousingQA | False against Table 1: Snap-HyRE HousingQA avg is 59.3 vs raw RAG 63.5, and complete Llama rows are below raw. | "raw-question RAG remains strongest and Snap-HyRE trails by 2.5-3.3pp, although a Gemma 26B evidence probe shows prompt conditioning can still improve top-five exposure." |
| Section 5.1 | "The lift grows with model size" | Answer-accuracy deltas grow (+2.4, +4.0, +5.1), but retrieval Hit@5 is not monotone: 9.5, 12.1, 11.0. | "The answer gain grows with model size, while retrieval exposure is not strictly monotone: Gemma 4 26B has the highest Hit@5." |
| Section 5.1/5.2 HousingQA | "parity in Table 1" and "generated-query rows trade some labeled-evidence exposure for broader corpus context that the answer model still uses well" | The table does not show parity; it shows raw answer accuracy remains stronger. Retrieval is mixed, with only the Gemma 26B Snap-HyRE evidence row beating raw in Table 2. | "Most generated-query rows trail the state-filtered raw retrieval baseline; the Gemma 26B Snap-HyRE row is a small positive retrieval exception, not an answer-accuracy win." |
| Figure 3 caption | HousingQA averages over Llama 3.1 8B, Gemma 4 26B, and Llama 3.3 70B | The final top-k figure/table uses the two complete full-cache generated-query model rows for HousingQA: 8B and 70B. Gemma 26B appears in Table 2/probe evidence, not in the Figure 3 HousingQA mean. | Caption now says HousingQA averages over the two complete generated-query models, Llama 3.1 8B and Llama 3.3 70B, with the state filter. |
| BarExamQA retrieval delta prose | "+9.5pp Hit@5 / +4.8pp MRR@5" | Correct as an average, but stale phrasing read like one cell. | Adds "average" to clarify the aggregation basis. |
| Section 5.4 | "matches or exceeds the labeled gold passage on the two larger models" | Overstates the 70B comparison by treating +0.5pp as a real exceedance. | "exceeds ... on Gemma 4 26B ... and approximately matches it on Llama 3.3 70B." |
| Section 5.5 / Table 4 | Snap-HyRE efficiency = 2001 input tokens/q, 376 output tokens/q, 268.3 correct/M over 4 cells | Stale aggregate; it omitted the HousingQA 70B Snap-HyRE token row. | Snap-HyRE efficiency = 2062 input tokens/q, 338 output tokens/q, 258.8 correct/M over five logged cells. |
| Section 5.6 exemplar | "lifts retrieval on every benchmark we tested" and "recovers retrieval without costing answer accuracy" | Overbroad and too absolute. Legal-Link-EU raw retrieval is 90.0 Hit@5, while exemplar Snap-HyRE is 75.8; answer accuracy on HousingQA is 62.8 vs 63.0, so the right claim is no detectable answer change, not no cost in an absolute sense. | Final text says +Exemplar improves canonical Snap-HyRE retrieval on the two main datasets, beats raw on HousingQA Hit@5, and the paired HousingQA answer slice is statistically unchanged (`p=1.0`). |
| Main exemplar table | Includes Legal-Link-EU row: 68.2/55.6 -> 75.8/62.6 and references raw Legal-Link 90.0 | The Legal-Link numbers are real diagnostics, but they undermine a main-paper exemplar headline because raw search is much stronger. They are boundary evidence, not the main story. | Final Table 5 reports only BarExamQA and HousingQA main-dataset `N=500` probes. |
| Legal-Link paragraph | Main text explains Legal-Link-EU limitations | True but stale/distracting for a two-benchmark main paper under time pressure; also forced a confusing "every benchmark" exemplar frame. | Removed from main narrative. |
| Conclusion | "Snap-HyRE reaches parity with raw-question RAG" on corpus-shaped questions | False/overbroad for HousingQA answer accuracy. | Conclusion now says raw-question RAG remains the stronger answer baseline on HousingQA, while evidence probes show targeted prompt conditioning can recover exposure. |
| Appendix coverage | "31 signed cells out of 42 expected" | Stale count and operational wording. | "32 validated cells out of 42 expected; missing cells are left blank rather than imputed" plus coverage and row-note tables. |
| Appendix exemplar details | Uses `q500` wording and still mentions Legal-Link-EU corpus settings | Carries old diagnostic scope into the final two-benchmark paper. | Uses `N=500`, restricts reported interfaces to BarExamQA and HousingQA, and removes Legal-Link-EU from the reported exemplar details. |
| Reproducibility | Generic "released package will include records..." | Too vague for reviewer reproduction and did not name route/provider behavior. | Adds route labels, provider mapping, deterministic decoding, parse/answer retry constraints, same-route retries, and no silent fallback. |
| References | Zheng venue placeholder: "Proceedings ... on ZZZ" | Clear bibliography placeholder. | Fixed to CS and LAW 2025 / Symposium on Computer Science and Law venue metadata. |

## Table and Figure Data Changes

### Table 1: Main Answer Accuracy

No numeric data changed between the stale PDF and final PDF. The final paper
keeps:

- BarExamQA Snap-HyRE: 56.9 / 82.0 / 79.7, average 72.9.
- HousingQA Snap-HyRE: 59.0 / blank / 59.6, average 59.3.
- HousingQA raw-question RAG: 62.3 / 66.1 / 62.1, average 63.5.

The damage was not the table; it was stale prose calling HousingQA "parity"
despite this table.

### Table 2: Gemma 26B Evidence Exposure

No numeric data changed. The final caption adds that this is retrieval
exposure, not answer accuracy.

The key correction is interpretive: HousingQA Snap-HyRE at Gemma 26B has
better evidence exposure than raw RAG in Table 2 (38.1/24.5 vs 36.9/23.3),
but this is not a completed answer-accuracy win in Table 1.

### Figure 3: Top-k Curves

The figure was regenerated from
[`after_report/tables/topk_retrieval_metrics.csv`](after_report/tables/topk_retrieval_metrics.csv),
which points each plotted row to a retrieval-cache JSONL. The regenerated PNG is:

[`after_report/plots/figure3_topk_retrieval_curves_regenerated.png`](after_report/plots/figure3_topk_retrieval_curves_regenerated.png)

The caption correction matters more than the pixels: HousingQA generated-query
means are over the two complete full-cache rows, not all three model sizes.

### Table 4: Token Efficiency

This is the main numeric table change:

| Metric | Stale PDF | Final PDF |
|---|---:|---:|
| Snap-HyRE cells | 4 | five logged cells per method |
| Snap-HyRE input tokens/q | 2001 | 2062 |
| Snap-HyRE output tokens/q | 376 | 338 |
| Snap-HyRE correct/M answer-stage tokens | 268.3 | 258.8 |

The final value is still the best non-gold evidence retrieval method:
Snap-HyRE 258.8 vs HyDE 244.5 and raw RAG 241.5.

### Table 5: Exemplar Probe

The final table removes Legal-Link-EU and keeps only the main datasets:

| Dataset | Final reported change |
|---|---:|
| BarExamQA | Snap-HyRE 13.0/6.4 -> +Exemplar 13.6/6.3 |
| HousingQA | Snap-HyRE 38.2/24.3 -> +Exemplar 41.2/26.5 |

The final strongest claim is:

> One sanitized corpus passage improves generated-query retrieval exposure on
> Gemma 26B `N=500` probes, especially HousingQA, while the paired HousingQA
> answer slice is statistically unchanged.

It is not an answer-accuracy claim and not a universal "generated queries beat
raw search" claim.

### Appendix Table 13: Top-k Bolding

The final source package also corrects Appendix Table 13 formatting: the
HousingQA `Mean over 2 full models` HyDE row is now bolded on Hit@3, Hit@5,
Hit@10, and MRR@10 because those values are higher than the corresponding
Snap-HyRE mean row. This changes emphasis only, not the reported numbers.

## Current Final State

- `paper/main.pdf` is the upload PDF to trust.
- `paper/FINALFINALVERSION.zip` contains the same `main.pdf`.
- `paper/reported_data_lineage.md` maps paper numbers to JSONL/cache sources.
- Figure 3 was regenerated from the final `topk_retrieval_metrics.csv`.
- No unresolved conflict markers were found.
