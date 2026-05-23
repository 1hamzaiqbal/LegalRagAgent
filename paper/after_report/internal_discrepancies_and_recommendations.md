# Internal Discrepancies and Recommendations

This note separates true paper problems from acceptable caveats in the final
`main.pdf`.

## Stale PDF Problems

These are present in `paper/icml_submission.pdf` and fixed in `paper/main.pdf`:

- HousingQA is called "parity" even though Table 1 favors raw-question RAG.
- Exemplar prompting is described as lifting retrieval on every tested
  benchmark, while Legal-Link-EU raw retrieval remains much higher.
- The exemplar result is phrased as "without costing answer accuracy" instead
  of a paired `N=500` no-detectable-change result.
- The Gold Evidence comparison says "matches or exceeds" on 70B, where the
  difference is only +0.5pp.
- Figure 3's HousingQA caption says it averages over three models, including
  Gemma 26B, while the final complete full-cache generated-query mean is over
  the two complete Llama rows.
- Table 4's Snap-HyRE token-efficiency row is stale: 4 cells and 268.3
  correct/M instead of five cells and 258.8 correct/M.
- Appendix coverage says 31 signed cells instead of 32 validated cells.

Full details are in [`damage_report.md`](damage_report.md).

## Final PDF Items That Are Acceptable but Easy to Misread

### Table 2 vs Table 1 HousingQA Gemma 26B

Table 2 reports Gemma 26B retrieval exposure for HousingQA Snap-HyRE:
Hit@5 38.1 and MRR@5 24.5. Table 1 leaves the HousingQA Gemma 26B Snap-HyRE
answer cell blank.

This is not a numeric contradiction. It means the retrieval cache/probe exists
for evidence exposure, while the final answer-accuracy row is not included as
a main full-N answer result. The final text correctly calls this an evidence
probe/diagnostic, not an answer-accuracy win.

Recommendation: keep the Table 2 caption's "retrieval exposure, not answer
accuracy" wording. If there is room, add "answer cells not shown in Table 1 are
not imputed" to the appendix rather than the main table.

### Figure 3 HousingQA Mean

Figure 3 uses complete HousingQA generated-query top-k rows for Llama 3.1 8B
and Llama 3.3 70B. The Gemma 26B HousingQA retrieval row appears in Table 2
and the exemplar/probe discussion, but not in the Figure 3 HousingQA generated
mean.

Recommendation: the current caption is correct. Do not revert to the stale
"three model sizes" caption.

### Table 5 Filename and CSV Name

The table source file is named `exemplar_probe_q20.tex`, and the CSV is named
`exemplar_probe_q20_metrics.csv`, but the final table reports `N=500` values.
The caption and values are correct; the filenames are stale.

Recommendation: after submission, rename these files to
`exemplar_probe_n500.*` and update `\input{...}` references. Do not do this
right before upload unless the zip is rebuilt and checked, because it is a
source hygiene fix, not a PDF-data fix.

### Legal-Link-EU Exemplar Rows Removed From Final CSV

`figures/exemplar_probe_q20_metrics.csv` previously included Legal-Link-EU
rows even though final Table 5 omits them.

This was not a PDF discrepancy because the table source excluded Legal-Link-EU,
but it was a package hygiene issue: a reader inspecting the CSV could wonder
why those rows were not reported. The unused rows have been removed from the
final source-package CSV.

Recommendation: keep Legal-Link-EU out of the final main exemplar narrative
unless the paper is expanded to report it as a boundary case.

### Historical Generator Drift

`paper/snap_hyre_2025_05_18/build_current_paper_assets.py` is not fully aligned
with the final source package.

Recommendation: do not rerun it blindly for final submission. Patch it or build
a small final-package generator that exactly emits the final tables/figures.

### Root Paper README Drift

`paper/README.md` still points readers toward `paper/snap_hyre_2025_05_18/`
and the older generator workflow. That is stale for the final upload package.

Recommendation: use this after-report and `paper/FINALFINALVERSION.zip` as the
current package map. Update `paper/README.md` after the deadline so it no longer
looks like the final build contract.

### Current Audited Rows CSV Stale Label

`paper/snap_hyre_2025_05_18/current_audited_rows.csv` has a row labeling the
HousingQA Gemma exemplar diagnostic as `snap_hyre`. The final package corrects
the figure metric label to `snap_hyre_exemplar_q500` and does not use it as a
full-N answer cell.

Recommendation: fix or archive the stale CSV before future audits, because it
is the easiest file to accidentally use incorrectly.

## Final PDF Items I Did Not Find as Discrepancies

- Table 1 bolding is consistent with the caption: gold is separated and not
  eligible; deployable-method maxima are bolded.
- Table 2 bolding is consistent: Snap-HyRE is the largest value in all four
  Gemma 26B retrieval columns.
- Appendix Table 13 bolding is now consistent with the caption: the HousingQA
  mean row correctly bolds HyDE, which is higher than Snap-HyRE on all four
  top-k metrics.
- Table 5 bolding is consistent with the caption: BarExamQA splits the winner
  across Hit@5 and MRR@5; HousingQA +Exemplar wins both.
- The BarExamQA "answer gain grows with model size" claim is accurate for
  answer deltas (+2.4, +4.0, +5.1pp), and the final paper separately notes
  retrieval is not monotone.
- The "raw RAG remains stronger on HousingQA answer accuracy" claim is
  consistent with Table 1.

## Recommendations Before Any Further Upload

1. Upload `paper/main.pdf`, not `paper/icml_submission.pdf`.
2. Keep `paper/FINALFINALVERSION.zip` synchronized with the rebuilt
   `paper/main.pdf`.
3. Include `paper/after_report/number_lineage.md` internally as the audit map;
   do not include it in the anonymous submission unless allowed.
4. Do not mention active/in-progress runs in the paper.
5. After the deadline, clean the final generator and stale filenames so the
   final package can be regenerated from one command.
