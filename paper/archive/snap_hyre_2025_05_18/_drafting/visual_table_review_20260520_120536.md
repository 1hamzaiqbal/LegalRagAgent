# Visual and Table Review - 2026-05-20 12:05

This note records the naive plot/table pass for the timestamped Overleaf export.
The active paper uses `icml2026`, not NeurIPS. The current source is an internal
preprint/export copy with author info visible via `\usepackage[preprint]{icml2026}`;
remove `[preprint]` for a blind ICML review copy.

## Applied in this pass

- Replaced the generic intro art with a deterministic signal diagram showing the
  private snap answer, the HyRE passage used as the retrieval query, and the
  second answer call that does not receive the tentative answer as evidence.
- Clarified figure captions for accuracy versus retrieval exposure, MAS
  same-source proxy metrics, dataset-specific y-axes, and unbalanced means.
- Relabeled retrieval-answer scatter x-axis to `Hit@5 / same-source@5`.
- Added `open` markers for missing retrieval slots in the retrieval-by-method
  panels.
- Reworded oracle-control captions away from strong "ceiling/headroom" language.
- Added units and sign conventions to table captions/headers.
- Bolded strongest non-oracle answer cells in the main answer/control tables.
- Changed retrieval-table missing notation: `Hit@5/n.s.` means MRR@5 is not
  signed; `--` means no audited retrieval row.

## Keep Main For Now

- Main answer matrix and heatmap: central empirical accounting.
- Snap-HyRE versus raw deltas: direct claim support.
- Retrieval-answer delta scatter: important mixed-quadrant story.
- Retrieval-by-method panels: useful, but keep the dataset-specific y-axis
  caveat visible.
- Oracle controls: useful diagnostic, not a true upper bound.

## Likely Appendix Or Cut Later

- Dataset-level generated-query mean deltas: useful as descriptive context, but
  easy to misread as a balanced comparison.
- Method mean accuracy: still visually leaderboard-like despite caveats.
- Conceptual call-count plot: exploratory and not an actual dollar/token cost.
- Answer-pass token plot/table: useful only with the caveat that first-stage
  generation usage is excluded.
- Completion grid: appendix-only; it mostly documents matrix coverage.
- Exemplar probe: appendix/probe-only until full-row answer checks are audited.

## Imagegen Decision

The imagegen skill was reviewed for this request. I did not add an AI-generated
bitmap to the paper because the useful diagrams here require exact labels,
method invariants, and reproducible layout. The better fit is code-native
diagramming from the paper asset script. If we use image generation later, it
should be for a non-textual visual metaphor or cover-style asset, not for the
core method diagram.

## Next Visual Target

The most useful new visual would be a worked-example strip comparing three
methods on a Snap-HyRE win, for example:

- HousingQA Hawaii `idx=1066`: Raw RAG `No`, HyDE `No`, Snap-HyRE `Yes`, gold
  `Yes`; Snap-HyRE retrieves `HI Rev Stat § 521-69`, the landlord-remedy statute
  for tenant waste.
- BarExamQA `mbe_1159`: Raw RAG `D`, HyDE `D`, Snap-HyRE `B`, gold `B`; the
  Snap-HyRE row focuses retrieval on the Fourth Amendment seizure standard.

Before promoting either into the main paper, manually quote-check the retrieved
passage snippets and keep the example framed as illustrative rather than
representative.
