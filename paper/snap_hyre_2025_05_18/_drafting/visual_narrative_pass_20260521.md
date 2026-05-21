# Visual and Narrative Pass - 2026-05-21

## Main-paper decisions

- Keep Figure 1 as the simple Snap-HyRE mechanism diagram, not the full method
  ladder. The figure should explain that the first call writes private initial
  reasoning and separate search text; only the search text retrieves evidence;
  the final answer call receives the original question and retrieved evidence.
- Keep the main answer table to deployable rows plus a single gold-passage
  oracle control. Move `golden_plus_neighbors` out of the final-facing matrix;
  it is useful only as a neighbor-dilution diagnostic.
- Add a direct BarExamQA retrieval-gain figure showing Hit@5 and MRR@5 gains
  over raw-question RAG. This is clearer than the earlier retrieval-answer
  scatter for the main claim.
- Keep HousingQA framed as an interface boundary: state filtering fixes the
  dominant retrieval error, and generated-query methods should be compared only
  after that corpus scope is fixed.

## Appendix decisions

- Use compact tables for coverage, control rows, and neighbor dilution instead
  of carrying redundant appendix plots.
- Keep the top-k retrieval curves as the only appendix plot: it shows shape
  across cutoffs and is harder to replace with one compact main-text number.
- Keep the worked example in the appendix as a compact one-column table to avoid
  spending main-paper space on mechanism narration.
- Keep exemplar-probe rows explicitly probe-only.

## Verification

- Regenerated paper assets with:
  `python3 paper/snap_hyre_2025_05_18/build_current_paper_assets.py`
- Rebuilt the PDF from `paper/snap_hyre_2025_05_18` with:
  `tectonic main.tex`
- Render check after the plot-cull rebuild and appendix page break: 14 total
  pages; main paper ends on page 6, references are page 7, appendix starts on
  page 8, and active figures are limited to the method diagram, BarExamQA
  retrieval-gain plot, and top-k appendix curves.
