# Upload Instructions

Upload these files into the Overleaf project root, preserving paths:

- `sections/2Preliminary.tex`
- `sections/6Conclusion.tex`

Choose overwrite/replace when Overleaf asks. No rename, delete, or `main.tex`
edit is needed.

What changes:

- `sections/2Preliminary.tex` becomes a standalone `Background` section.
- `sections/6Conclusion.tex` now contains `Related Work` followed by the
  existing `Conclusion`.

Temporary compile check against `paper/scope_edit_v0.zip`: `tectonic main.tex`
completes with only underfull-box warnings, and the conclusion begins on
main-text page 8.

This upload pack intentionally does not apply the later data/claim fixes. Those
are tracked in `paper/scope_edit_dropins/DEFERRED_DATA_CLAIM_FIXES.md`.
