# Scope Edit Drop-ins

These files are writing-only drop-ins for `paper/scope_edit_v0.zip`.
They split the short combined `Background and Related Work` section into:

- `2Background.tex`: replace `sections/2Preliminary.tex` with this content.
- `6RelatedWork.tex`: add before `sections/6Conclusion.tex`.
- `main_input_snippet.tex`: minimal `main.tex` wiring.
- `DEFERRED_DATA_CLAIM_FIXES.md`: later data/claim cleanup checklist for this
  scope draft.

No result tables, figures, data values, bibliography entries, or method claims
are changed here. The sections use citation keys already present in
`scope_edit_v0.zip`.

Temporary compile check against `paper/scope_edit_v0.zip`:

- `tectonic main.tex` completes with only underfull box warnings.
- With these two drop-ins, the conclusion begins on main-text page 8.
- References continue after the conclusion, and the appendix begins after the
  references.
