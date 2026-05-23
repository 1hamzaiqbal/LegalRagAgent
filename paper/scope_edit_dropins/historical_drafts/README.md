# Historical Draft Sections

Reference material for the scope-edit work: older paper drafts where
Background, Method, and Analysis sections were substantially longer than the
trimmed final submission. Use these to mine prose when expanding
[`../../scope_edit_v0.zip`](../../scope_edit_v0.zip).

Only `sections/*.tex` and `main.tex` are kept here. The full source trees
(figures, tables, .bib, .pdf, _drafting/) live outside the repo at
`~/grad/paper_local_archive/_zz_archive_2026-05-22/` — restore from there if
you need rendered context.

## Section-size comparison (characters, `.tex` source)

|                                          | Abstract | Intro | Prelim | Method | Analysis | Conclusion |
|---|---:|---:|---:|---:|---:|---:|
| **current `scope_edit_v0`**              | ~1500 | 6346 | 1539 | 3720 | 6012 | 834 |
| `overleaf_snap_hyre_20260520_151438`     | 1196 | 6966 | **4782** | **13732** | **10337** | 4162 |
| `overleaf_snap_hyre_20260520_154326`     | 1183 | 6644 | **5083** | **14236** | **10687** | 4071 |
| `overleaf_snap_hyre_20260521_192702`     | 1505 | 4395 | 2035 | 5260 | **7485** | 1634 |
| `root_duplicate_draft_20260520`          | 1701 | 5265 | **4290** | **7323** | **8443** | 2859 |

## Which to mine for which section

- **Background / Preliminary** (currently 1539 chars; needs expansion per
  [`../../EDIT_QUEUE.md`](../../EDIT_QUEUE.md)):
  - Fullest prose is in `overleaf_snap_hyre_20260520_154326/sections/2Preliminary.tex`
    (5083 chars) and `..._20260520_151438/sections/2Preliminary.tex`
    (4782 chars).
  - Alternate framing: `root_duplicate_draft_20260520/sections/2Preliminary.tex`
    (4290 chars).
- **Related Work**: none of these snapshots have a standalone
  `6RelatedWork.tex`; older Related material is embedded in `Preliminary` or
  `Appendix`. Cross-reference with the current
  [`../6RelatedWork.tex`](../6RelatedWork.tex) drop-in.
- **Method**: heavyweight versions in the May 20 snapshots (~14K chars) walk
  through the two-call algorithm in more detail; useful if the current 3720-char
  version reads too sparse.
- **Analysis**: the May 20 snapshots have a 10K-char Analysis vs the current
  6012-char; useful for restoring claim-by-claim discussion that may have been
  cut in the damage-control pass.

## Snapshot provenance

- `root_duplicate_draft_20260520/` — extracted from
  `_zz_archive_2026-05-22/archive_pre_current_icml_2026-05-20/root_duplicate_draft/`.
  This is the pre-cleanup `paper/` root draft from before sources were
  consolidated under `snap_hyre_2025_05_18/`.
- `overleaf_snap_hyre_*` — extracted from
  `_zz_archive_2026-05-22/Toward_Adaptive_HyRE_*/older_versions/`. These are
  timestamped Overleaf exports from May 20-21, 2026, picked to span the range
  from heaviest content (May 20) through the May 21 trim.
- Three additional May 21 snapshots in the original tree
  (`_20260521_091545`, `_115917`, `_125408`) and the near-final
  `_20260522_021649` are excluded here as near-duplicates of the cut-down
  scope_edit_v0 state.
