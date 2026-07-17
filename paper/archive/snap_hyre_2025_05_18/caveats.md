# Caveats & soft-edit candidates

Suggestions I held back from the paper at your direction (no softening). These
are written so you can scan them and accept/reject one by one. None of them is
applied to `sections/` or `tables/`.

## 1. Per-model winner asymmetry on BarExamQA

The intro and Section 5.1 frame Snap-HyRE as improving over raw-question RAG on
all three model sizes (true). But Snap-HyRE only beats every non-oracle
baseline (incl. `llm_only` and `rag_hyde`) on 1 of 3 BarExamQA model slices:

- 8B: best non-oracle is `llm_only` at 57.3; Snap-HyRE 56.9.
- 26B: Snap-HyRE wins at 82.0.
- 70B: HyDE wins at 80.2; Snap-HyRE 79.7.

Optional softening (not applied): add a half sentence "best non-oracle on 26B,
within 0.5 pp of the per-slice best on 8B/70B." Table 6 already shows this.

## 2. MRR figure annotation precision

Section 5.2 currently says `+4.8pp MRR@5`. The unrounded value is +4.77 pp;
Figure 2 prints `+4.8`. Either is defensible.

## 3. Section 5.4 "Cost and Context"

Snap-HyRE's lower answer-stage token count (1,762 vs 2,634) is mechanical — its
retrieved passages are simply shorter — not a method efficiency claim. The
section reads as cost-positive but the comparison hides the first-call
generation overhead (mentioned, but pushed to Table 9).

Options if you want to reframe:

- Keep as-is (current).
- Move to appendix and replace in main with a one-liner pointer to Table 9.
- Reframe as a context-density statement: Snap-HyRE answers from a tighter
  evidence window without losing accuracy.

## 4. Table 5 "Result anatomy" prose columns

`Raw evidence regime` and `Interpretation` are full sentences inside table
cells. This is information-dense but visually heavier than the surrounding
tables.

Options:

- Keep as-is.
- Drop the two prose columns and absorb their content into the surrounding
  appendix paragraph.

## 5. Table 11 (top-k retrieval) row density

16 rows × 9 columns at `\scriptsize` is borderline at print size. Means rows
plus per-model rows triple the row count.

Options:

- Keep as-is (full transparency).
- Show means + Snap-HyRE per model; drop per-model HyDE rows; HyDE means stay.

## 6. Worked example promotion

`Table 12` (worked example) is the most concrete artifact in the paper and
directly anchors Section 5.3. Currently lives in `A.4` appendix.

Option: promote into main paper, half-column, immediately after Section 5.3.

## 7. Table 2 wide layout

`current_answer_matrix.tex` has 8 numeric columns + the missing-Gemma Housing
column. It uses `\scriptsize` + `\resizebox`, which still leaves it visually
busy.

Options:

- Keep as-is.
- Split into two side-by-side small tables (BarExam, Housing) at `\footnotesize`
  without resizebox.

## 8. ICML preprint flag

`main.tex` line 15 still uses `\usepackage[preprint]{icml2026}`. Switch to
the anonymous / accepted flag before any blind workshop submission.

## 9. Reference list

7 citations. Worth considering for completeness (not softening):

- Zheng et al. on BarExamQA/HousingQA construction (you cite the benchmark
  source elsewhere but not the dataset paper).
- A multi-query / RAG-Fusion reference to position `rag_rewrite` as a control.
- A reranker reference for `ms-marco-MiniLM-L-6-v2` if a reader cares about
  the retrieval stack.

## 10. Figure 3 (top-k curves) y-scale convention

The two HousingQA panels use 0–60 (Hit) and 0–30 (MRR). BarExamQA panels use
0–20 and 0–10. The axes are independent which is honest, but a quick scan
might over-read magnitude differences.

Option: add an explicit "note: y-scales differ by panel" line in the caption.

## 11. langlin feedback items already absorbed

The feedback triage at the bottom of `paper/main_langlin_feedback.tex` lists
the items already addressed in the current draft. The unaddressed ones are:

- **Figure 1 redraw**: partially done — caption mentions proximate-cause
  example, and the orange box now carries an explicit "(not in Call 2)"
  clarifier (this round). Could still rebuild with a hand-drawn / vector look
  if you want a more polished aesthetic.
- **Algorithm pseudocode**: the current paper drops the explicit algorithm
  block in favor of equations + the three-step operational description in
  Section 3.3. Langlin's note was that the original algorithm block read like
  prose rather than pseudocode; the current treatment sidesteps that.

The other feedback items (HyDE background in abstract/intro, name explanation
on first use, "ladder" terminology removed, wordy method table replaced with
itemized list, benchmark intros shortened, provider details deemphasized,
redundant figures removed, conclusion/limitations split) are all live in the
current draft.
