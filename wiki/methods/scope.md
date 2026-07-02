---
title: SCOPE (Snap-answer COnditioned Pseudo-document Embedding)
type: method
tags: [scope, generated-query, legal-rag, hyde-family]
created: 2026-07-02
updated: 2026-07-02
status: maintained
code: eval/eval_harness.py (mode `snap_hyre`; legacy alias `rag_snap_hyde_2call`)
impl_status: validated
---

# SCOPE — our two-call generated-query method

**What it does.** One legal question → two LLM calls with retrieval between:

1. **Call 1** (same model): generate a *private* snap answer $a_0$ (immediate
   parametric intuition, never shown later) **and** a pseudo-document $p$
   written in the style of the formal legal authority that would support $a_0$.
2. **Retrieve**: embed **only $p$** (gte-large-en-v1.5 → Chroma dense top-k;
   cross-encoder ms-marco-MiniLM-L-6-v2 rerank; k=5). The raw question is *not*
   part of the retrieval query (unlike [[hyde]], which averages the query
   vector in; unlike [[query2doc]], which concatenates).
3. **Call 2**: answer from original question + retrieved evidence; $a_0$ and
   $p$ are discarded (the "confirmation-bias guardrail", Eq. 7 of
   [[scope-paper-2026]] — asserted, not yet ablated: criticism C12 in
   [[icml-ai4law-2026-rejection]]).

**Design intent**: mimic the practitioner loop *intuition → find controlling
authority → judge evidence on its own merits*. The snap draft is an
information bottleneck choosing *what kind* of authority to look like.

## Where it stands after the rejection + generalization sprint

- **As a method claim ("beats HyDE"): dead.** Answer-side snap-vs-HyDE is
  non-significant in 13 of 16 McNemar-tested full-N pairs; where it is
  significant the direction is dataset-dependent (pro-snap Legal-Link-EU Gemma
  +4.17pp p=0.004; pro-HyDE HousingQA-70B up to −6.45pp p=1.4e-28) — see
  [[snap-vs-hyde-ledger]]. Retrieval-side direction flips by dataset and has
  *no* significance tests anywhere yet. Internal grounding
  ([RELATED_WORK_GROUNDING](../../paper/submission/RELATED_WORK_GROUNDING.md))
  and reviewer oSUu independently converge on this.
- **As a weak-query-regime tool: real.** On [[weak-vs-strong-query-regime]]
  weak ends (BarExamQA, MedQA-textbook style), SCOPE-style expansion lifts
  gold exposure ~8× over raw queries and is the *only* family that works where
  corpus-steered expansion ([[csqe]]) collapses (needs real text to steer with).
- **As a robustness point: promising.** On strong-query BEIR, HyDE collapses
  (−31% Hit@5) while SCOPE degrades mildly (−12%) — low [[query-drift]]
  variance is the one snap-conditioning benefit that survived adversarial
  checks ([[beir-phase1]]).
- **As a mechanism vehicle: the actual contribution.** SCOPE benefit is
  predicted per-query by CE/embedding affinity movement toward gold
  (ρ≈0.34–0.45 across 7 datasets × 3 retrievers), and its failures are
  geometric, not factual ([[geometry-vs-factuality]]).

## Validation gate
Signed rows only via [signoff_log](../../docs/signoff_log.md);
`NO_SILENT_FALLBACK=1`; strict cache replay
(`scripts/build_generation_cache.py`, `scripts/build_retrieval_cache.py`).
Snap block must contain an exact final answer line; the snap letter is always
stripped from anything the final agent sees (`tests/test_sanitizer.py` —
showing it always hurt).

## Links
[[scope-paper-2026]] · [[hyde]] · [[query2doc]] · [[lamer-gar]] ·
[[koblex-parser]] (near-twin) · [[generated-query-family]] ·
[[weak-vs-strong-query-regime]] · [[regime-routing]] ·
[[answer-conversion-gap]] · [[icml-ai4law-2026-rejection]]
