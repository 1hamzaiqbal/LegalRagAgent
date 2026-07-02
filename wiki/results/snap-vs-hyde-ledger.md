---
title: Snap-vs-HyDE Ledger (the C7 evidence)
type: result
tags: [scope, hyde, significance, c7]
created: 2026-07-02
updated: 2026-07-02
date: compiled 2026-07-02 (rows 2026-05-15 .. 2026-05-26)
verdict: answer-equivalent in most cells; dataset-dependent where significant
evidence: docs/signoff_log.md, docs/generated/medqa_usmle_widening_2026-05-26.md
---

# Every signed SCOPE(snap_hyre)-vs-HyDE(rag_hyde) comparison

Reviewer oSUu (C7): "No significance test is reported for any
SCOPE-versus-HyDE comparison." **The tests existed** — 16 McNemar-tested
full-N answer pairs in [signoff_log](../../docs/signoff_log.md) (+ a CI'd
MedQA q200 pair) — the paper just never included them. Compiled verdict
(verified by adversarial pass 2026-07-02):

**Answer side: 13/16 pairs non-significant** (|Δ| ≤ 2pp, p ≥ 0.10). The three
significant cells split by dataset:
- **Pro-snap**: Legal-Link-EU Gemma 26B **+4.17pp, p=0.00361** — the single
  strongest pro-snap answer datum we own.
- **Pro-HyDE**: HousingQA Llama-70B **−6.45pp p=1.4e-28** (unfiltered),
  **−2.57pp p=1.7e-06** (state-filtered) — *worse* than the paper's "parity"
  framing; SCOPE is significantly the weaker expansion there.

**Retrieval side (Hit@5, point estimates — no significance tests exist
anywhere yet; do not cite as tested):** snap wins BarExamQA 3/3 models
(0.083→0.095, 0.114→0.121, 0.105→0.111) and Legal-Link-EU 2/3 hugely
(0.49→0.68 Gemma, 0.55→0.72 70B — though the 70B case converts to zero answer
gain, the [[answer-conversion-gap]] in miniature); HyDE wins CaseHOLD
(0.512 vs 0.450), SCALR-70B (0.615 vs 0.552), MASLegalBench 3/3, Housing-70B.

**Robustness side** (the surviving snap claim): strong-query BEIR drift —
HyDE −31.3pp vs SCOPE −12.2pp pooled Hit@5, SCOPE less-drifting on 5/5
datasets and across 4 generators (17–29pp net gap) — [[beir-phase1]],
`docs/generated/beir_phase1b_model_breadth_2026-05-26.md`. Now local and
artifact-backed post fast-forward, but not signoff-gated as an answer row.

**The revision-safe statement** (use this wording): snap-conditioning does not
change answer accuracy vs HyDE in 13/16 tested pairs; where it does, direction
is dataset-dependent (+4.2pp LLEU/Gemma, −2.6 to −6.5pp Housing/70B). Its
measurable value is retrieval-side and regime-dependent (consistent small
weak-query gains; large citation-linkage gains) plus drift-robustness on
strong queries. Any stronger claim needs the C12 ablations first.

## Links
[[icml-ai4law-2026-rejection]] (C7) · [[scope]] · [[hyde]] · [[beir-phase1]] ·
[[answer-conversion-gap]] · [signoff_log](../../docs/signoff_log.md)
