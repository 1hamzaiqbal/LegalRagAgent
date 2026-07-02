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

**Retrieval side — NOW SIGNIFICANCE-TESTED** (2026-07-02 sweep, exact McNemar
+ bootstrap CIs over all full caches:
[retrieval_significance_2026-07-02](../../docs/generated/retrieval_significance_2026-07-02.md),
97/128 pairs significant):
- **BarExamQA (weak-query): SCOPE ≈ HyDE.** NS for 3/4 models (+0.59 p=0.61;
  +1.26 p=0.19; +0.67 p=0.52); only ministral-8b significant (+2.76pp
  p=1.2e-03). The "snap wins BarExam" point-estimate read does not survive
  testing.
- **BEIR (strong-query): SCOPE ≫ HyDE, everywhere.** All 20 dataset×generator
  cells positive, 19/20 significant, +16 to +45pp (SciFact +30.7pp p=5e-17
  Gemma, +45.3pp p=6e-31 DeepSeek; NFCorpus +24 to +39pp; SciDocs +21 to
  +27pp; TREC-COVID +24 to +28pp) — the drift-robustness claim is now
  significance-tested across 4 generators.
- **Housing (intermediate regime): direction flips by model.** State-filtered:
  Gemma **+7.44pp pro-SCOPE** (p=2.4e-33), llama8b +2.61 (p=9e-06), but
  llama70b **−11.83pp pro-HyDE** (p=6.2e-90). Generator identity matters on
  the boundary regime — an open mechanism question.
- Legal-Link-EU / CaseHOLD / SCALR cells: see the full table (LLEU pro-snap
  large where tested; CaseHOLD/SCALR pro-HyDE).

**Robustness side** (the surviving snap claim): strong-query BEIR drift —
HyDE −31.3pp vs SCOPE −12.2pp pooled Hit@5, SCOPE less-drifting on 5/5
datasets and across 4 generators (17–29pp net gap) — [[beir-phase1]],
`docs/generated/beir_phase1b_model_breadth_2026-05-26.md`. Now local and
artifact-backed post fast-forward, but not signoff-gated as an answer row.

**The revision-safe statement** (updated 2026-07-02 after the significance
sweep): snap-conditioning does not change answer accuracy vs HyDE in 13/16
tested pairs, and does not change weak-query retrieval either (BarExamQA ns
3/4 models). **Its one large, replicated, significance-tested advantage is
drift-robustness on strong-query corpora** (+16–45pp Hit@5 over HyDE, 19/20
cells p<0.05, 4 generators × 5 BEIR datasets), with model-dependent behavior
on the intermediate Housing regime. Any stronger claim needs the C12
ablations first.

## Links
[[icml-ai4law-2026-rejection]] (C7) · [[scope]] · [[hyde]] · [[beir-phase1]] ·
[[answer-conversion-gap]] · [signoff_log](../../docs/signoff_log.md)
