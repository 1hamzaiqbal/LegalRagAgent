---
title: Leakage Audit — the weak-query lift survives without gold entailment
type: result
tags: [leakage, yoon, falsification, mechanism, win]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: win (thesis-v2 prediction 1 supported; Yoon account rejected for our setting)
evidence: docs/generated/leakage_audit_barexam_2026-07-02.md, docs/generated/leakage_audit_barexam_2026-07-02_points.jsonl
---

# Yoon-style knowledge-leakage audit, BarExamQA

**Question** (pre-stated as [[thesis-v2]] prediction 1): is the weak-query
retrieval lift of SCOPE-style generation explained by knowledge leakage — the
generator reproducing content entailed by the gold passage
([[yoon2025leakage]], ACL'25 Findings) — or does the lift survive on
unmatched generations, as the geometry account predicts?

**Setup.** 3 exemplar-anchored SCOPE samples/question × 1,192 BarExamQA
questions (Gemma 4 26B, `3scope_raw` generation cache) matched to per-sample
**dense-stage** retrieval (pool cache components; raw component signature
1.43% Hit@5, SCOPE samples 10.2–11.0% — no CE confound). Matcher: per Yoon,
a sample is *matched* if any sentence is entailed by the gold passage
(nli-deberta-v3-base, premise=gold, hypothesis=sentence), τ ∈ {0.7, 0.8, 0.9}.
7,703 NLI pairs on MPS.

**Results** (stable across all τ):

| τ | matched samples | Hit@5 matched (lift) | unmatched samples | Hit@5 unmatched (lift) |
|---|---:|---:|---:|---:|
| 0.7 | 551 (15.4%) | 28.5% (+27.2pp) | 3,025 | 7.3% (**+5.9pp**) |
| 0.9 | 490 (13.7%) | 29.6% (+28.6pp) | 3,086 | 7.6% (**+6.1pp**) |

Strictest stratum — questions where **no** sample contains any gold-entailed
sentence: any-SCOPE-sample Hit@5 **10.5% vs raw 1.5%** (τ=0.9, n=925,
McNemar b/c=88/5, **p=1.1e-20**; τ=0.8: n=902, 84/5, p=1.4e-19).

**Reading.**
1. **The Yoon account fails here**: their finding (unmatched expansion falls
   *below* the no-expansion baseline) predicts ≤0pp lift in the unmatched
   stratum; we measure +6pp (>4× raw), p≈1e-20 at question level. On
   weak-query legal corpora the lift is not a memorization artifact.
2. Matched samples do retrieve far better (+28pp) — leakage-compatible rows
   carry ~45% of the aggregate lift despite being 14% of samples. The honest
   statement: leakage *amplifies* but does not *explain* the effect.
3. Consistent with [[geometry-vs-factuality]]: a generated passage need not
   state the gold rule (entailment) to move the query into the right
   doctrinal region of embedding space; and statutes/rules phrased many ways
   in the corpus mean entailment-to-the-*labeled* gold under-counts semantic
   proximity — which is exactly why the mechanism is geometric, not textual.

**Caveats.** Exemplar-anchored 3SCOPE generations, not the canonical
single-SCOPE texts (never committed; exemplar prompts bias matched-rate UP →
conservative for our conclusion). NLI matcher imperfect below the entailment
threshold (we swept τ to 0.7). This audits retrieval-lift leakage, not answer
train-test contamination.

## BEIR replication (same day — canonical single-passage generations)

SciFact/NFCorpus/SciDocs × {HyDE, SCOPE}, canonical Gemma generation caches,
τ∈{0.8,0.9} ([report](../../docs/generated/leakage_audit_beir_2026-07-02.md)):
- **Matched rates collapse outside legal: 0–7%** (most cells ≤1%) — on
  scientific corpora the generator essentially never produces gold-entailed
  sentences, so expansion outcomes there are all-geometry by construction.
- **help_m = 0 in all six cells**: every expansion-help-over-raw event occurs
  on an *unmatched* generation. Even inside strong-query corpora, the help
  pockets are not leakage-driven.
- SCOPE's drift-robustness (unmatched delta −1.8/−4.4pp ns on
  SciDocs/NFCorpus vs HyDE's −23.5/−35.9pp, p≤1e-29) is likewise orthogonal
  to leakage.

**Combined statement for the paper**: knowledge leakage cannot explain
generative-expansion behavior in either regime — weak-query legal lift
survives at +6pp on the 85% unmatched stratum (p=1e-20), and on strong-query
scientific corpora matched generations are ~1% of rows with zero
leakage-gated help events. Yoon et al.'s fact-verification finding does not
transfer to specialist-corpus retrieval.

**What it changes.** The strongest external threat to the mechanism paper
(Path A) is defused on the flagship dataset — and turned into a positive
section: we can now report the matched/unmatched decomposition the way
Yoon's paper demands, with the opposite outcome on the regime where
generative expansion actually matters.

## Links
[[thesis-v2]] (prediction 1 ✓) · [[yoon2025leakage]] ·
[[geometry-vs-factuality]] · [[weak-vs-strong-query-regime]] ·
[[beir-phase1]] · [[icml-ai4law-2026-rejection]] (C4/C9) ·
[report](../../docs/generated/leakage_audit_barexam_2026-07-02.md)
