# Related-Work Grounding & Repositioning (2026-05-26)

From three parallel literature scans (QPP / generative expansion / adaptive-RAG).
Purpose: stop claiming as novel what IR/NLP already named, cite the right work,
and identify pre-emption risks to engage. Verify each cite before use.

## TL;DR — what our "findings" actually are

| Our finding / idea | Established name | Must-cite | Our defensible angle |
|---|---|---|---|
| No-gold raw-retrieval confidence metric (top-1 score, score spread/entropy, bi-encoder sim) | **Query Performance Prediction (QPP)**, unsupervised post-retrieval | NQC (Shtok'09), WIG (Zhou'07), SMV (Tao'14); dense: Vlachou&Macdonald'24; survey Faggioli'23 (2305.10923) | Instantiate QPP on cross-encoder/dense scores for a HyDE-gate; validate it transfers (it often doesn't on dense — Faggioli 2302.09947) |
| SCOPE method | **HyDE-family generative query expansion** | HyDE (Gao, 2212.10496); Query2Doc (Wang, 2303.07678) | Narrow novelty: *private parametric snap-answer → authority-style passage → REPLACES query → answer-blind generation* + mechanism analysis |
| "SCOPE backfires on strong queries / breaks raw's hit" | **Query drift / topic drift** (classic PRF) + retriever-strength effect | Weller'24 "When do Gen Query/Doc Expansions Fail?" (2309.08541) | We localize it to *per-query* strong queries with a clean predictor (CE-sim-to-gold gain, ρ≈0.44), ruling out perplexity/OOV |
| Route SCOPE on/off by difficulty | **Selective query expansion / selective query processing** (gated by QPP) | classic Cronen-Townsend'04; Adaptive-RAG (2403.14403); Mallen'23 (2212.10511) | Label-free, inference-time, expand-vs-not (not depth/retrieve-vs-not), framed by a weak/strong-query regime theory |
| "Answer-conversion bottleneck" | **retrieval–generation gap** / distractor harm / context-utilization bottleneck | Power of Noise (2401.14887); Lost-in-the-Middle (2307.03172) | OK to use our term as a synthesis label, but anchor to these |
| Snap-answer / answer-conditioned pseudo-doc | answer-conditioned expansion already exists | LameR (2304.14233); GAR (2009.08553) | Ours is a *private, pre-retrieval parametric* draft, discarded before final answer |

## Pre-emption risks to engage head-on (the near-twins)
1. **Emami et al. 2026, "Can QPP Choose the Right Query Variant?" (arXiv 2604.22661)** — label-free QPP selection among LLM query variants for RAG, and independently names the retrieval-vs-answer objective gap. *Closest pre-emption of our routing idea.* Differentiate: we do expand-vs-don't (binary) gated by a **vocabulary-gap regime theory**, in specialist domains (legal/USMLE).
2. **Tian et al. 2025, "Am I on the Right Track?" (arXiv 2507.10411)** — unsupervised QPP of LLM-generated queries predicts RAG answer quality. Differentiate: we QPP the *raw* query to decide whether to invoke SCOPE.
3. **Weller et al. 2024 (2309.08541)** — expansion helps weak retrievers, hurts strong ones (macro). We localize to per-query + give the mechanism.

## Hard constraints this puts on the paper
- **Reframe the metric as QPP**; report NQC/WIG/SMV + a dense-coherence predictor as baselines, not homemade features. Report correlation against the **Kendall τ ≥ 0.5** reliability bar for selective processing (Datta'25, 2504.01101).
- **Validate dense-QPP transfer** — classic predictors degrade on dense/neural scores (Faggioli'23). Our CE/bi-encoder setting needs the correlation shown, not assumed.
- **Narrow SCOPE's novelty claim** to the composition + mechanism + regime theory; HyDE/Query2Doc/LameR/GAR already cover the parts.
- **Name the strong-query failure "query drift"** and cite Weller'24.

## Must-cite shortlist (start here)
HyDE 2212.10496 · Query2Doc 2303.07678 · Weller'24 2309.08541 · NQC (Shtok'09) ·
Faggioli'23 2302.09947 · Adaptive-RAG 2403.14403 · Mallen'23 2212.10511 ·
Power of Noise 2401.14887 · Lost-in-the-Middle 2307.03172 · Emami'26 2604.22661 (engage).

## Read-first (3)
1. Faggioli'23 "Are We There Yet?" (2302.09947) — why dense QPP is hard.
2. Emami'26 (2604.22661) — the near-twin to distinguish from.
3. Weller'24 (2309.08541) — the strong-query failure, established.

> Caveat: the 2026/2603/2604-series arXiv items are very recent — verify
> peer-review/venue status before treating them as load-bearing anchors.

---

# Deep-read gap analysis (2026-05-26) — two full-text reads converge

## Verdict on the three candidate anchors
- **(A) MECHANISM — per-query affinity-margin: STRONG, ~85–90% OPEN. Recommended anchor.**
  The macro phenomenon "expansion helps weak / hurts strong retrievers" is TAKEN
  (Weller'24 2309.08541; classical Collins-Thompson'09 "expansion failure"/Robustness
  Index). What is OPEN: a *per-query, label-free, geometric* mechanism = the
  **gold-vs-distractor affinity margin** `margin = aff(q,gold) − max_d aff(q,distractor)`.
  No read paper measures expansion gain against this margin; Weller is macro +
  needs target-domain labels + 30-example error analysis; Collins-Thompson's "risk" is
  term-weight variance with *uniform* dampening (no per-query predictor); Cronen-Townsend
  clarity predicts query *difficulty* not expansion headroom; GAR's gold-overlap is macro
  lexical ROUGE. CSQE'24 attributes expansion failure to LLM *hallucination/knowledge gap*
  — which our data FALSIFIES (OOV<1%, perplexity ruled out). Cross-domain (legal+medical)
  + confound control is the wedge.
- **(B) METHOD — snap-answer beats HyDE: WEAK/RISKY. Use as a probe, not a claim.**
  Answer-conditioning is heavily pre-empted (HyDE, Query2Doc, LameR, GAR). SCOPE's only
  mechanical novelty is *private parametric draft → discard → embed passage only*. But the
  DISCARD half is the contrarian/weaker design in the literature: Query2Doc Table 4
  (pseudo-doc-only 48.7 vs query+doc 66.2 nDCG) and GAR ("answer alone retrieves false
  positives") show keep+concat > discard. A powered ablation could land null/negative.
  Reframe snap/discard-vs-keep as a *lever that varies the margin* (a mechanism test).
- **(C) ROUTING / selective expansion: TAKEN + empirically DEAD for us. Drop as primary.**
  Concept is ~20 yrs old (Amati'04 selective QE via clarity); modern RAG instance is
  Emami'26 (2604.22661, label-free QPP variant-selection); negative result that such
  routing is marginal/non-generalizing is Datta'25 (2504.01101, τ ceilings ~0.37);
  Faggioli'23 (2302.09947) explains dense-QPP is hardest. Our τ≈0.11 < 0.37. Reframe Q1 as
  a *principled negative* ("why no cheap per-query router works here") + the regime rule.

## The most under-owned space (second pillar)
The **answer-conversion gap** — retrieval gains not converting to answer gains — is only
OBSERVED, never modeled. Emami'26 documents it sharply (NQC r=0.33 with nDCG@5 but ≈0 with
answer nuggets) and explicitly punts "generation-aware predictors / answer-grounding
potential" to future work; Datta sees the same; Tian'25 finds only weak QPP↔answer signal.
Our retrieval-positive/answer-flat SCOPE rows live exactly here.

## Must out-design / distinguish from
1. **Weller'24 (2309.08541)** — owns the macro phenomenon; our wedge = per-query, label-free,
   geometric margin, confounds controlled, 2nd domain.
2. **Emami'26 (2604.22661)** — near-twin on selection; our distinctions = binary expand-vs-not
   (not 30-variant), specialist legal/medical (not general web), a regime/margin THEORY (not a
   QPP bake-off), and a rigorous answer-conversion analysis (which they punt).

## Recommended anchor (provisional, pending sign-off)
**"An affinity-margin account of when generative query expansion helps — and why its
retrieval gains don't convert to answers."** Pillar 1 (retrieval): benefit governed by the
per-query gold-vs-distractor margin; crossover at margin≈0; rules out vocab/perplexity/OOV/
knowledge-gap/retriever-identity. Pillar 2 (answer): characterize the margin→answer decoupling.
Snap/HyDE/discard-vs-keep = mechanism probes. Routing = principled negative + regime rule.

## Design instruments (rigor)
- Operationalize `margin = CE(q,gold) − max_d∈retrieved CE(q,distractor)`; test benefit ∝ Δmargin, crossover at 0.
- BREAK the domain/format/length confound: stratify by measured raw margin WITHIN each dataset
  (show the help→hurt crossover holds within-dataset, not just across); optionally construct
  strong-query variants of the SAME items.
- Pre-state falsifiable predictions: (i) crossover at margin≈0; (ii) Δmargin monotone with benefit;
  (iii) margin dominates domain/format/length/OOV in a joint model; (iv) answer-gain ∝ margin-gain is weak.
- Powered: full N, McNemar, bootstrap CIs; report Collins-Thompson **Robustness Index** + risk-reward curves.
- Baselines: raw · HyDE (keep-query, Eq.8) · Query2Doc (keep+concat) · SCOPE (discard) — keep-vs-discard tests the lever.
