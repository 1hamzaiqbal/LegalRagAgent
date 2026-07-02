---
title: The Power of Noise + Lost in the Middle (SIGIR 2024 / TACL 2024)
type: source
tags: [rag, context-utilization, distractors, position-bias, answer-conversion, reader-bottleneck]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2401.14887 ; https://arxiv.org/abs/2307.03172
local: references/power-noise-lostmiddle.pdf (= references/power-of-noise.pdf) ; references/lost-in-the-middle.pdf
authors: Cuconasu et al. (Power of Noise); Liu et al. (Lost in the Middle)
year: 2024
venue: SIGIR 2024 (Power of Noise); TACL 2024 (Lost in the Middle)
code: https://github.com/florin-git/The-Power-of-Noise ; https://nelsonliu.me/papers/lost-in-the-middle
---

# TL;DR

Two canonical reader-side RAG papers. **The Power of Noise** (Cuconasu et al., SIGIR '24; NQ-open, ~21M Wikipedia passages, Contriever + BM25, Llama2-7B/Falcon-7B/Phi-2/MPT-7B) shows that *distracting* documents — the retriever's highest-scoring passages that do not contain the answer — actively damage answer accuracy (a single distractor costs up to -25%; stacks of them up to -67%), while truly *random* documents can paradoxically help (up to +35-36%). **Lost in the Middle** (Liu et al., TACL '24; multi-document QA on NQ-open plus synthetic key-value retrieval; GPT-3.5-Turbo/Claude-1.3/MPT-30B/LongChat-13B) shows a U-shaped position curve — models use evidence at the start (primacy) or end (recency) of context and degrade sharply when the answer passage sits in the middle — and that reader accuracy saturates long before retriever recall does. Together they are the standard citations for why retrieval gains fail to convert into answer gains.

# Key claims / numbers

- **Distractor harm (PoN, Table 1)**: adding one top-scoring non-answer ("distracting") document to the gold document cuts accuracy sharply, with peaks of 0.24 absolute (-25%); increasing distractors degrades accuracy by more than 0.38 (-67%) in some cases (e.g., Llama2 gold-only 0.5642 → 0.2643 with 8 far-position distractors). *our-relevance:* this is the mechanism behind our golden-plus-neighbors dilution (CaseHOLD 97.5% gold-only → 79.4% gold+4 neighbors; SCALR 93.5% → 83.0%) and grounds C9/C10 — SCOPE's extra retrieved neighbors are exactly "distracting documents".
- **Better retrievers don't fix it (PoN §5.1)**: swapping in ADORE, a hard-negative-trained dense retriever, still yields 0.4068/0.3815/0.3626 with 1/2/4 distractors vs 0.5642 gold-only; the authors conclude relevance-vs-distraction "cannot be mitigated simply by changing the dense retrieval method." *our-relevance:* directly supports our answer-conversion-gap framing against C9 — improving retrieval (SCOPE's 8x Hit@5 lift on BarExamQA) is not sufficient for answer lift.
- **Random noise helps (PoN Tables 2-4)**: random Wikipedia documents added near the query improve accuracy up to +0.08 (+36%) in the oracle setting and up to +0.07 (+35%) on top of 4 retrieved documents; even Reddit text (+9%) and nonsense random-word sentences help; attention entropy shows a 3x increase (entropy-collapse hypothesis). Best practice they propose: retrieve 3-5 documents, pad with random. *our-relevance:* semantic similarity without the answer is the harmful category — quantifies why SCOPE's query drift on strong-query HousingQA (retrieving plausible-but-wrong statutes) regresses answers (C10) where sheer irrelevance would not.
- **U-shaped position curve (LitM Fig 1, 5)**: accuracy is highest when the answer document is first or last in context and degrades >20% mid-context; in 20- and 30-document settings GPT-3.5-Turbo's mid-position performance falls *below* its 56.1% closed-book score. Oracle (gold-only) accuracy is far higher (GPT-3.5-Turbo 88.3%, Claude-1.3 76.1%). *our-relevance:* independent, well-known evidence that context utilization, not retrieval, is the binding constraint (C9); also motivates auditing gold position within our k=5 rerank order.
- **Reader saturation (LitM §5, Fig 11)**: reader accuracy saturates long before Contriever recall — going from 20 to 50 retrieved documents adds only ~1.5% (GPT-3.5-Turbo) / ~1% (Claude-1.3); recommendation: rerank relevant info toward the start or truncate the ranked list. *our-relevance:* the canonical statement of the retrieval-generation gap; our "8x retrieval lift moves average answers only 72.3→72.9" is a legal-domain instance and must be framed as such (C5, C9).
- **Robustness across settings (LitM §4)**: the U-shape persists in base (non-instruction-tuned) models, appears in Llama-2 13B/70B (7B is recency-only), and extended-context models are no better within shared lengths; query-aware contextualization fixes only synthetic key-value retrieval, not multi-document QA. *our-relevance:* the bottleneck is architectural/behavioral, not fixable by our answer-prompt tweaks alone — supports arguing that answer-side conversion is a distinct research axis from SCOPE's retrieval-side contribution.

# Bearing on the review

- **C9 (headline vs weakest baseline; answers driven by parametric knowledge)**: these papers let a revision *predict* rather than excuse the near-zero LLM-only deltas: when gold recall is low and retrieved neighbors are distractors, RAG should underperform closed-book (LitM shows mid-context RAG below closed-book; PoN shows distractors dragging below gold-only and below no-noise baselines). A revised paper must present retrieval exposure (Hit@k/MRR) and answer accuracy as two explicitly decoupled outcomes, cite these two works as the established mechanism for the decoupling, and report conditional accuracy (gold-retrieved-but-wrong; gold-missing-but-correct), which our harness already logs.
- **C10 (HousingQA regression framed as parity)**: reframe the -3pp regression as distractor harm under query drift, quantitatively consistent with PoN's single-distractor -25% ceiling effect; stop calling it parity.
- **C5 (marginal gains)**: the reader-saturation result gives the honest interpretation — SCOPE's contribution is at the retrieval stage, and the small answer deltas reflect a known, domain-general conversion bottleneck; the revision should say this with these citations instead of implying the answer deltas validate the method.
- **Golden-plus-neighbors ablation**: our CaseHOLD/SCALR/Housing gold-vs-gold+neighbors drops are a replication of PoN's distractor finding in the legal domain; the revision should present them as such (a strength: legal corpora with 686K-1.8M passages vs their Wikipedia setup), not as a novel observation.
- Secondary: **C11** — both papers mark statistical significance (PoN uses Wilcoxon p<0.01 per cell); our revision needs comparable per-comparison tests, including SCOPE-vs-HyDE (C7).

# Differentiation

These are reader/context-side papers on general-domain Wikipedia QA with *controlled oracle* injections; neither studies query formulation, generated queries, or legal corpora. SCOPE sits on the query side (HyDE-family expansion under vocabulary gap) and treats answer accuracy as downstream. We are not pre-empted on method. We *are* pre-empted on two findings we partially re-derived: (1) semantically-similar non-gold context hurts answers (our neighbor dilution = PoN distractor harm), and (2) retrieval improvements saturate at the reader (our 8x-lift/flat-answers = LitM reader saturation). Any revision must cite both and claim only the legal-domain, weak-query-regime instantiation plus the per-query geometric mechanism as ours. One nuance we add that they do not: our dilution appears even at k=5 with gold guaranteed present (PoN's far/mid/near analysis is per-position; LitM's is per-position at larger k), and our regime split ties distractor harm to *query strength* rather than to context length or position.

# Links

- [[answer-conversion-gap]] — both papers are the primary anchors for this page
- [[scope]], [[hyde]], [[generated-query-family]] — SCOPE produces the queries whose retrieved neighbors become PoN-style distractors
- [[query-drift]] — drift manifests as distracting documents on strong-query regimes
- [[weak-vs-strong-query-regime]], [[regime-routing]] — where distractor harm bites vs where retrieval lift matters
- [[vocabulary-gap]], [[qpp]] — the retrieval-side story these papers deliberately do not cover
- [[geometry-vs-factuality]] — our mechanism layer beneath the conversion gap
- [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]] — where C5/C9/C10 originated
- Sibling sources: [[parser-koblex]] (C6 prior art), [[gure]] (C8 legal-IR rewriting), if/when ingested

# Raw source

- references/power-noise-lostmiddle.pdf (primary, The Power of Noise; duplicate copy at references/power-of-noise.pdf)
- references/lost-in-the-middle.pdf (secondary, Lost in the Middle)
