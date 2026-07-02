---
title: Adaptive-RAG (NAACL 2024) + When Not to Trust LMs / PopQA (ACL 2023)
type: source
tags: [adaptive-retrieval, routing, query-complexity, entity-popularity, rag, open-domain-qa]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2403.14403
local: references/adaptive-rag.pdf, references/mallen-popqa.pdf
authors: Jeong et al. (Adaptive-RAG); Mallen et al. (PopQA)
year: 2024 (Adaptive-RAG); 2023 (Mallen)
venue: NAACL 2024; ACL 2023
code: https://github.com/starsuzi/Adaptive-RAG ; https://github.com/AlexTMallen/adaptive-retrieval
---

# Adaptive-RAG + Mallen et al. (selective/adaptive retrieval lineage)

## TL;DR

Two papers that define the "decide per query how much retrieval machinery to spend" lineage. Mallen et al. (ACL 2023) show LM parametric memory tracks entity popularity (Wikipedia page views), that retrieval helps the long tail but actively hurts popular questions, and propose Adaptive Retrieval: a binary retrieve-vs-not decision from a per-relation popularity threshold. Adaptive-RAG (NAACL 2024) generalizes the routing target from binary to three tiers — no retrieval / single-step RAG / multi-step iterative RAG — routed by a trained T5-Large query-complexity classifier with silver labels harvested from model outcomes plus dataset inductive bias. Both route retrieval *effort*; neither routes query *expansion*, which is our axis.

## Key claims / numbers

- **Mallen: routing signal = subject-entity popularity** (Wikipedia monthly page views) plus relationship type; retrieval is triggered only when popularity falls below a per-relationship-type threshold tuned on a dev set. No trained router. *our-relevance:* a query-only, zero-training routing signal — the ancestor of our unsupervised [[qpp]] signals; but page-view popularity has no analogue for statutes/case passages, so legal RAG needs a different signal (C3-adjacent contextualization).
- **Mallen: scaling doesn't fix the long tail** — on the 4,000 least popular PopQA questions, GPT-Neo 20B gets 16% and GPT-3 davinci-003 only 19%; scaling gains concentrate on high-popularity questions. *our-relevance:* independent evidence that some question regimes are structurally retrieval-dependent, backing our [[weak-vs-strong-query-regime]] framing.
- **Mallen: retrieval can hurt** — for 10% of PopQA questions, retrieval flips a correct parametric answer to wrong; on exactly those questions Contriever recall@1 is 0.14 vs 0.42 overall (their Table 1). *our-relevance:* the canonical "bad retrieval misleads a model that already knew the answer" result — the same mechanism as our HousingQA SCOPE regression and [[query-drift]], and it grounds C10 (regression is a known failure class, not "parity").
- **Mallen: Adaptive Retrieval wins on accuracy AND cost** — up to +10% on PopQA (best config GPT-3 davinci-003 with GenRead+Contriever adaptively: 46.5%, +5.3 over any non-adaptive), while davinci-003 retrieves on only ~40% of questions; API-cost reduction ~half in the headline setting, 15% even on long-tail-heavy EntityQuestions. *our-relevance:* template for selling [[regime-routing]] as accuracy-preserving cost/robustness control rather than a raw-accuracy headline (C5, C9 framing).
- **Adaptive-RAG: routing signal = trained query-complexity classifier** (T5-Large, 770M) over three labels: A = no retrieval, B = single-step retrieve-then-read, C = multi-step iterative (IRCoT-style). Labels are auto-collected: silver labels from which strategy answered correctly (ties go to the simpler strategy) plus dataset inductive bias (single-hop datasets → B, multi-hop → C). *our-relevance:* the supervised end of the routing-signal spectrum; our QPP routing deliberately avoids the trained-classifier requirement (C3 lineage positioning).
- **Adaptive-RAG: accuracy ≈ multi-step at a fraction of the cost** — FLAN-T5-XL averaged over 6 QA datasets: EM 37.17 / F1 46.94 at 2.17 steps and 3.60 relative time vs always-multi-step EM 39.00 at 4.69 steps / 8.81 time; GPT-3.5: 37.97 vs 38.13 EM at 1.46 vs 3.33 time. *our-relevance:* honest "flat accuracy, big efficiency" framing that reviewers accepted at NAACL — directly usable posture for our routing claims against C5/C9.
- **Adaptive-RAG: Mallen-style binary routing transfers poorly to mixed-complexity workloads** — their re-implementation of popularity-threshold Adaptive Retrieval scores EM 23.87 (FLAN-T5-XL average) vs Adaptive-RAG 37.17, because retrieve-vs-not cannot express "needs multiple hops". *our-relevance:* per-query routing signals are brittle out of their home regime — consonant with our finding that no-gold per-query QPP fails (best WIG-CE tau ≈ −0.11) while regime-level routing works (C11 honesty).
- **Adaptive-RAG: classifier is weak but still useful, and oracle headroom is reported** — classifier accuracy is only 54.52%, yet routing helps; oracle-classifier Adaptive-RAG reaches EM 45.00 vs realized 37.17. Per-label costs: 0.35s (no retrieval) / 3.08s (single) / 27.18s (multi-step) per query. *our-relevance:* the report-oracle-and-realized pattern is exactly the rigor structure a revised SCOPE paper needs for its routing section (C11), and "noisy router can still pay" is the right expectation to set.

## Bearing on the review

- **C3 ("essentially HyDE for legal")**: this lineage lets a revision position SCOPE-with-routing inside an established adaptive-RAG taxonomy while claiming a *distinct routing axis*: Mallen routes retrieve-vs-not, Adaptive-RAG routes retrieval iterations, we route **expand-vs-not** (and expansion pooling). The revision must cite both and state the axis difference explicitly; neither paper touches query expansion or generated pseudo-documents.
- **C9/C10 (parametric knowledge drives answers; regression framed as parity)**: Mallen's Table 1 is the citation that retrieval harming already-correct parametric answers is a known, quantified failure mode (10% of questions, recall@1 0.14). A revision should present the HousingQA regression through this lens — retrieval-quality-conditional harm on strong queries — instead of calling it parity, and should adopt LLM-only as a first-class comparator the way both papers do (no-retrieval is a headline row in each).
- **C5 (marginal gains) / C11 (rigor)**: both papers show the accepted way to sell small accuracy deltas — pair them with cost/latency accounting (steps, time/query, API cost) and an oracle-router upper bound. Our revision's routing section needs: held-out routing evaluation (ours is currently in-sample), realized-vs-oracle numbers, and token accounting that includes the expansion call (the exact C11 complaint).

## Differentiation

- **Different routing axis, honestly narrower evidence**: neither paper decides *whether to generate an expansion*; both decide *whether/how much to retrieve*. Our expand-vs-not routing is not pre-empted. But their per-query routers (popularity heuristic; trained classifier) demonstrably beat random on their benchmarks, whereas our per-query no-gold QPP routing failed (no predictor cleared tau ≥ 0.5) and we only have dataset/regime-level routing, in-sample. They are ahead of us on router quality; we are ahead on the mechanism story ([[geometry-vs-factuality]], CE-affinity movement) which neither paper has.
- **Signal availability**: Mallen's popularity signal requires an entity-linked, page-view-instrumented corpus — unavailable for legal statutes; Adaptive-RAG's classifier requires silver-label harvesting runs per strategy. Our QPP signals (NQC/WIG/SMV, CE margins) are unsupervised and corpus-agnostic, which is the defensible niche, provided we admit their current per-query weakness.
- **Domain**: both are Wikipedia factoid QA (PopQA, EntityQuestions; SQuAD/NQ/TriviaQA/MuSiQue/HotpotQA/2WikiMultiHopQA). No legal, no vocabulary-gap regime. They do not pre-empt the legal weak-query finding; they also give us no cover on legal-NLP grounding (C2 must be answered elsewhere, e.g. [[gure]], [[koblex-parser]]).

## Links

[[scope]], [[hyde]], [[query2doc]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[query-drift]], [[qpp]], [[regime-routing]], [[generated-query-family]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]], [[gure]], [[koblex-parser]], [[scope-paper-2026]]

## Raw source

- references/adaptive-rag.pdf (Jeong et al., Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity, NAACL 2024, arXiv:2403.14403)
- references/mallen-popqa.pdf (Mallen et al., When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories, ACL 2023, arXiv:2212.10511)
