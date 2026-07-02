---
title: LegalRagAgent Wiki Index
type: hub
tags: [index]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# LegalRagAgent Wiki — Index

Compounding knowledge base for the legal-RAG / generative-query-expansion
project. **Current north star: understand and exploit *when generated queries
help retrieval and when retrieval helps answers*** — the mechanism
([[geometry-vs-factuality]]), the regime law
([[weak-vs-strong-query-regime]]), and the conversion bottleneck
([[answer-conversion-gap]]) — after the SCOPE method paper was rejected
([[icml-ai4law-2026-rejection]]). See [[WIKI_GUIDE]] for conventions,
[[START_HERE]] for orientation, [[log]] for chronology.

## Hubs
- [[START_HERE]] — orienting read
- [[direction-2026-07]] — direction map for the 2026-07-02 meeting (next steps, candidate anchors)
- [[icml-ai4law-2026-rejection]] — criticism inventory C1–C12 + assessment (the pivot document)

## Concepts
- [[thesis-v2]] — **the working thesis**: generation is a low-margin instrument; answer-conditioning dampens drift; after pooling, selection binds — with pre-stated falsifiable predictions
- [[weak-vs-strong-query-regime]] — the regime law: expansion helps ∝ query–corpus gap
- [[geometry-vs-factuality]] — mechanism + falsification: failures are geometric, not hallucinated
- [[vocabulary-gap]] — the motivating mismatch, and why it's geometric not lexical
- [[query-drift]] — the established name for strong-query expansion failure
- [[qpp]] — query performance prediction; our routing work repositioned
- [[answer-conversion-gap]] — retrieval gains ≠ answer gains; the under-owned space
- [[regime-routing]] — the operational recipe (expand weak; pool strong)
- [[generated-query-family]] — taxonomy: HyDE/Q2D/GAR/LameR/CSQE/ParSeR/GuRE/SCOPE
- [[legal-rag-benchmarks]] — benchmark landscape + sharp edges
- [[expert-judgment-replication]] — candidate new direction (Thinking Machines pattern → legal)

## Methods (ours)
- [[scope]] — the two-call method: what it does, what survives of it
- [[judge-pilot-v0]] — Tinker-trained legal relevance judge vs the CE (Path C v0, running 2026-07-02)

## Sources — reviewer-named prior art & family
- [[koblex-parser]] — **the C6 near-twin** (parametric provisions, EMNLP'25); adjudication: pre-empted on core move, not on mechanism/regime
- [[gure]] — **the C8 miss** (trained legal query rewriter, NLLP'25); long-tail analysis template; feasible baseline
- [[zheng-cslaw]] — our benchmark source; authors already ran legal query expansion; calibrates C5 (our Hit@5 beats their best)
- [[hyde]] · [[query2doc]] — zero-shot ancestors (keep-vs-discard evidence)
- [[lamer-gar]] — answer-conditioned expansion prior art
- [[csqe]] — corpus-steered expansion; collapses on weak-query legal

## Sources — QPP / expansion-failure / RAG-control
- [[weller-drift]] — expansion helps weak, hurts strong (macro precedent)
- [[faggioli-qpp]] — why dense QPP is hard; QPP evaluation protocol
- [[emami-qpp-variant]] — near-twin: QPP selects query variants; punts generation-aware prediction
- [[tian-right-track]] — near-twin: QPP of generated queries vs answer quality
- [[datta-qpp-reliability]] — τ≥0.5 reliability bar; ceilings ≈0.37
- [[adaptive-rag-mallen]] — selective retrieval lineage (retrieve-vs-not)
- [[power-noise-lostmiddle]] — distractor harm + position effects (answer-conversion anchors)

## Sources — legal NLP landscape (discovery)
- [[li2026legalmalr]] — 2026 RL multi-agent statute-query reformulation; crowds the weak-query space, retrieval-only
- [[yoon2025leakage]] — **knowledge-leakage rival explanation** for HyDE-family gains; defines the NLI audit we must run
- [[afane2026laborbench]] — 2026 legal RAG benchmark entrant
- [[guha2023legalbench]] — LegalBench (community task suite)
- [[magesh2024hallucinationfree]] — RAG reliability study of commercial legal tools
- [[jiang2023syllogism]] — legal syllogism prompting (legal-reasoning tradition, C1)
- [[legal-rag-benchmarks-src]] — LegalBench-RAG + Legal RAG Bench pair
- [[thinking-machines-expert-judgment]] — expert-judgment replication pattern (direction inspiration)
- [[qe-survey-2025]] — the organizing QE taxonomy; regime-gating stated as best practice; QPP named as the open gap
- [[reuter2025sac]] — Document-Level Retrieval Mismatch + summary-augmented chunking (corpus-side dual of our lane)
- [[lexpath2026]] — IRAC-guided expansion beats HyDE on Chinese legal; reproduces our answer-conversion gap

## Results (dated, evidence-linked)
- [[affinity-margin-mechanism]] — the pre-registered legal mechanism result (ρ≈0.44)
- [[beir-phase1]] — mechanism replicates 5/5 BEIR; ungated expansion net-negative on strong queries; SCOPE ≫ HyDE robustness
- [[three-retriever-generality]] — mechanism survives gte+CE / BM25 / E5 (0.34/0.35/0.39)
- [[factuality-falsification]] — geometry AUC 0.79–0.94 vs judged factuality 0.55–0.58; C4 rebuttal
- [[qpp-routing-negative]] — per-query routing closed as principled negative (τ≈0.09–0.11)
- [[pooling-regime]] — raw∪SCOPE pooling wins strong/mid, destroys weak; CSQE crossover
- [[musique-cross-domain]] — bridge-recall +15–16pp; pool-structure caveat
- [[snap-vs-hyde-ledger]] — every signed SCOPE-vs-HyDE pair (the C7 evidence)
- [[judge-pilot-v0-results]] — **Tinker-trained judge un-buries the pool**: 20.6% vs CE 3.8% Hit@5, trained > prompted p=1e-04 (2026-07-02)
- [[leakage-audit-barexam]] — **Yoon leakage account rejected, both regimes**: weak-legal lift survives unmatched (+6pp, p=1e-20); BEIR matched rates ~1%, zero leakage-gated help events (2026-07-02)
- [[judge-pilot-housing]] — **strong-regime judge replication**: 55.0% vs CE 38.2% (p=2.5e-23), 96.5% conversion — regime routing superseded (2026-07-02)

## Reviews
- [[icml-ai4law-2026-rejection]] — inventory, assessment, resubmission checklist, meta-lessons

## Anchors into the repo (linked, not duplicated)
- Citation gate: [signoff_log](../docs/signoff_log.md) · Run ledger: [experiments.jsonl](../logs/experiments.jsonl)
- Generated analyses: [docs/generated/](../docs/generated/) · Ideas log: [ideas.md](../ideas.md)
- Lit repositioning (pre-review): [RELATED_WORK_GROUNDING](../paper/submission/RELATED_WORK_GROUNDING.md)
- Submitted paper + reviews: [official_paper_and_review_icml_ai_4_law/](../official_paper_and_review_icml_ai_4_law/)
- Raw sources: `references/` (gitignored; archived at `wustl:/engrfs/tmp/jacobsn/hiqbal_legalrag/references/papers/`)
