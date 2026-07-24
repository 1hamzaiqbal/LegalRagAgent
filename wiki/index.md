---
title: LegalRagAgent Wiki Index
type: hub
tags: [index]
created: 2026-07-02
updated: 2026-07-22
status: maintained
---

# LegalRagAgent Wiki — Index

Compounding knowledge base for the legal-RAG project. **Current north star:
reader-conditioned marginal evidence-set utility under search cost**
([[three-dial]]), with [[opd-distillation]] as a gated implementation route.
The SCOPE method is historical ([[scope-old]]). Start with
[[research-state-2026-07-17]], then [[literature/index]] and
[[coverage-audit-2026-07-17]]. See [[WIKI_GUIDE]] for conventions,
[[START_HERE]] for orientation, and [[log]] for chronology.

## Hubs
- [[START_HERE]] — orienting read
- [[research-state-2026-07-17]] — current local/EIT/literature synthesis and decision gates
- [[three-dial]] — primary research track
- [[opd-distillation]] — gated skill-distillation track
- [[opd-math-source-transfer]] — M/O teacher-training versus student-rollout
  source matrix with task reward and an exact-token sampled reverse-KL path
- [[opd-data-value-design-2026-07-24]] — gated study of which data is worth
  distilling, with observer-conditioned and epiplexity-compatible diagnostics
- [[opd-math-eit-handoff-2026-07-18]] — exact EIT commits, environments,
  bounded smoke evidence, custody correction, scientific blockers, and gates
- [[opd-math-verifier-recovery-2026-07-20]] — current O-only recovery
  boundary, strict reward audit, OPD diagnostic, and successor launch order
- [[opd-verifier-ledger-boundary-2026-07-22]] — score-once/attest-many
  verifier policy, gold-only symbolic estimand, bounded unknowns, and the
  independently reconstructed passing O-teacher gate
- [[opd-objective-family-expansion-2026-07-20]] — design-stage successor
  matrix for task RL, ungated/clipped/gated/bare K1, and pinned-veRL fidelity;
  no expanded arm has launched
- [[opd-program-goal-2026-07-20]] — active three-step execution goal: O_M/O_O
- [[opd-objective-family-implementation-freeze-2026-07-20]] — current fixed
  36-arm objective-family implementation, fidelity gates, and EIT launch order
  objective family, outcome-blind DeepMath qualification, then conditional
  fresh O/C source transfer; the failed M teacher remains permanently excluded
- [[opd-m-teacher-clarification-and-source-options-2026-07-20]] — exact meaning
  of the failed M teacher gate, allowed MATH roles, and an outcome-blind audit
  of candidate second teacher sources
- [[opd-math-scientific-cutover-2026-07-18]] — superseded exact-environment
  predecessor boundary and demoted evidence
- [[self-distillation-cluster-update-2026-07-17]] — OPSD/SDFT/SDPO baseline,
  capability, and method-custody update
- [[scope-old]] — historical SCOPE branch/archive map
- [[literature/index]] — persistent primary-source vault and manifests
- [[coverage-audit-2026-07-17]] — novelty/coverage audit from nine new full-paper reads
- [[direction-2026-07]] — direction map for the 2026-07-02 meeting (next steps, candidate anchors)
- [[icml-ai4law-2026-rejection]] — criticism inventory C1–C12 + assessment (the pivot document)
- [7_2_review_meeting/](7_2_review_meeting/00-README.md) — meeting packet: submission + critique + related work + pivot + EDA + results + narrative (transient, 2026-07-02)

## Concepts
- [[skill-distillation-bridge]] — post-meeting priority direction: internalize big-model agentic retrieval skills into a small model
- [[helpfulness-benchmark]] — DORMANT direction (meeting Idea 3): measure whether retrieval helped the reader, not whether it hit gold; evidence inventory ready
- [[opd-skill0-design]] — the OPD x SKILL0 experiment ladder (E0-E4): model matrix for 1-2 H100s, tokenizer constraint, decision gates
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
- [[effort-conditioned-resource-allocation]] — candidate three-dial/OPD bridge: one policy allocates thinking, retrieval, context, and verification under reader-specific costs; novelty gates and baselines recorded

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
- [[skill0]] — **skill internalization via curriculum RL** (arXiv 2604.02268, ZJU/Meituan); the meeting's distillation-bridge anchor; PDF+repo archived on EIT
- [[sdar]] — task RL + gap-gated on-policy self-distillation; bare OPSD collapse is the OPD safety correction
- [[ema-policy-gradient]] — K1/K4 gradient correction, Top-k KL, and why EMA
  anchoring is a different teacher experiment
- [[verl-opd-trainer]] — pinned upstream veRL OPD implementation, objectives,
  token custody, and hardware boundary
- [[skill1]] — unified skill selection/use/distillation; broad skill-internalization novelty is occupied
- [[opsd-self-distilled-reasoner]] · [[sdft-continual-learning]] ·
  [[sdpo-rich-feedback]] — verified-solution, demonstration, and rich-feedback
  self-distillation; generic context-to-weights novelty is occupied
- [[predicting-retrieval-utility]] · [[cue-r]] · [[beyond-relevance-utility]] — utility prediction/intervention landscape
- [[budget-constrained-agentic-search]] — fixed search-budget/cost evidence
- [[inkling-controllable-effort]] · [[training-language-models-to-reason-efficiently]] · [[l1-length-control]] — fixed-penalty versus single-model controllable reasoning-effort lineage
- [[acting-less-otc]] · [[autosearch]] · [[budget-aware-tool-use]] — direct action-efficiency neighbors: minimal tool calls, capability-aware search depth, and prompt-level budget awareness
- [[sure-rag]] · [[conflictrag]] · [[arbgraph]] — set sufficiency and conflict-arbitration landscape

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
- [[judge-mixed-legal]] — **general legal judge**: mixed-label training holds both domains (BarExam 22.1%, Housing 55.4%), zero specialization tax, $0 EIT lane (2026-07-02)
- [[offline-bandit-v0]] — bridge rung 1: cheap per-query retrieval allocation fails in all 5 cells; oracle headroom 8-24pp unreachable from features (2026-07-02)
- [[alloc-internalization-rung2]] — E1: 9B internalizes regime-level allocation from outcome labels (trained >> zero-shot); no per-question edge; frontier-positive-ns under cost pressure (2026-07-02)
- [[judge-pilot-scidocs]] — cross-domain: zero-shot judge > CE (+8.5pp p=3e-05) but citation-proxy label training *hurts* (−14pp) — label semantics decide (2026-07-02)
- [[judge-pilot-fiqa]] — label-semantics resolved: zero-shot judge > CE in all 4 domains; training = label quality × headroom (2026-07-02)
- [[medqa-fulln-matrix]] — **q200 headline retired** (+5.5pp didn't replicate); raw-RAG hurts, SCOPE repairs (+3.0pp p=0.002); dial-3 law 5-for-5 (2026-07-02)
- [[judge-capacity-dial]] — 27B ≤ 9B at the judge task: label-bound, not capacity-bound (2026-07-02)
- [[judge-answer-conversion]] — **the two-regime conversion law**: BarExam — 5.4× exposure ≠ answers (break-even ≈61%, gold-absent −3.8pp); Housing — monotone conversion, judge-evidence +11.4pp over llm_only p=5e-08, gold-absent +12pp (2026-07-02)

## Reviews
- [[icml-ai4law-2026-rejection]] — inventory, assessment, resubmission checklist, meta-lessons

## Anchors into the repo (linked, not duplicated)
- Citation gate: [signoff_log](../docs/signoff_log.md) · Run ledger: [experiments.jsonl](../logs/experiments.jsonl)
- Generated analyses: [docs/generated/](../docs/generated/) · Operating runbook: [OPERATIONS](../docs/OPERATIONS.md)
- Current literature boundary: [[coverage-audit-2026-07-17]] · Historical direction log: [[direction-2026-07]]
- Submitted paper/review provenance and recovery map: [[scope-old]] · Review synthesis: [[icml-ai4law-2026-rejection]]
- Raw sources: `references/` (gitignored working copy); persistent EIT vault
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/`, with tracked manifests
  under `wiki/literature/manifests/`.
