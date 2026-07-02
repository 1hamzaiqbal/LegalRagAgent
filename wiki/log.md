---
title: Wiki Op Log
type: hub
tags: [log]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# Wiki Op Log (append-only)

## [2026-07-02] init | Wiki created
Bootstrapped LegalRagAgent wiki in the BoundEO LLM-Wiki pattern, triggered by the
ICML AI4Law rejection of the SCOPE paper (submission #97, two strong rejects).
First campaign: ingest the reviews, pull the reviewer-named prior art
(KoBLEX/ParSeR, GuRE) + the QPP/expansion literature already mapped in
[RELATED_WORK_GROUNDING](../paper/submission/RELATED_WORK_GROUNDING.md), survey
post-submission repo work, and produce a criticism-by-criticism assessment +
direction map for the 2026-07-02 meeting.

## [2026-07-02] ingest | 23 sources pulled + paged (22-agent campaign)
28 raw files into `references/` (mirrored to
`wustl:/engrfs/tmp/jacobsn/hiqbal_legalrag/references/papers/`; KoBLEX + GuRE
repos cloned under `.../repos/`). Source pages: koblex-parser, gure, hyde,
query2doc, zheng-cslaw, weller-drift, faggioli-qpp, emami-qpp-variant,
tian-right-track, adaptive-rag-mallen, lamer-gar, csqe,
legal-rag-benchmarks-src, datta-qpp-reliability, power-noise-lostmiddle,
thinking-machines-expert-judgment + discovery pulls (yoon2025leakage,
li2026legalmalr, afane2026laborbench, guha2023legalbench, jiang2023syllogism,
magesh2024hallucinationfree). Discovery sweeps returned ~24 ranked candidates
(Reuter'25 SAC, LexPath'26, LEMUR'26, LexRAG'25, CRAwLeR'26, L-MARS,
LegalAgentBench, Chain-of-Logic, Makri'08 lawyer info-seeking, Aletras'16 …)
— unpulled ones are TODO markers.

## [2026-07-02] experiment | Judge pilot v0 launched on Tinker (Path C)
Built `scripts/judge_pilot/` (dataset from signed caches + qa.csv gold;
passage texts hydrated from EIT corpus CSV + HF test/validation splits after
discovering the experiment-box Chroma used the full 857K corpus vs our 686K).
Trained Qwen3.5-9B LoRA (rank 32, 84 steps, loss 2.18→0.15) on 3,500
question–passage relevance pairs; eval = rerank the identical raw∪SCOPE pools
the CE reranked (399 held-out pools, 22.8% recall ceiling). Zero-shot arm
already decisive: judge-zeroshot Hit@5 15.3% vs CE 3.8% vs SCOPE-alone 12.0%,
converting 61/91 gold-in-pool. **Trained arm: Hit@5 20.6% / MRR@5 0.138,
90.1% gold-in-pool conversion; all deltas McNemar-significant (vs CE p=1.4e-17,
vs SCOPE-alone p=3.4e-06, vs zeroshot p=1.0e-04).** The May "pooling destroys
weak-query gains" verdict was a CE artifact, now fixed by a trained selector.
Full read: [[judge-pilot-v0-results]].

## [2026-07-02] synthesize | Review postmortem + concepts + results + direction
Wrote reviews/icml-ai4law-2026-rejection (C1–C12 inventory + assessment),
10 concept pages, methods/scope, 8 results pages (numbers verified against
docs/generated/ artifacts after fast-forwarding local scope-generalization 44
commits to shrango head 21126a0), and direction-2026-07 (meeting map: Path A
mechanism paper / Path B legal companion / Path C judgment-replication pilot).
Adversarial verifiers: KoBLEX = substantial-overlap-differentiable; snap-vs-
HyDE ledger = 13/16 NS with dataset-split significant cells; memory claims
(a)–(d) all CONFIRMED with per-dataset caveats (three-retriever closure is
mean-level; TREC-COVID low). Corrected an overstatement in methods/scope.md
("null on every pair" → 13/16 NS, direction dataset-dependent).
