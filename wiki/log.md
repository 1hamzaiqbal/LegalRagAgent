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

## [2026-07-02] experiment | Answer-conversion wall measured: break-even ≈ 61% Hit@5
Four paired arms on the 399 judge-test questions (groq-llama70b, strict
harness replay, ~1,600 calls): llm_only 77.7% ≥ CE-evidence 76.7% ≥
SCOPE-evidence 76.2% ≥ judge-evidence 75.2% (all ns) — the 5.4× exposure fix
does not convert on BarExamQA/70B. Decomposition: gold-present evidence
+2.4pp, gold-absent −3.8pp → break-even Hit@5 ≈ 61% vs pool ceiling 22.8%.
Post-hoc score-gated evidence: +0.75pp best case (ns). Constraint moves
up-stack to pool recall + evidence-conditional answering; Housing answer run
is the decisive next arm. Also: wiki lint (57 pages, 0 orphans, 0 broken
links); EVAL_QA_CSV harness override added. Page: [[judge-answer-conversion]].

## [2026-07-02] experiment | SciDocs judge: idea transfers, blind training doesn't
Prediction 3 revised: zero-shot judge beats CE cross-domain (SciDocs +8.5pp,
p=3.3e-05 — CE is the weakest link in all 3 domains tested), but training on
SciDocs' citation-proxy gold HURTS (−14pp vs zero-shot, p=6.5e-06). New
statement: label semantics, not domain, decide whether judgment-training
pays — TM-consistent, and a caution against fine-tuning rerankers on
behavioral proxies. Page: [[judge-pilot-scidocs]].

## [2026-07-02] experiment | Housing judge (strong regime): routing superseded
Same recipe, new regime: group-level splits, 5,000 pairs, 13-min Tinker LoRA.
On 500 held-out state-filtered pools (57.0% ceiling): trained judge Hit@5
55.0% / MRR 0.477 / 96.5% conversion vs CE-pool 38.2% (p=2.5e-23), SCOPE-alone
41.2% (p=8.5e-12), raw 33.4% (p=1.5e-24); trained>zeroshot p=0.043 (MRR
+0.092). Thesis-v2 prediction 2 supported → [[regime-routing]] superseded by
always-pool + trained judge. Page: [[judge-pilot-housing]].

## [2026-07-02] experiment | BEIR leakage replication: help is never leakage-gated
Canonical HyDE/SCOPE generations on SciFact/NFCorpus/SciDocs: matched rates
0–7% (most ≤1%), help_m=0 in all six cells — every expansion-help event
occurs on unmatched generations. Combined with BarExamQA: leakage cannot
explain expansion behavior in either regime. Appended to
[[leakage-audit-barexam]].

## [2026-07-02] experiment | Leakage audit: Yoon account rejected on BarExamQA
Queue #1 executed locally (deberta-v3-base NLI on MPS, 7,703 pairs over 1,192
questions × 3 SCOPE samples, dense-stage retrieval). Matched (gold-entailed)
samples = 14–15%; unmatched lift = **+5.9 to +6.1pp Hit@5 over raw 1.4%**;
strictest all-unmatched stratum: 10.5% vs 1.5%, McNemar 88/5, p=1.1e-20.
Thesis-v2 prediction 1 supported; the strongest external threat to Path A
defused and converted into a positive decomposition section. Caveats: 3SCOPE
exemplar variant (conservative), single dataset/generator. Page:
[[leakage-audit-barexam]].

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
