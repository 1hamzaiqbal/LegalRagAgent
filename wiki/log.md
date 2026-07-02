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

## [2026-07-02] experiment | MedQA full-N: q200 headline retired, law holds
N=1,273 strict replay: llm_only 85.6% / raw-RAG 83.1% (hurts, p=0.005) /
HyDE 85.2% / SCOPE 86.1% (+0.55 ns vs llm_only; +2.99pp p=0.002 vs raw).
The q200 "+5.5pp over llm_only" claim does NOT replicate (hard slice by
luck). Dial-3 law now 5-for-5 cells. Page: [[medqa-fulln-matrix]].

## [2026-07-02] experiment | Reader-size 2x2: conversion = parametric deficit
8-arm groq-llama8b replay (same evidence artifacts): regimes INVERT at 8B —
BarExam evidence pays (+11.8pp SCOPE-ev p=5.6e-05; judge +8.8pp p=0.0026;
even gold-absent +7.3pp) while Housing stops paying (gold-present -6.5pp).
Across the 4 (reader x task) cells, evidence pays iff llm_only is weak
(crossover ~60%): dial 3 = reader-task parametric deficit, pre-testable with
one llm_only run. Also: capacity curve completed (prompted 235B = prompted 9B
= 15.3% < trained 9B 20.6%); EIT lane v2 resubmitted (job 93606, dedicated
venv after transformers 4.57 lacked qwen3_5).

## [2026-07-02] experiment | Capacity dial: judge is label-bound, not capacity-bound
Qwen3.6-27B on identical BarExam pools: zeroshot 14.0%/trained 18.5% vs 9B's
15.3%/20.6% — 3x parameters bought nothing; training lift replicates at both
scales (+4.5/+5.3pp). Spend on labels, not parameters. 235B prompted-frontier
arm running. Page: [[judge-capacity-dial]].

## [2026-07-02] experiment | FiQA judge: four-domain picture completes
Zero-shot judge beats the CE in ALL FOUR domains (+8.5..+14.6pp, p≤3e-05) —
the ms-marco CE is universally the weakest stack component. Training =
label-quality × headroom: legal human gold helps; FiQA human labels at
ceiling are neutral (trained≈zeroshot, p=0.52); SciDocs citation proxies
harm. Page: [[judge-pilot-fiqa]].

## [2026-07-02] experiment | Housing answer arms: conversion pays — two-regime law complete
Four paired Housing arms (500 questions, state-filtered strict replay,
groq-llama70b): llm_only 54.2 → CE-ev 61.8 (+7.6, p=1.6e-04) → SCOPE-ev 63.2
→ **judge-ev 65.6% (+11.4pp, p=5.5e-08; beats CE arm +3.8pp p=0.048)** —
monotone in evidence quality. Gold-absent evidence HELPS here (+12.0pp) vs
HURTS on BarExam (−3.8pp): the break-even cost term is regime-dependent.
Full pipeline law recorded in [[thesis-v2]]. Ops notes: pipeline-status bug
masked a state-filter guard failure (fixed with pipefail); EIT login-node I/O
degraded → extraction moved to sbatch (job 93491). Page updated:
[[judge-answer-conversion]].

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

## [2026-07-02] eit-lane-validated | Free judge-training lane reproduces Tinker exactly
EIT A100 job 93632 (local_judge.py HF PEFT port, Qwen3.5-9B, LoRA r=32,
micro 4 × accum 32) landed Hit@5 20.55% / MRR 0.1345 / 82/399 — identical
hit count to the Tinker reference (20.6% / 0.138 / 82/399). Racing A40 job
93629 cancelled. Judge training is now $0/run on general-gpu A100-SXM4
(gres type `a100-sxm4`; the `a100s` alias is stale). Section added to
results/judge-pilot-v0-results. Queued next on the free lane: mixed-label
barexam+housing "general legal judge" (motivated by the 46.4% transfer
specialization result).

## [2026-07-02] meeting-packet | 7_2_review_meeting folder + 5 Codex lanes
Created wiki/7_2_review_meeting/ (transient meeting packet, 00-README pins
conventions). Five Codex (gpt-5.5 xhigh) lanes launched in parallel, one per
document: 01 submission summary + primary sources, 02 critique dossier
(fixed/fixable/remains), 03 overlapping related work, 04 generalization-pivot
memo + 05 datasets EDA (with figures), 06 consolidated results + meeting
questions, 07 experiment narrative since submission (with timeline figure).

## [2026-07-02] meeting-packet-complete | All 7 documents landed, reviewed, committed
All five Codex (gpt-5.5 xhigh; account rejected the -codex variant) lanes
finished: 01 submission reconstruction + primary sources in assets/primary/,
02 critique dossier (quotes enriched to verbatim reviewer excerpts), 03
overlapping related-work dossier (KoBLEX/GuRE = clear uncited misses; Zheng =
cited-but-not-engaged; coverage map of surviving white space), 04
generalization-pivot memo (RQ1-RQ7), 05 datasets EDA (11-dataset
vocabulary-gap violin; Jaccard-vs-TFIDF metric caveat added), 06 consolidated
results + 12 meeting questions with leans, 07 experiment narrative (Era 0-7,
dead ends included, timeline + claims-alive/killed figures). Lane B's original
codex task died instantly on the gpt-5.5-codex model rejection and was
relaunched in blocking mode. Staleness fixes applied post-landing: EIT race
resolution in 06/07, MRR 0.135-vs-0.138 in 04. Folder indexed from wiki index.

## [2026-07-02] mixed-legal-judge | One judge holds both domains, zero specialization tax
EIT A100 job 93660 ($0, ~2.5h): Qwen3.5-9B LoRA r=32 trained on 8,500 mixed
barexam+housing pairs. BarExam pools 22.1% Hit@5 / MRR 0.142 (88/399, 96.7% of
ceiling) vs specialist 20.6% — McNemar 7/1, p=0.070 directionally BETTER.
Housing pools 55.4% / MRR 0.486 (277/500, conversion 97.2%) vs specialist
55.0% — 3/1, p=0.625 tied. The 46.4% transfer-specialization result is
superseded as a deployment limit: it was an artifact of single-domain
training. New page results/judge-mixed-legal; thesis-v2 P3, judge-pilot-housing
transfer §, and meeting-packet 06 updated. Scores mirrored to
scripts/judge_pilot/data/local_{,housing_}scores_mixed_*.json.

## [2026-07-02] mentor-meeting + skill0-ingest | Meeting notes recorded; distillation bridge drafted
The mentor meeting happened. Notes at 7_2_review_meeting/08-meeting-notes
(verbatim-ish + found-evidence annotations: the golden/neighbor/gold-absent
experiments Idea 3 asked to find are inventoried from signed rows). Priority
thread per HI: skill internalization x distillation — SKILL0 (arXiv
2604.02268) ingested with PDF + SkillZero repo archived to EIT
references/{papers,repos}; source note sources/skill0; direction draft
concepts/skill-distillation-bridge (cross-scale twist on SKILL0, teacher-
access technical paths, v0 sketch on the free EIT lane; novelty checks
queued: SDAR, SKILL1, closed-teacher distillation survey). direction-2026-07
gained a post-meeting addendum; packet README + index updated.

## [2026-07-02] idea3-dormant + bandit-rung1 | Helpfulness benchmark recorded; offline bandit built and run
Idea 3 written up as a dormant, pickupable direction (concepts/
helpfulness-benchmark: metric family EHE/harm-adjusted/cost-per-solved-task,
evidence inventory from signed rows, pick-up checklist; no active work).
Bridge rung 1 executed: scripts/bandit/offline_bandit_v0.py replays the
paired 7/2 arms as a single-turn retrieve-or-not/arm-choice bandit (5 cells,
zero new LLM calls). Verdict: instructive negative — no cheap policy
(features + judge scores, logistic or gate) beats the best fixed arm
anywhere (deltas -0.3 to -4.0pp, all ns), while the per-question oracle sits
8-24pp above (noise-inflated but large): allocation headroom is real and
unreachable from external features. Extends qpp-routing-negative to
answer-level allocation; motivates rung 2 = internalized policy on the EIT
lane. Report: docs/generated/offline_bandit_v0_2026-07-02.md + frontier PNG;
wiki page results/offline-bandit-v0.

## [2026-07-02] rung2-launched + opd-design | Internalized allocation training on EIT; OPD ladder designed
Rung 2 launched as EIT job 93770 (free A100): 9B LoRA on 6,136
(question, reader, strategy)->Yes/No pairs built from all 5 bandit cells with
rung-1-IDENTICAL seed-0 splits (scripts/bandit/build_alloc_dataset.py);
scores 6,148 held-out pairs trained+zeroshot (score_alloc_pairs.py); policy
= argmax(score - lam*cost), analyzed against rung-1 fixed arms on the same
test halves. OPD x SKILL0 experiment design written
(concepts/opd-skill0-design): E0 bandit (done) -> E1 rung 2 (running) -> E2
teacher skill-gap A/B -> E3 OPD vs outcome-labels on 1-2 H100 (Qwen3 track:
32B teacher 1xH100, 235B-A22B-FP8 TP=2 on 2; Llama track: 70B; hard
same-tokenizer constraint) -> E4 multi-turn Search-R1-style + curriculum.
Codex lane building scripts/opd/ scaffold (vLLM teacher client, reverse-KL
OPD loss + CPU tests, train skeleton, A100 smoke test). Job 93770 also
prints nvidia-smi to settle whether a100-sxm4 = 80GB (if so, one a100s node
= 2-H100-class TP capacity for free).

## [2026-07-02] opd-scaffold-landed + smoke-launched | scripts/opd complete, A100 smoke on EIT
Codex lane delivered scripts/opd/ (teacher_client vLLM prompt_logprobs
scoring, opd_loss reverse-KL policy-gradient w/ clamp + ratio-clip +
kd_forward_loss closed-teacher fallback, opd_train LoRA skeleton w/ --mode
kd, smoke_test.sh, CPU unit tests). Review pass: loss math verified correct;
patched the one real gap (vllm serve defaulted to 0.9 gpu-memory-utilization
— would starve the co-located student; now OPD_TEACHER_GPU_FRAC=0.55).
Wrote scripts/opd/skills/allocation.md (the E2/E3 skill file, distilling the
three-dial rules with measured numbers). CPU tests re-verified locally (all
PASS). End-to-end smoke (Qwen3-8B teacher -> Qwen3-1.7B student, 3 OPD
steps) submitted as EIT job 93773 in a dedicated opd_lane venv (vllm pins
its own torch; kept away from judge_lane). Rung-2 job 93770 running in
parallel.
