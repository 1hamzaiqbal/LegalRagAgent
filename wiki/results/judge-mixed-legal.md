---
title: Mixed-Label Legal Judge — one judge holds both domains, for free
type: result
tags: [judge, mixed-label, generalization, eit, free-lane, win]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: win — mixed-label training carries zero specialization tax on either domain
evidence: scripts/judge_pilot/data/local_scores_mixed_barexam.json, scripts/judge_pilot/data/local_housing_scores_mixed_housing.json
---

# Mixed-label legal judge (EIT free lane, job 93660)

**Question** (queued by the [[judge-pilot-housing]] transfer result): the
BarExam-trained judge transferred to Housing at only 46.4% — below Housing
zero-shot (52.8%) — so judgment training *specializes*. Does training one
judge on **mixed** barexam+housing labels recover both domains, or does
mixing dilute both?

**Setup.** 8,500 mixed pairs (3,500 BarExam + 5,000 Housing, shuffled; each
row keeps its own domain prompt template baked into `prompt_text`), otherwise
the exact v0 recipe: Qwen3.5-9B LoRA r=32, loss on " Yes"/" No" tokens only,
3 epochs, lr 1e-4, effective batch 128. Trained and scored entirely on the
free EIT A100 lane (`scripts/judge_pilot/local_judge.py`, Slurm job 93660,
checkpoint `ckpt_mixed_legal_v5`, ~2.5h, $0).

| Held-out pools | specialized judge | **mixed judge** | McNemar (mixed vs spec) |
|---|---:|---:|---|
| BarExamQA (399, ceiling 22.8%) | 20.6% / MRR 0.138 | **22.1% / MRR 0.142** (88/399) | b/c=7/1, p=0.070 |
| HousingQA (500, ceiling 57.0%) | 55.0% / MRR 0.477 | **55.4% / MRR 0.486** (277/500) | b/c=3/1, p=0.625 |

Gold-in-pool conversion: BarExam 88/91 = 96.7% (specialist 90.1%); Housing
277/285 = **97.2%** (specialist 96.5%).

**Reading.**
1. **Specialization is a single-domain-training artifact, not a limit of
   judgment training.** The 46.4% transfer number measured what happens when
   you train on one relevance notion and test on another; mixing the labels
   removes the problem entirely — both domains at or directionally above
   their specialists (BarExam +1.5pp, 7/1 discordant, p=0.070; Housing tied).
2. **A deployable "general legal judge" costs nothing extra**: same 9B, same
   recipe, one training run over pooled free outcome labels. This strengthens
   P3 of [[thesis-v2]] — the selector is trainable across relevance notions
   simultaneously when the labels encode each notion explicitly.
3. Mild positive transfer on BarExam (88 vs 82 hits at a 91-hit ceiling —
   now 96.7% of ceiling) suggests the two label sets share a "controlling
   legal authority" core; worth one line in the paper, not a claim.

**Caveats.** One seed, one model family. The Housing comparison is
cross-infrastructure (mixed = EIT local PEFT, specialist = Tinker), justified
by the exact 82/399 Tinker↔EIT reproduction on BarExam
([[judge-pilot-v0-results]] §Free-infrastructure replication). Labels remain
in-distribution benchmark gold — the lawyer-label rung of Path C is untouched
by this result. Pool ceilings still bind (22.8% / 57.0%).

**Reproduce.** `judge_lane5_mixed.sbatch` on EIT (`sbatch -A engr-lab-jacobsn`);
mixed train file is a shuffled concat of `train.jsonl` + `housing_train.jsonl`;
score per-domain with `--prefix ""` and `--prefix housing_` against
`ckpt_mixed_legal_v5`.

## Links
[[judge-pilot-v0-results]] · [[judge-pilot-housing]] (transfer §, superseded
as a deployment limit) · [[judge-capacity-dial]] · [[thesis-v2]] (P3) ·
[[expert-judgment-replication]] · [[direction-2026-07]]
