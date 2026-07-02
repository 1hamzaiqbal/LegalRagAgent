---
title: Judge Pilot v0 Results — trained judge un-buries the pool
type: result
tags: [judge, tinker, path-c, reranking, win]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: win (all deltas McNemar-significant)
evidence: scripts/judge_pilot/data/eval_results.json, scripts/judge_pilot/data/train_info.json
---

# Judge pilot v0 — results (BarExamQA, 399 held-out pools)

Setup in [[judge-pilot-v0]]: rerank the *identical* raw∪SCOPE 20-candidate
pools the ms-marco CE reranked in the signed cache; 399 question-held-out
pools, gold-in-pool recall ceiling **22.8%** (91/399). Judge = Qwen3.5-9B on
Tinker; trained arm = LoRA rank 32, 84 steps on 3,500 outcome-free relevance
pairs (gold ids + retrieved hard negatives), loss 2.18 → ~0.15, wall ~23 min.

| Arm (same pools) | Hit@5 | MRR@5 | gold-in-pool conversion |
|---|---:|---:|---:|
| raw-question top5 (cached) | 1.3% | 0.009 | — |
| **CE ms-marco (cached — what the paper used)** | 3.8% | 0.018 | 16.5% (15/91) |
| SCOPE-alone top5 (cached) | 12.0% | 0.055 | — |
| judge-zeroshot (same 9B, same prompt, untrained) | 15.3% | 0.070 | 67.0% (61/91) |
| **judge-trained (LoRA, 84 steps)** | **20.6%** | **0.138** | **90.1% (82/91)** |

Significance (exact McNemar on per-pool Hit@5):
- trained vs CE: b/c 70/3, **p=1.4e-17**
- trained vs SCOPE-alone: 44/10, **p=3.4e-06**
- trained vs zeroshot: 25/4, **p=1.0e-04** ← the trained-beats-prompted delta
- zeroshot vs CE: 48/2, p=2.3e-12

## What this changes

1. **The selector was the bottleneck.** May's verdict "pooling destroys the
   weak-query gain" ([[pooling-regime]]: pool 3.9% vs SCOPE 12.0%) was a fact
   about the *ms-marco CE*, not about pooling: with a trained judge the same
   pools yield 20.6% — the best BarExamQA retrieval number in the project's
   history, at 90% of the pool ceiling. The CE-buries-gold diagnosis
   (ideas.md §5) is now a validated, fixed failure.
2. **Regime-routing may simplify.** Pool+trained-judge wins on the weak end
   where pooling previously lost; if it also holds on strong regimes
   (Housing — untested), "route expansion" may collapse into "always pool,
   judge with a trained selector" ([[regime-routing]] update owed).
3. **The Thinking-Machines thesis transfers at academic scale**
   ([[thinking-machines-expert-judgment]]): the same model with the same
   prompt gains +5.3pp Hit@5 / +23pp conversion from 23 minutes of LoRA on
   3,500 free labels — judgment training beats prompting, significantly, at
   9B and $≈15 of compute.
4. **C1 gets a constructive answer**: this is a *legal-judgment* model
   (which authority controls this fact pattern), not another query trick.

## Caveats (honest, before anyone gets excited)
- **Retrieval-side only.** The [[answer-conversion-gap]] still stands between
  20.6% Hit@5 and answer accuracy; the May prompted-judge result on Housing
  showed exposure-without-answer-gain is possible. Next cell: q200 answer run
  with judge-top5 evidence vs SCOPE evidence.
- Ceiling is the pool: 22.8% — raising it needs better candidate generation
  (deeper k, better expansion), which is orthogonal.
- Single dataset (BarExamQA), single seed/model; gold labels are the
  benchmark's single-gold annotation (Hit@k pessimism caveat from
  [[zheng-cslaw]] applies to all arms equally).
- Training labels come from the same benchmark's gold ids (in-distribution
  supervision); the Path C vision — replicating *lawyer* judgments beyond
  benchmark labels — is still ahead. This pilot proves the training recipe
  and the selector diagnosis, not judgment transfer.
- Judge scoring costs ~2 forward passes/candidate at 9B — fine for k=20
  pools, not a drop-in for 686K-corpus first-stage retrieval.

## Free-infrastructure replication (EIT, 2026-07-02 evening)

The Tinker recipe was ported to a plain HF PEFT script
(`scripts/judge_pilot/local_judge.py`: same LoRA r=32 on attn+mlp, loss on
" Yes"/" No" tokens only, 3 epochs, lr 1e-4, effective batch 128) and run on
a free EIT `general-gpu` A100-SXM4 (Slurm job 93632, ~1h train+score,
Qwen3.5-9B slow-attention fallback path, micro-batch 4 × accum 32):

| Lane | Hit@5 | MRR@5 | hits |
|---|---:|---:|---:|
| Tinker-trained (reference) | 20.6% | 0.138 | 82/399 |
| **EIT local-trained (A100, free)** | **20.6%** | **0.135** | **82/399** |

Identical hit count, MRR within 0.003 — the recipe is
infrastructure-independent and the training lane is now **$0/run**
(checkpoint `.../judge_pilot_v0_data/ckpt_barexam_local_v4` on EIT; scores
mirrored to `data/local_scores_trained_local_v4.json`). A racing A40 job
(93629) was cancelled once the A100 landed. All follow-on judge training
(mixed-label legal judge, deeper pools, MedQA) runs on this free lane.

## Reproduce
```
scripts/judge_pilot/build_judge_dataset.py   # dataset from signed caches
scripts/judge_pilot/train_tinker_judge.py    # Tinker LoRA (TINKER_API_KEY)
scripts/judge_pilot/eval_judge_rerank.py     # arms: trained,zeroshot
scripts/judge_pilot/local_judge.py           # free EIT lane (HF PEFT port)
```
Checkpoint: `tinker://f359dd9b-…/sampler_weights/barexam-judge-v0-final`
(recorded in `data/train_info.json`).

## Links
[[judge-pilot-v0]] · [[pooling-regime]] · [[answer-conversion-gap]] ·
[[thinking-machines-expert-judgment]] · [[expert-judgment-replication]] ·
[[regime-routing]] · [[direction-2026-07]]
