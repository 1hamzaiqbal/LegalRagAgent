---
title: Judge Pilot — SciDocs: judge idea transfers, blind label-training does not
type: result
tags: [judge, tinker, scidocs, cross-domain, negative, label-semantics]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: mixed — zero-shot judge > CE (win); training on citation-proxy labels hurts (informative negative)
evidence: scripts/judge_pilot/data/scidocs_eval_results.json, scripts/judge_pilot/data/scidocs_train_info.json
---

# SciDocs judge — the cross-domain test (thesis-v2 prediction 3)

**Question**: does the free-label judge recipe transfer outside legal?
**Answer: half of it.** Same recipe (Qwen3.5-9B LoRA, 2,645 pairs from qrels
gold + retrieved hard negatives, query-level splits), 400 held-out
raw∪SCOPE pools, ceiling 77.2%:

| Arm | Hit@5 | MRR@5 | conversion |
|---|---:|---:|---:|
| raw top5 | 48.2% | 0.306 | — |
| SCOPE top5 | 46.3% | 0.284 | — |
| CE ms-marco (cached) | 52.0% | 0.326 | 67.3% |
| **judge-zeroshot** | **60.5%** | **0.359** | **78.3%** |
| judge-trained | 46.5% | 0.250 | 60.2% |

McNemar: zeroshot vs CE **+8.5pp, 50/16, p=3.3e-05** ✓ · trained vs zeroshot
**−14.0pp, 48/104, p=6.5e-06** ✗ · trained vs CE −5.5pp, p=0.097.

**Reading — the P3 refinement this forces.**
1. **The selector-bottleneck claim is now cross-domain**: an LLM judge beats
   the ms-marco CE on identical pools in all three domains tested (BarExam
   zeroshot +11.5pp, Housing +14.6pp, SciDocs +8.5pp over CE). The CE is the
   weakest link everywhere.
2. **Label semantics decide whether training helps.** Legal gold =
   human-annotated controlling authority → training adds +5.3pp/+2.2pp and
   large MRR gains. SciDocs "gold" = citation/co-view behavior — a noisy
   behavioral proxy in which topically relevant non-cited papers are labeled
   negative; training on it teaches the judge to suppress exactly what pool
   reranking needs (−14pp vs its own zero-shot). **The recipe's value tracks
   label quality, not domain** — fully consistent with
   [[thinking-machines-expert-judgment]], whose labels were true expert
   judgments, and a caution for anyone fine-tuning rerankers on click/citation
   proxies.
3. Prediction 3 as pre-stated ("gain ∝ CE bury-rate") gets a friendly
   amendment: SciDocs' CE bury-rate was the lowest of the three (CE already
   converts 67%), so headroom was small and label noise dominated.

**Caveats.** One proxy-label domain; the trained judge used our default
hyperparameters (no dev-set early stop — the 300-pair dev file exists,
unused); prompt framing for SciDocs ("related-work seeking") may mismatch
task semantics; FiQA/NFCorpus (graded human qrels) would separate
label-noise from domain-shift explanations — queued as the natural follow-up.

## Links
[[thesis-v2]] (prediction 3: revised, not killed) · [[judge-pilot-v0-results]]
· [[judge-pilot-housing]] · [[thinking-machines-expert-judgment]] ·
[[expert-judgment-replication]] · [[direction-2026-07]]
