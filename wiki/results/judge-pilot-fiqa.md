---
title: Judge Pilot — FiQA: human labels are safe; training's value = label quality × headroom
type: result
tags: [judge, tinker, fiqa, label-semantics, cross-domain]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: win (zero-shot > CE again) + label-semantics account refined
evidence: scripts/judge_pilot/data/fiqa_eval_results.json, scripts/judge_pilot/data/fiqa_train_info.json
---

# FiQA judge — the label-semantics resolver

**Question** (after [[judge-pilot-scidocs]]'s training collapse): with *human
relevance* qrels instead of citation proxies, does label-training help again,
or was SciDocs a domain problem? Same recipe: 1,689 pairs (FiQA is small:
648 pooled questions; test 250 pools, ceiling 90.8%), Qwen3.5-9B LoRA.

| Arm | Hit@5 | MRR@5 | conversion |
|---|---:|---:|---:|
| SCOPE top5 (drifted, known) | 36.0% | 0.214 | — |
| raw top5 | 64.8% | 0.503 | — |
| CE ms-marco (cached) | 70.0% | 0.555 | 77.1% |
| judge-trained | 82.4% | 0.637 | 90.7% |
| **judge-zeroshot** | **84.0%** | 0.629 | **92.5%** |

McNemar: zeroshot vs CE **+14.0pp, 41/6, p=1.8e-07** · trained vs CE +12.4pp,
p=9.3e-06 · trained vs zeroshot −1.6pp, 9/13, **p=0.52 (parity)**.

**Reading — the four-domain judge picture completes:**
1. **Zero-shot LLM judge > ms-marco CE in every domain tested** (BarExam
   +11.5, Housing +14.6, SciDocs +8.5, FiQA +14.0pp — all p≤3e-05). The
   general-domain CE is the weakest link of the standard RAG stack,
   universally.
2. **Training's effect = label quality × remaining headroom**:
   - legal human gold + big headroom → helps (+5.3/+2.2pp, MRR ++)
   - FiQA human relevance + zero-shot already at ceiling (92.5% conversion of
     90.8% ceiling) → **neutral** (safe, slight MRR gain)
   - SciDocs citation proxy → actively harmful (−14pp)
   Human-grade labels are never harmful; proxy labels are worse than no
   training. This is the Thinking-Machines thesis with a sign structure.
3. Train-data scale caveat: 1,689 pairs (smallest of the four); neutrality at
   ceiling is expected regardless.

## Links
[[thesis-v2]] (P3 final form) · [[judge-pilot-v0-results]] ·
[[judge-pilot-housing]] · [[judge-pilot-scidocs]] ·
[[thinking-machines-expert-judgment]] · [[expert-judgment-replication]]
