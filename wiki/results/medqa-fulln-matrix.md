---
title: MedQA Full-N Matrix — q200 headline retired; the law holds 5-for-5
type: result
tags: [medqa, medical, conversion, correction, full-n]
created: 2026-07-02
updated: 2026-07-02
date: 2026-07-02
verdict: correction (q200 +5.5pp over llm_only does NOT replicate) + law confirmation
evidence: logs/eval_{llm_only,rag_simple,rag_hyde,snap_hyre}_groq-llama70b_20260702_*_medqa_detail.jsonl
---

# MedQA-USMLE full-N answer matrix (N=1,273, groq-llama70b, strict replay)

**Question**: does the q200 probe's headline — the only significant SCOPE
answer win over llm_only (+5.5pp, p=0.019) — survive full N? **No.**

| Arm | full-N acc | vs llm_only | q200 probe |
|---|---:|---|---:|
| llm_only | 85.6% | — | 78.0% |
| raw-question RAG | 83.1% | **−2.44pp, p=0.005 (hurts)** | 76.5% |
| HyDE | 85.2% | −0.31pp ns | — |
| SCOPE | 86.1% | +0.55pp ns | 83.5% |

SCOPE vs raw-RAG: **+2.99pp, b/c=91/53, p=0.002** (holds). SCOPE vs HyDE:
+0.86pp ns (consistent with the answer-parity ledger).

**Readings.**
1. **The q200 slice was hard by luck** (its llm_only 78.0% vs full-N 85.6%);
   the +5.5pp-over-llm_only claim is retired. Full-N discipline caught it
   before it reached a paper — exactly what the May postmortem demanded.
2. **The MedQA pattern = BarExamQA-at-70B**: raw retrieval injects
   distractors and *hurts* a parametric-strong reader; generated-query
   evidence repairs the damage to parity (SCOPE +3.0pp over raw, p=0.002).
   Expansion's answer-side value on strong readers is **harm avoidance**,
   not lift.
3. **Dial 3 goes 5-for-5**: llm_only 85.6% → low parametric deficit → no
   conversion, as the [[judge-answer-conversion]] 2×2 law predicts. MedQA
   was the stress test (the q200 probe had suggested a violation); the law
   held and the anomaly resolved as sampling noise.
4. Prediction the law now makes: MedQA evidence should pay for the **8B**
   reader (its llm_only will sit far lower) — a cheap future arm.

**Caveats.** No gold passage ids on MedQA (answer-only metrics; no
retrieval-side or conversion decomposition). Single reader size so far.
Detail logs are same-day strict cache replays (1,273/1,273 rows each,
NO_SILENT_FALLBACK).

## Links
[[thesis-v2]] · [[judge-answer-conversion]] · [[weak-vs-strong-query-regime]]
· [[icml-ai4law-2026-rejection]] (C5) ·
[q200 probe doc](../../docs/generated/medqa_usmle_widening_2026-05-26.md) (superseded on the answer headline)
