# Figure captions + context

All figures live in `docs/presentation/figures/` as PNGs. Click any figure path in VSCode to open it inline.

## `01_llama70b_method_matrix.png`

**Llama 3.3 70b dense × MuSiQue N=200 — paper headline.** 8 method bars sorted by mode name. Two methods lift over the rag_simple baseline: `multi_hyde_diverse` +8pp p=0.020 SIG (green) and `iterative_planning_table` +8.5pp p=0.053 TRENDING (yellow). `subagent_rag` -12pp p=0.0007 SIG NEG (red). All other methods are NS (gray). The blue dashed line is the `rag_simple` baseline at 27.5%.

Source data: `docs/signoff_log.md` Section B.1 + `docs/mcnemar_2026-04-27.md`.

## `02_barexam_cross_size.png`

**BarExam Tier 3 (full corpus N=1195) — `rag_snap_hyde` wins on both Gemma 4 sizes.** Gemma 4 26B-A4B: 78.08% → 81.17% (+3.09pp). Gemma 4 E4B: 58.49% → 62.18% (+3.69pp). The lift is consistent cross-size, which is what makes this a paper-grade Tier 3 finding rather than single-model noise.

Source data: `docs/audit_log.md` (post-fix re-scored) + `docs/signoff_log.md` Section A.

## `03_mechanism_decomposition.png`

**Mechanism — Llama 70b MuSiQue N=200.** rag_simple 27.5% → rag_multi_query 29.0% (+1.5pp NS, query diversity alone) → multi_hyde_diverse 35.5% (+8pp SIG). The +8pp lift decomposes into ~+1.5pp from diversity and ~+6.5pp from HyDE-style answer-bearing passages. **HyDE-style passages do ~80% of the work; diversity alone is non-significant.**

Source data: `docs/mcnemar_2026-04-27.md` Section B.2.

## `04_cross_domain_specificity.png`

**Methods are domain-specific.** The BarExam winner (`rag_snap_hyde`) is +3.09pp on its native domain (BarExam) but -3.5pp NS on MuSiQue. The MuSiQue winner (`multi_hyde_diverse`) is +8pp SIG on its native domain but -2.5pp NS on BarExam (paired N=200 against the Tier 3 baseline's first 200 records). Conclusion: there is no universal RAG trick — different tasks expose different bottlenecks.

Source data: `docs/signoff_log.md` Section B.4 + `docs/mcnemar_2026-04-27.md` Update 12:30.

## `05_cross_family_check.png`

**Cross-family check — mhd lift is NOT yet universal at Tier 2.** Llama 3.3 70b dense: +8pp p=0.020 SIG. Gemma 3 27B dense: +2.5pp p=0.59 NULL. The mhd lift held on Llama at full Tier 2 confirmation but did NOT replicate on Gemma 3 27B. Gemma 4 26B and Qwen3 30B MoE full-corpus runs are in flight (will land after meeting).

Source data: `docs/mcnemar_2026-04-27.md` Sections B.1 + B.3.

## `07_cost_vs_accuracy.png`

**Cost vs accuracy — Llama 70b MuSiQue N=200.** 4-panel figure:
- **A. LLM calls per question**: `iterative_planning_table` and `iter_hyde` use 6.3-6.8 calls/q (6-7× the simple methods). `mhd` uses 2 calls/q.
- **B. Input tokens per question**: `iter_hyde` is 4× the average (3,619 input tokens/q). Most other methods are 800-2,500.
- **C. Output tokens per question**: `iter_hyde` generates 822 tokens/q vs 84 for `rag_simple`. mhd generates 417/q.
- **D. Efficiency frontier**: scatter of EM vs total tokens/q. **`mhd` is the cheapest of the lifters** (top-left = better). `iterative_planning_table` matches mhd's EM but at ~4× the token cost. `iter_hyde` is the most expensive method and lifts none.

Source data: `logs/experiments.jsonl` rows tagged `captain-llama70b-musique-*-n200`.

## `06_barexam_26b_full_matrix.png`

**BarExam Gemma 4 26B-A4B Tier 3 method matrix (full corpus N=1195), sorted by EM.** Winner: `rag_snap_hyde` at 81.17% (+3.09pp). Runner-up: `snap_only_in_final` at 80.59% (+2.51pp). `llm_only` at 79.75% is a strong no-retrieval reference point. `subagent_hybrid` at 74.23% is the only method that materially hurts (-3.85pp).

Source data: `docs/audit_log.md` (post-fix re-scored from cluster vLLM detail logs) + `docs/signoff_log.md` Section A.1.

---

## Audit + caveats summary (cross-references)

- **`rag_snap_hyde` snap-dominance** (76-83% pred==snap on 26B): BY DESIGN, not contamination. See `docs/audits/2026-04-27_barexam_26b_audit.md` and signoff Top 5 #1.
- **`subagent_rag` -12pp + 100% gap-routing trigger**: implementation caveat. See `docs/audits/2026-04-27_llama70b_musique_audit.md` and signoff Top 5 #5.
- **OR-Gemma serving runaway loops** (gemma4_full + iter_ptable Gemma killed): cluster vLLM unaffected. See signoff Section D'.
- **Tier system**: N<200 directional only. See signoff Section A header.
