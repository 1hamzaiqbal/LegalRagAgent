# 2026-04-27 Meeting Brief v2 (post-McNemar reckoning)

## Update 2026-04-27 ~12:30 CDT

Change reason: Llama 70b N=200 landed after this v2 brief was drafted. Updated meeting headline: `multi_hyde_diverse` is a Llama 70b Tier 2 paper headline (**35.5%** vs `rag_simple` **27.5%**, +8.0pp, McNemar p=0.0195), but not a confirmed cross-family claim because Gemma 3 27B N=200 remains NULL (+2.5pp, p=0.5901). Llama mechanism decomposition is now Tier 2: `rag_multi_query` contributes only +1.5pp NS (p=0.728), so HyDE-style answer passages explain about +6.5pp of the +8pp lift. `subagent_rag` is a significant negative on Llama MuSiQue (-12.0pp, p=0.0007). Current HEAD is `44427ad`.

## 1. Headline (revised)
`multi_hyde_diverse` is now a Llama 70b Tier 2 paper headline: N=200 is 35.5% vs `rag_simple` 27.5%, +8.0pp, McNemar p=0.0195. The cross-family narrative still does not survive confirmation because Gemma 27B N=200 is only +2.5pp with p=0.5901, a NULL result.

## 2. What we can cite (Tier 2+ confirmed, N>=200 or repeated cross-size)
- BarExam `rag_snap_hyde` remains the Tier 3 legal-MC positive: Gemma 4 26B-A4B full N=1195 is 81.17% vs `rag_simple` 78.08% (+3.09pp), and Gemma 4 E4B full N=1195 is 62.18% vs 58.49% (+3.69pp).
- MuSiQue Llama 70b `multi_hyde_diverse` is the Tier 2 multi-hop positive: 35.5% vs `rag_simple` 27.5% (+8.0pp, p=0.0195).
- Methodology fixes are citeable qualitatively: formatter and retrieval-query bugs were fixed before the stronger BarExam reruns.
- This list is now thin: MuSiQue mhd is no longer Tier 2 cross-family confirmed.

## 3. Tier 1 preliminary findings (direction-only, N=100)
- Preliminary / direction only / awaiting Tier 2: mhd x Llama 70b N=100 is +12.0pp over `rag_simple`, p=0.0227.
- Preliminary / direction only / awaiting Tier 2: mhd x Qwen3 30B MoE N=100 is 28% vs `rag_simple` 24% (+4pp).
- Small-N consistency signal, not a result citation: `iter_hyde` hurts Gemma 3 27B, Scout 17b, and Qwen3 30B, while Llama 70b is roughly flat; the direction is consistent but not Tier 2.
- Scout calibration example: N=200 `rag_multi_query` was 30.5% vs `rag_simple` 30.0%; the preliminary / direction-only N=100 -5pp multi-query hit was noise.

## 4. Mechanism decomposition
- Llama 70b N=200 logged `rag_simple` 27.5%, `rag_multi_query` 29.0% (+1.5pp, p=0.728 NS), and mhd 35.5% (+8.0pp, p=0.0195 SIG).
- The clean paper mechanism is now: HyDE-style answer-bearing passages do most of the work, about +6.5pp of the +8pp lift; diversity alone is a minor non-significant component.
- Scout and Qwen mechanism rows remain direction-only unless they land at N>=200.

## 5. Friend/foe attribution probe
- Gemma 27B: kept-snap rates were self/foe/control = 90.0%/90.0%/90.0%, with 4/30 outcome changes.
- Llama 70b: kept-snap rates were self/foe/control = 83.3%/83.3%/73.3%, with 6/30 outcome changes.
- Llama drops keep-rate under unattributed control, so the mechanism is real and not just tonal.
- Limited scope: only 10/60 records changed final answer under attribution variation, about 16.7%.

## 6. Methodology hardening shipped today
- Pre-flight gate caught DeepSeek auth failure and or-llama70b rate-limit failure cleanly.
- Empty-retrieval guard marked bogus empty-retrieval runs as failed instead of citeable.
- Silent-empty fallback fixes landed in mhd and `iter_hyde`.
- McNemar paired-test infrastructure now exists for N=200 confirmation.
- Open follow-up: add an mhd final-answer empty check.

## 7. Open questions for the meeting
- Does the Llama-only MHD Tier 2 win hold in the running Tier 3/full MuSiQue jobs?
- If Gemma 27B null is the real signal, should we frame MHD as model-class-specific rather than cross-family?
- Full corpus Tier 3 runs in flight: mhd-pair x Gemma 4 26B-A4B x N=2400 MuSiQue PID 2487 ~6-8hr (source needed); mhd-pair x Qwen3 30B MoE x N=2400 MuSiQue PID 2675 ~8-10hr (source needed); SLURM 55040 BarExam mhd/iter_hyde x Gemma 4 26B N=200 ~30 min more (source needed). When they land, do they survive?

## 8. Reproducibility footer
Branch `hpc-setup`, HEAD `44427ad`. All unmarked cited numbers trace to `docs/signoff_log.md`, `docs/compiled_results.md`, `docs/mcnemar_2026-04-27.md`, `docs/friend_foe_bias_analysis_2026-04-27.md`, or `logs/experiments.jsonl`. N<200 results are preliminary / direction-only.
