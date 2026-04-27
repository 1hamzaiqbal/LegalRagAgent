# 2026-04-27 Meeting Brief v2 (post-McNemar reckoning)

## 1. Headline (revised)
The cross-family `multi_hyde_diverse` lift narrative from earlier today did not survive Gemma 27B Tier 2 confirmation: N=200 moved from the preliminary / direction-only N=100 +8.0pp trend to only +2.5pp, with McNemar p=0.5901, a NULL result. Llama 70b N=100 still holds as preliminary / direction only at +12.0pp with p=0.0227. That Llama result cannot be promoted until N=200 lands; it is blocked until the after-meeting Groq TPD reset.

## 2. What we can cite (Tier 2+ confirmed, N>=200 or repeated cross-size)
- BarExam `rag_snap_hyde` remains the main surviving positive: E4B full logged 58.41% vs `rag_simple` 55.73% (+2.68pp), and 26B-A4B seed repeat logged 75.40% vs 71.80% (+3.60pp).
- Additional BarExam support exists at N=200: E4B postfix logged `rag_snap_hyde` 67.5% vs `rag_simple` 61.0% (+6.5pp), but this is not a full-corpus claim.
- Methodology fixes are citeable qualitatively: formatter and retrieval-query bugs were fixed before the stronger BarExam reruns.
- This list is now thin: MuSiQue mhd is no longer Tier 2 cross-family confirmed.

## 3. Tier 1 preliminary findings (direction-only, N=100)
- Preliminary / direction only / awaiting Tier 2: mhd x Llama 70b N=100 is +12.0pp over `rag_simple`, p=0.0227.
- Preliminary / direction only / awaiting Tier 2: mhd x Qwen3 30B MoE N=100 is 28% vs `rag_simple` 24% (+4pp).
- Small-N consistency signal, not a result citation: `iter_hyde` hurts Gemma 3 27B, Scout 17b, and Qwen3 30B, while Llama 70b is roughly flat; the direction is consistent but not Tier 2.
- Scout calibration example: N=200 `rag_multi_query` was 30.5% vs `rag_simple` 30.0%; the preliminary / direction-only N=100 -5pp multi-query hit was noise.

## 4. Mechanism decomposition (preliminary)
- Preliminary / direction only: Llama 70b N=100 logged `rag_simple` 21%, `rag_multi_query` 25% (+4pp diversity), and mhd 33% (+12pp total), implying HyDE-style passages add about +8pp beyond diversity.
- Preliminary / direction only: Scout N=100 logged `rag_simple` 30%, `rag_multi_query` 25% (-5pp), and mhd 29% (flat-ish), but Scout N=200 later showed multi-query was noise.
- Codex flagged that the mechanism could be interaction-dominated rather than strictly additive.
- Mechanism story stays preliminary until N=200 confirms on Llama and the decomposition controls.

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
- Is the mhd story salvageable with Llama 70b N=200, or does it null too?
- If Gemma 27B null is the real signal, should we pivot to a BarExam-only paper?
- Full corpus Tier 3 runs in flight: mhd-pair x Gemma 4 26B-A4B x N=2400 MuSiQue PID 2487 ~6-8hr (source needed); mhd-pair x Qwen3 30B MoE x N=2400 MuSiQue PID 2675 ~8-10hr (source needed); SLURM 55040 BarExam mhd/iter_hyde x Gemma 4 26B N=200 ~30 min more (source needed). When they land, do they survive?

## 8. Reproducibility footer
Branch `hpc-setup`, HEAD `83fb2fc`. All unmarked cited numbers trace to `docs/mcnemar_2026-04-27.md`, `docs/mhd_mechanism_2026-04-27.md`, `docs/friend_foe_bias_analysis_2026-04-27.md`, or `logs/experiments.jsonl`; live-run IDs/ETAs marked `(source needed)` were not present in those four sources. NO N=30 results are cited; the only N=30 mention is explicitly framed as a small-N consistency signal. All N=100 results are explicitly framed as preliminary / direction only.
