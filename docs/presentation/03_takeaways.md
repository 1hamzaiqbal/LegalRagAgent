# Takeaways

- BarExam has a single-domain winner: `rag_snap_hyde` wins cross-size at full corpus, lifting Gemma 4 26B-A4B by +3.09pp and Gemma 4 E4B by +3.69pp.

📊 [`figures/02_barexam_cross_size.png`](figures/02_barexam_cross_size.png) · [`figures/06_barexam_26b_full_matrix.png`](figures/06_barexam_26b_full_matrix.png)

- BarExam snap agreement is a caveat, not a rejection. Gemma 4 has strong legal priors, snap reasoning dominates, and HyDE supplies marginal lift or occasional conflicting evidence; this is BY DESIGN architecture and mechanism understanding.

- MuSiQue has a multi-hop winner on Llama 70b: `multi_hyde_diverse` is the clean significant simple-method lift at N=200, 35.5%, +8pp, p=0.0195. * (Tier 2 paired; full-corpus replicate would solidify).

📊 [`figures/01_llama70b_method_matrix.png`](figures/01_llama70b_method_matrix.png) — paper headline 8-method bar chart

- Complex methods are not uniformly bad. `iterative_planning_table` reaches 36.0%, +8.5pp, p=0.0533 TRENDING-SIG. * (Tier 2 paired; full-corpus replicate would solidify). `subagent_rag` is a real -12.0pp finding because current gap-routing over-abstains, and prompt reframing could likely close part of that gap.

- The mechanism is not just query diversity. `rag_multi_query` adds only +1.5pp and is non-significant, while HyDE-style answer passages explain about +6.5pp of the +8pp `multi_hyde_diverse` lift.

📊 [`figures/03_mechanism_decomposition.png`](figures/03_mechanism_decomposition.png) — diversity vs HyDE split

- The lifts are task-specific. The BarExam method `rag_snap_hyde` is negative on MuSiQue, and the MuSiQue method `multi_hyde_diverse` is negative on BarExam paired N=200.

📊 [`figures/04_cross_domain_specificity.png`](figures/04_cross_domain_specificity.png) — native vs cross-domain Δ

- The cross-family story is still a caveat, not a universal claim. Llama 70b is significant, but Gemma 3 27B is 31.0%, +2.5pp, p=0.5901 NULL.

📊 [`figures/05_cross_family_check.png`](figures/05_cross_family_check.png) — Llama SIG vs Gemma 3 27B NULL

📂 All figures + captions: [`figures/captions.md`](figures/captions.md)

- Methodology is now much harder to fool: N<200 is directional only, per-entry audits gate citations, paired McNemar is the default, and the OR-Gemma serving issue was caught before it became a Tier 3 claim.

Source: docs/signoff_log.md Sections A.3, B.1, B.2, B.3, B.4, and D'; subagent caveat from docs/audits/2026-04-27_llama70b_musique_audit.md Section 8.

## What we'd do next

1. Finish the in-flight full-corpus BarExam replicate for `multi_hyde_diverse` and `iter_hyde` on Gemma 4 26B-A4B, then audit before citation.

2. Replicate `iterative_planning_table` cross-model at N=200 or higher, because it is the best raw Llama EM but sits just outside p<0.05.

3. Re-test Gemma multi-hop methods on cluster vLLM rather than OR-served Gemma, since the serving issue creates runaway-loop and latency confounds.

4. Treat `multi_hyde_diverse` as the next MuSiQue mainline, but require cross-family or full-corpus confirmation before calling it universal.
