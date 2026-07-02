# Yoon-style leakage audit, BarExamQA 3SCOPE (Gemma 4 26B) - 2026-07-02

Question: does the SCOPE-family weak-query retrieval lift survive on
generations NOT entailed by the gold passage (unmatched), or is it
leakage-concentrated (Yoon et al. 2504.14175)?

Data: 3 exemplar-anchored SCOPE samples/question x 1192 questions;
per-sample Hit@5 vs shared raw-question Hit@5. NLI: nli-deberta-v3-base,
premise=gold passage, hypothesis=generated sentence, matched = max
p(entail) >= tau. CAVEAT: exemplar-anchored variant (canonical texts not
in repo); exemplar prompts bias matched-rate UP -> conservative for the
survives-on-unmatched claim.

Raw-question Hit@5 on these rows: 0.0143

| tau | samples matched | Hit@5 matched | lift matched | samples unmatched | Hit@5 unmatched | lift unmatched |
|---|---:|---:|---:|---:|---:|---:|
| 0.7 | 551 (15.4%) | 0.2849 | +27.22pp | 3025 | 0.0731 | +5.85pp |
| 0.8 | 523 (14.6%) | 0.2906 | +28.11pp | 3053 | 0.0740 | +5.90pp |
| 0.9 | 490 (13.7%) | 0.2959 | +28.57pp | 3086 | 0.0755 | +6.06pp |

Reading: 'lift' = Hit@5(sample) - Hit@5(raw) within the stratum. Yoon
predicts unmatched lift <= 0; the geometry account predicts unmatched
lift > 0 (smaller magnitude allowed).
