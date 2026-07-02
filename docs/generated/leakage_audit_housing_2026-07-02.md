# Yoon-style leakage audit, HousingQA statefilter 3SCOPE (Gemma) - 2026-07-02

Subsample 1200 questions (hydrated-gold, seed 42); dense-stage
per-sample retrieval vs shared raw component. Matched = any generated
sentence entailed by gold statute text (nli-deberta-v3-base).

Raw (dense component) Hit@5: 0.3525

| tau | matched | Hit@5 matched (lift) | unmatched | Hit@5 unmatched (lift) |
|---|---:|---:|---:|---:|
| 0.7 | 254 (7.1%) | 0.4567 (+9.45pp) | 3346 | 0.3595 (+0.78pp) |
| 0.8 | 218 (6.1%) | 0.4450 (+9.63pp) | 3382 | 0.3613 (+0.86pp) |
| 0.9 | 179 (5.0%) | 0.4302 (+8.94pp) | 3421 | 0.3631 (+0.99pp) |