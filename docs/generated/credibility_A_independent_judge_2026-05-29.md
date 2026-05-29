# Credibility Battery Phase A - Independent Factuality Judge

Independent non-Gemma factuality judge analysis over the completed subset of the q200 feature sample. No `paper/` files were edited.

## Verdict

- Factuality falsification headline: **survives**. Original Gemma factuality AUC `0.581`, independent factuality AUC `0.586`, geometry AUC `0.816`, independent+geometry AUC `0.816`, marginal lift `0.000`.
- Inter-rater reliability: Spearman `0.671`, Pearson `0.691`, Cohen kappa `0.507` on score >= 0.75.
- Status: **provisional/rate-limited**. OpenRouter `openai/gpt-4o` completed 3,848 judge records before the monthly key limit stopped the run: all five BEIR sets are complete for the q200/q50 sample, BarExamQA is partial at 224 joined feature rows, and HousingQA has no independent-judge rows in this cache.

## AUC Table

| Dataset | Feature set | N | Failures | AUC | Pseudo-R2 |
|---|---|---:|---:|---:|---:|
| BarExamQA | OOV + logPPL | 224 | 3 | 0.704 | 0.116 |
| BarExamQA | Original gemma factuality | 224 | 3 | 0.665 | 0.052 |
| BarExamQA | Independent factuality | 224 | 3 | 0.710 | 0.070 |
| BarExamQA | Geometry | 224 | 3 | 0.962 | 0.271 |
| BarExamQA | Independent factuality + geometry | 224 | 3 | 0.991 | 0.421 |
| FiQA | OOV + logPPL | 400 | 155 | 0.583 | 0.009 |
| FiQA | Original gemma factuality | 400 | 155 | 0.586 | 0.029 |
| FiQA | Independent factuality | 400 | 155 | 0.597 | 0.027 |
| FiQA | Geometry | 400 | 155 | 0.803 | 0.225 |
| FiQA | Independent factuality + geometry | 400 | 155 | 0.805 | 0.226 |
| NFCorpus | OOV + logPPL | 400 | 91 | 0.583 | 0.010 |
| NFCorpus | Original gemma factuality | 400 | 91 | 0.571 | 0.045 |
| NFCorpus | Independent factuality | 400 | 91 | 0.622 | 0.059 |
| NFCorpus | Geometry | 384 | 91 | 0.777 | 0.161 |
| NFCorpus | Independent factuality + geometry | 384 | 91 | 0.776 | 0.165 |
| SciDocs | OOV + logPPL | 400 | 67 | 0.513 | 0.000 |
| SciDocs | Original gemma factuality | 400 | 67 | 0.536 | 0.024 |
| SciDocs | Independent factuality | 400 | 67 | 0.583 | 0.033 |
| SciDocs | Geometry | 400 | 67 | 0.824 | 0.243 |
| SciDocs | Independent factuality + geometry | 400 | 67 | 0.827 | 0.245 |
| SciFact | OOV + logPPL | 400 | 146 | 0.559 | 0.009 |
| SciFact | Original gemma factuality | 400 | 146 | 0.661 | 0.106 |
| SciFact | Independent factuality | 400 | 146 | 0.651 | 0.074 |
| SciFact | Geometry | 400 | 146 | 0.850 | 0.303 |
| SciFact | Independent factuality + geometry | 400 | 146 | 0.850 | 0.303 |
| TREC-COVID | OOV + logPPL | 100 | 17 | 0.584 | 0.017 |
| TREC-COVID | Original gemma factuality | 100 | 17 | 0.669 | 0.123 |
| TREC-COVID | Independent factuality | 100 | 17 | 0.714 | 0.126 |
| TREC-COVID | Geometry | 56 | 9 | 0.797 | 0.158 |
| TREC-COVID | Independent factuality + geometry | 56 | 9 | 0.799 | 0.168 |
| Pooled | OOV + logPPL | 1924 | 479 | 0.519 | 0.001 |
| Pooled | Original gemma factuality | 1924 | 479 | 0.581 | 0.031 |
| Pooled | Independent factuality | 1924 | 479 | 0.586 | 0.022 |
| Pooled | Geometry | 1864 | 471 | 0.816 | 0.234 |
| Pooled | Independent factuality + geometry | 1864 | 471 | 0.816 | 0.235 |

## Sources

- Original features: `docs/generated/factuality_feature_points_q200_2026-05-28.jsonl`
- Independent judge cache: `docs/generated/factuality_judge_independent_q200_2026-05-29.jsonl`
- Joined points: `docs/generated/credibility_A_independent_judge_2026-05-29_points.jsonl`
