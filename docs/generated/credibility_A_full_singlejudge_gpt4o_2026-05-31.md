# Credibility A++ Single-Judge GPT-4o Factuality - 2026-05-31

No `paper/` files were edited.

## Verdict

Headline: **survives**.
The single-judge GPT-4o replication keeps factuality below the pre-stated AUC ceiling and adds little after geometry, so Phase A's geometry-over-factuality reading survives.

Pooled retrieval-hurt results: GPT-4o factuality AUC `0.548`, geometry AUC `0.823`, joint AUC `0.826`, marginal lift `0.003`. Gemma-vs-GPT-4o q200 IRR: Spearman `0.681`, kappa@0.5 `0.614`.

## Run Scope

OpenRouter usage moved from `$65.97` to `$87.93`. Task spend was `$21.96`. Provider total credits were `$95.00` at task start and `$120.00` at the final check, leaving `$32.07`. The configured hard cap was `$25` spend with roughly `$4` reserve.

All completed rows used `judge_provider=custom`, `judge_model=openai/gpt-4o`, and an empty OpenRouter provider route. No GPT-4o-mini substitution was used.

| Dataset | Complete questions | Judge records | Judge models | Provider routes |
|---|---:|---:|---|---|
| BarExamQA | 1195 | 4780 | `openai/gpt-4o x4780` | `{} x4780` |
| FiQA | 648 | 2592 | `openai/gpt-4o x2592` | `{} x2592` |
| NFCorpus | 323 | 1292 | `openai/gpt-4o x1292` | `{} x1292` |
| SciDocs | 520 | 2080 | `openai/gpt-4o x2080` | `{} x2080` |
| SciFact | 300 | 1200 | `openai/gpt-4o x1200` | `{} x1200` |
| TREC-COVID | 50 | 200 | `openai/gpt-4o x200` | `{} x200` |

HousingQA was not started in this wave: after BarExamQA plus the five BEIR datasets, a GPT-4o q1000 Housing pass would have risked the requested reserve. The task allowed Housing only after the higher-priority full-N BarExamQA and BEIR passes.

## Retrieval-Hurt AUC

| Dataset | N arms | Hurt rows | Hurt rate | Factuality AUC | Geometry AUC | Joint AUC | Marginal lift | Factuality beta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | 2390 | 22 | 0.9% | 0.524 | 0.909 | 0.921 | 0.012 | 0.101 |
| FiQA | 1296 | 484 | 37.3% | 0.536 | 0.782 | 0.782 | -0.000 | -0.150 |
| NFCorpus | 646 | 156 | 24.1% | 0.617 | 0.778 | 0.782 | 0.004 | -0.761 |
| SciDocs | 1040 | 194 | 18.7% | 0.553 | 0.816 | 0.816 | -0.000 | -0.311 |
| SciFact | 600 | 214 | 35.7% | 0.631 | 0.846 | 0.848 | 0.002 | -0.630 |
| TREC-COVID | 100 | 17 | 17.0% | 0.714 | 0.797 | 0.799 | 0.002 | -0.998 |
| Pooled | 6072 | 1087 | 17.9% | 0.548 | 0.823 | 0.826 | 0.003 | -0.212 |

`N arms` counts expansion arms, so each complete question contributes one HyDE row and one SCOPE row.

## Gemma-vs-GPT-4o IRR on q200 Overlap

| Dataset | Overlap arms | GPT-4o mean | Gemma mean | Spearman rho | Kappa@0.5 |
|---|---:|---:|---:|---:|---:|
| BarExamQA | 400 | 0.244 | 0.165 | 0.792 | 0.749 |
| FiQA | 400 | 0.338 | 0.169 | 0.621 | 0.459 |
| NFCorpus | 400 | 0.129 | 0.065 | 0.650 | 0.592 |
| SciDocs | 400 | 0.099 | 0.030 | 0.509 | 0.412 |
| SciFact | 400 | 0.200 | 0.151 | 0.711 | 0.681 |
| TREC-COVID | 100 | 0.245 | 0.150 | 0.639 | 0.569 |
| Pooled | 2100 | 0.204 | 0.118 | 0.681 | 0.614 |

## Reading

- The pre-stated survival gate is met: pooled factuality AUC is at or below `0.65`, and the AUC lift after geometry is at or below `+0.03`.
- Treat this as a single-judge replication, not the full A++ two-independent-judge closeout. The Claude/Sonnet second judge remains pending.

## Sources

- Judge cache: `docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl`
- Feature points: `docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31_points.jsonl`
- Gemma q200 comparison: `docs/generated/factuality_feature_points_q200_2026-05-28.jsonl`
- BEIR geometry source: `/tmp/beir_phase1_verification_2026-05-26_points.jsonl`
- Legal geometry source: `/tmp/affinity_margin_oncache_2026-05-26_points.jsonl`

## Reproduction

```bash
NO_SILENT_FALLBACK=1 LLM_PROVIDER=custom LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=openai/gpt-4o EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py --datasets barexam --limit 0 --provider custom --resume --output docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl --concurrency 8 --batch-size 64
NO_SILENT_FALLBACK=1 LLM_PROVIDER=custom LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=openai/gpt-4o EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py --datasets beir_scifact beir_nfcorpus beir_fiqa beir_trec_covid beir_scidocs --limit 0 --provider custom --resume --output docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl --concurrency 8 --batch-size 64
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_singlejudge_gpt4o_factuality.py --judge-cache docs/generated/factuality_judge_full_singlejudge_gpt4o_2026-05-31.jsonl --features-out docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31_points.jsonl --output docs/generated/credibility_A_full_singlejudge_gpt4o_2026-05-31.md --end-usage 87.929304895 --total-credits 120.00
```

