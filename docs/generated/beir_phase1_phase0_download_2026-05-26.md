# BEIR Phase 1 Download - 2026-05-26

Phase 0 downloaded BEIR corpus, query, and test-qrels splits from Hugging Face and normalized local CSVs under `datasets/beir/`. No files under `paper/` were edited.

| Dataset | Eval key | Corpus docs | HF queries | Test queries with qrels | Test qrels | Positive qrels | Gold docs | Multi-gold queries | Missing query ids | Missing corpus ids |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scifact | `beir_scifact` | 5183 | 1109 | 300 | 339 | 339 | 283 | 23 | 0 | 0 |
| nfcorpus | `beir_nfcorpus` | 3633 | 3237 | 323 | 12334 | 12334 | 3128 | 300 | 0 | 0 |
| fiqa | `beir_fiqa` | 57638 | 6648 | 648 | 1706 | 1706 | 1706 | 428 | 0 | 0 |
| trec-covid | `beir_trec_covid` | 171332 | 50 | 50 | 66336 | 24673 | 17537 | 50 | 0 | 0 |
| scidocs | `beir_scidocs` | 25657 | 1000 | 1000 | 29928 | 4928 | 4020 | 1000 | 0 | 0 |

Local normalized files are intentionally under ignored `datasets/`; the committed artifact is this count report and the download script.

## Reproduction

```bash
uv run python scripts/download_beir_phase1.py
```
