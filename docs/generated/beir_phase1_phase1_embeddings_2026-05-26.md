# BEIR Phase 1 Embeddings - 2026-05-26

Phase 1 embedded normalized BEIR corpora into Chroma at `/home/techguy227/grad/LegalRagAgent/chroma_db` with `Alibaba-NLP/gte-large-en-v1.5`. No files under `paper/` were edited.
Embedding inputs were capped at `4096` characters before tokenization while full document text was stored in Chroma; the model itself is configured with `max_seq_length=512`.

| Dataset | Eval key | Collection | Raw corpus docs | Embedded docs expected | Empty-text docs | Before | After | Missing gold docs | Inserted this run | Status | Elapsed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| scifact | `beir_scifact` | `beir_scifact` | 5183 | 5183 | 0 | 5183 | 5183 | 0 | 0 | already_complete | 0.0s |
| nfcorpus | `beir_nfcorpus` | `beir_nfcorpus` | 3633 | 3633 | 0 | 3633 | 3633 | 0 | 0 | already_complete | 0.0s |
| fiqa | `beir_fiqa` | `beir_fiqa` | 57638 | 57638 | 38 | 57600 | 57638 | 0 | 38 | ok | 0.3s |
| trec-covid | `beir_trec_covid` | `beir_trec_covid` | 171332 | 171332 | 1 | 171331 | 171332 | 0 | 1 | ok | 0.9s |
| scidocs | `beir_scidocs` | `beir_scidocs` | 25657 | 25657 | 0 | 25657 | 25657 | 0 | 0 | already_complete | 0.0s |

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/embed_beir_phase1.py
```
