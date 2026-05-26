# BEIR Phase 1 - Phase 2 Loader Wiring

Date: 2026-05-26  
Branch: `scope-generalization`  
Scope: results-lane code only; no `paper/` edits.

## What Changed

- Added a `beir_*` dataset family in `eval/eval_config.py`.
- Registered five BEIR test-qrel subsets:
  - `beir_scifact` -> `datasets/beir/scifact/questions.csv`
  - `beir_nfcorpus` -> `datasets/beir/nfcorpus/questions.csv`
  - `beir_fiqa` -> `datasets/beir/fiqa/questions.csv`
  - `beir_trec_covid` -> `datasets/beir/trec-covid/questions.csv`
  - `beir_scidocs` -> `datasets/beir/scidocs/questions.csv`
- Registered matching Chroma collections in `eval/eval_harness.py`.
- Preserved qrels as multi-gold `gold_idx` JSON sets; existing retrieval scoring already treats any gold id match as a hit.
- Added BEIR-specific retrieval/generation prompts.
- Relaxed Snap-HyRE/SCOPE strict final-answer-line parsing only for non-answer-contract datasets such as BEIR. The existing strict final-answer guards remain unchanged for BarExamQA, HousingQA, CaseHOLD, LegalBench-SCALR, MASLegalBench, Legal-Link-EU, and MedQA.
- Added BEIR choices to `scripts/build_generation_cache.py` and `scripts/build_retrieval_cache.py`.

## Verification

Compile check:

```bash
python3 -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  scripts/build_generation_cache.py \
  scripts/build_retrieval_cache.py
```

Loader sanity check, with offline HF flags:

| Dataset | Loaded test queries | First row label | Collection | First-row gold count |
|---|---:|---|---|---:|
| `beir_scifact` | 300 | `beir_scifact_1` | `beir_scifact` | 1 |
| `beir_nfcorpus` | 323 | `beir_nfcorpus_PLAIN-1008` | `beir_nfcorpus` | 5 |
| `beir_fiqa` | 648 | `beir_fiqa_10034` | `beir_fiqa` | 2 |
| `beir_trec_covid` | 50 | `beir_trec_covid_1` | `beir_trec_covid` | 637 |
| `beir_scidocs` | 1000 | `beir_scidocs_01273bd34dacfe9ef887b320f36934d2f9fa9b34` | `beir_scidocs` | 5 |

Smoke retrieval command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_SILENT_FALLBACK=1 \
uv run python scripts/build_retrieval_cache.py \
  --dataset beir_scifact \
  --questions 3 \
  --seed 42 \
  --query-type raw_question \
  --max-k 10 \
  --collection beir_scifact \
  --out /tmp/beir_phase2_scifact_raw_q3.jsonl \
  --progress-interval 1
```

Smoke retrieval result:

| Label | Retrieved ids | Gold ids | Gold retrieved |
|---|---:|---:|---|
| `beir_scifact_569` | 10 | 1 | true |
| `beir_scifact_832` | 10 | 1 | true |
| `beir_scifact_385` | 10 | 2 | true |

The q3 smoke confirms the BEIR loader, row labels, collection routing, Chroma retrieval, qrel gold sets, and hit scoring are connected.
