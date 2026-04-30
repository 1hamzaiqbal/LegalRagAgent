# CaseHOLD Gold Mapping Repair - 2026-04-30

Purpose: fix the current CaseHOLD retrieval instrumentation gap before running
more answer-quality experiments.

## Problem

Current CaseHOLD audits show `gold_retrieved=0/200` for all arms. That should
not be interpreted as a retrieval failure. The local `test.csv` had no
`gold_idx`, and the holdings corpus was built only from train choices, so test
gold holdings were not reliably addressable as retrievable ids.

## Fix

Added `scripts/repair_casehold_gold_mapping.py` and updated
`utils/download_new_datasets.py casehold` to:

- assign each holding option a stable SHA1-derived id;
- add `gold_idx` to train/test rows;
- rebuild `holdings_corpus.csv` from all displayed train/test holdings;
- retain source provenance as `casehold_train`, `casehold_test`, or
  `casehold_train+test`.

Local repair output:

```text
train: 45000
test: 3600
holdings_corpus: 51296
```

## What This Enables

- Future CaseHOLD detail logs can compute meaningful `gold_retrieved`.
- We can audit whether retrieved holdings overlap the displayed choices,
  distractors, or the gold answer.
- CaseHOLD can become an option-disambiguation benchmark instead of an
  answer-only flatness probe.

## Remaining Work

The local and cluster Chroma collections must be rebuilt from the repaired
`datasets/casehold/holdings_corpus.csv` before new CaseHOLD retrieval runs.
Do not compare new gold-hit rates against old CaseHOLD logs.
