# CaseHOLD Llama 70B Flatness Audit - 2026-04-30

Join key: `idx`. Common paired rows: 200.

## Source Logs

- `top5`: `logs/eval_rag_simple_groq-llama70b_20260428_0259_detail.jsonl`
- `top1`: `logs/eval_rag_simple_groq-llama70b_20260429_2318_detail.jsonl`
- `two_call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0309_detail.jsonl`

## Run Summary

| Run | Accuracy | Gold retrieved | Gold-hit count histogram | Avg LLM calls | Avg input toks | Avg output toks |
|---|---:|---:|---|---:|---:|---:|
| `top5` | 144/200 (72.0%) | 0/200 (0.0%) | 0:200 | 1.00 | 672 | 438 |
| `top1` | 141/200 (70.5%) | 0/200 (0.0%) | 0:200 | 1.00 | 521 | 436 |
| `two_call` | 139/200 (69.5%) | 0/200 (0.0%) | 0:200 | 2.00 | 1232 | 700 |

## Pairwise vs Baseline

| Treatment | Rescued | Harmed | Net | Rescued gold delta | Harmed gold delta | Answer changed | Rescued subjects |
|---|---:|---:|---:|---|---|---:|---|
| `top1` | 10 | 13 | -3 | same_gold=10 | same_gold=13 | 23/23 | casehold=10 |
| `two_call` | 14 | 19 | -5 | same_gold=14 | same_gold=19 | 33/33 | casehold=14 |

## Complementarity

- Static best in this family: 144/200 (72.0%).
- Oracle any-correct across all listed runs: 163/200 (81.5%).
- All listed runs wrong: 37/200 (18.5%).

## Interpretation

- CaseHOLD flatness is real at the answer level: top-1, top-5, and two-call
  differ by only -1.5pp to -2.5pp against top-5, with small symmetric flips.
- The current `gold_retrieved` field is not informative for this dataset:
  all three logs show 0/200 gold retrieved and a 0-hit histogram. Do not use
  these rows as retrieval-recall evidence until CaseHOLD gold-option/holding
  ids are mapped to retrievable ids.
- The better read is "candidate-depth insensitive under current harness," not
  "retrieval is solved." The 81.5% any-correct oracle still shows per-question
  complementarity, but the available cheap arms mostly swap answers rather than
  giving a stable lift.
- Next audit target for CaseHOLD is not another LLM run; it is retrieval-only
  instrumentation that can tell whether the correct holding option appears in
  the retrieved context.

### Rescue Overlap

| Set | Count | Example ids |
|---|---:|---|
| `top1` rescues | 10 | ch_test_1174, ch_test_1637, ch_test_1765, ch_test_1788 (+6 more) |
| `top1` & `two_call` | 5 | ch_test_1174, ch_test_1765, ch_test_1788, ch_test_1957 (+1 more) |
| `two_call` rescues | 14 | ch_test_1174, ch_test_149, ch_test_1765, ch_test_1788 (+10 more) |

## Correctness Patterns

Pattern bit order: `top5`, `top1`, `two_call`.

| Pattern | Count | Example ids |
|---|---:|---|
| `111` | 117 | ch_test_1025, ch_test_1044, ch_test_1053, ch_test_1057 (+113 more) |
| `000` | 37 | ch_test_1340, ch_test_1397, ch_test_1465, ch_test_1480 (+33 more) |
| `110` | 14 | ch_test_1108, ch_test_120, ch_test_1877, ch_test_1936 (+10 more) |
| `001` | 9 | ch_test_149, ch_test_1793, ch_test_2321, ch_test_2329 (+5 more) |
| `101` | 8 | ch_test_1488, ch_test_170, ch_test_196, ch_test_2696 (+4 more) |
| `011` | 5 | ch_test_1174, ch_test_1765, ch_test_1788, ch_test_1957 (+1 more) |
| `100` | 5 | ch_test_1236, ch_test_1298, ch_test_2736, ch_test_351 (+1 more) |
| `010` | 5 | ch_test_1637, ch_test_179, ch_test_1953, ch_test_2453 (+1 more) |

## Mechanism Read

- Improvements are not pure answer-format noise: every baseline/treatment flip changed the normalized answer string in this audit.
- `more_gold` in rescued rows means the treatment retrieved more gold passage ids than baseline; `same_gold` means the gain came despite equivalent gold-id count.
- Treat this as a mechanism screen, not final causality: exact-match scoring and multi-hop aliasing can still hide semantically acceptable answers.
