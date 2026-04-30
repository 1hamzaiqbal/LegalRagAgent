# SCALR Depth Disagreement Audit - 2026-04-30

Purpose: inspect whether the LegalBench-SCALR top-1/top-5/top-10 result is
mechanistically coherent enough to support the bottleneck-taxonomy narrative.
This is a no-new-API analysis over landed detail logs.

## Source Logs

| Depth | Detail log | Result |
|---|---|---:|
| top-1 | `logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl` | 119/200 (59.5%) |
| top-5 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl` | 154/200 (77.0%) |
| top-10 | `logs/eval_rag_simple_groq-llama70b_20260430_0054_detail.jsonl` | 154/200 (77.0%) |

## Log Sanity

- All three logs have the same 200 unique `idx` values in the same order.
- All rows are `dataset=legalbench_scalr`, `provider=groq-llama70b`,
  `mode=rag_simple`.
- Evidence counts are exactly the requested depths: top-1 has 1 document/row,
  top-5 has 5, and top-10 has 10; empty retrieval rows = 0.
- Error rows = 0 and missing predictions = 0 in all three logs.
- `scripts/analyze_detail_flags.py` reports 0 top-level HyDE/report/knowledge
  artifacts and 0 nested artifacts for all three logs.
- `gold_idx` is populated and supports `gold_retrieved`. `gold_passage` is
  blank in these legacy SCALR logs, so qualitative gold-text inspection should
  read `datasets/legalbench_scalr/test.csv`.
- Found and patched a detail-log serialization bug: these legacy SCALR logs
  record only `choices.A-D`, while SCALR is 5-way and has `choice_e` in the CSV.
  The prompt/scoring path already used `format_casehold_prompt` and
  `extract_answer_mc5`, so the accuracy rows remain usable. Do not use the
  legacy logged `choices` field for SCALR qualitative option analysis.

## Aggregate Pattern

| Depth | Accuracy | Gold retrieved | Acc when gold retrieved | Acc when gold missing | Input tok/q | Output tok/q |
|---|---:|---:|---:|---:|---:|---:|
| top-1 | 59.5% | 65/200 (32.5%) | 64/65 (98.5%) | 55/135 (40.7%) | 518 | 430 |
| top-5 | 77.0% | 108/200 (54.0%) | 102/108 (94.4%) | 52/92 (56.5%) | 723 | 423 |
| top-10 | 77.0% | 126/200 (63.0%) | 115/126 (91.3%) | 39/74 (52.7%) | 979 | 417 |

Paired tests:

| Pair | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| top-5 vs top-1 | +17.5pp | 38/3 | 1.05e-08 | [+12.0, +23.5] pp |
| top-10 vs top-1 | +17.5pp | 42/7 | 3.62e-07 | [+11.0, +24.0] pp |
| top-10 vs top-5 | 0.0pp | 8/8 | 1.0 | [-4.0, +4.0] pp |

## Disagreement Shape

Correctness patterns are `(top-1, top-5, top-10)`:

| Pattern | Rows | Interpretation |
|---|---:|---|
| F/F/F | 37 | not solved by depth |
| F/F/T | 6 | only top-10 rescue |
| F/T/F | 2 | top-5 rescue, top-10 regression |
| F/T/T | 36 | stable top-5/top-10 rescue |
| T/F/F | 1 | top-1 only |
| T/F/T | 2 | top-10 recovers top-1 answer |
| T/T/F | 6 | top-10 regression |
| T/T/T | 110 | solved at all depths |

Gold-retrieval patterns are also monotone-ish but not strictly prefix-based:

| Pattern | Rows |
|---|---:|
| F/F/F | 73 |
| F/F/T | 18 |
| F/T/T | 44 |
| T/F/F | 1 |
| T/T/T | 64 |

The important rescue buckets:

- top-5 rescues 38 top-1 misses; 22/38 add the gold holding at top-5, while
  16/38 still lack `gold_idx` and are probably helped by semantically adjacent
  holdings or the model's parametric/legal reasoning.
- top-1 beats top-5 on only 3 rows.
- top-10 rescues 8 top-5 misses, but top-10 also regresses on 8 top-5-correct
  rows, producing no net accuracy gain.
- top-10 retrieves more gold holdings than top-5 (126 vs 108), but the extra
  recall does not translate into net answer accuracy.

## Retrieval-Depth Caveat

The harness does not compare a fixed ranked list truncated to k. Retrieval uses
a `k*3` first-stage pool before cross-encoder reranking, so changing k changes
both candidate-pool breadth and final evidence depth.

Observed list behavior:

- top-1 doc equals top-5 first doc on 119/200 rows.
- top-1 doc appears anywhere in top-5 on 188/200 rows and anywhere in top-10 on
  189/200 rows.
- top-5 exact list equals the first five top-10 docs on only 49/200 rows.
- top-5 set is a subset of the top-10 set on 188/200 rows; average top-5/top-10
  overlap is 4.935/5 documents.

Therefore the defensible wording is **retrieval-depth policy** or **candidate
set size**, not a pure final-context truncation experiment.

## Example Checks

- `scalr_76`: top-1 retrieves a plausible distractor and predicts D. Top-5
  adds the gold holding at rank 3 and flips to the correct A; top-10 stays
  correct.
- `scalr_509`: no depth retrieves `gold_idx`, but top-5/top-10 still flip from
  D to the correct C. This supports the "semantic neighborhood can help even
  without exact gold hit" caveat.
- `scalr_81`: top-10 adds the gold holding at rank 8 and rescues a top-5 miss.
  This is a real top-10 benefit case, but rare.
- `scalr_286`: neither top-5 nor top-10 retrieves `gold_idx`; top-5 predicts
  the correct D, while top-10 regresses to B after adding/reordering additional
  related habeas/AEDPA holdings. This is the extra-context distraction case.

## Bottom Line

The SCALR story survives log inspection, with one logging caveat now fixed for
future runs. SCALR is not "more depth is always better"; it is "top-1 is too
narrow, top-5 supplies the needed candidate set, and top-10 adds recall without
net answer gain." This is useful for the paper narrative because it makes the
intervention target specific: route to a small candidate set when the task is
candidate-depth limited, then stop before extra context becomes noise.
