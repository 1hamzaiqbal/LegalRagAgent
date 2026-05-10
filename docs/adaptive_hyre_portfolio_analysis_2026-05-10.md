# Adaptive HyRE Portfolio Analysis

Date: 2026-05-10

Source: joined N=200 detail logs from the HPC adaptive-HyRE checkout. This is
log-only analysis; it performs no fresh retrieval, embedding, or LLM calls.

## Cross-Dataset Takeaways

- Barexam has real portfolio headroom: `adaptive_snap_hyre_v2` reaches 86.0%,
  any-method oracle reaches 92.0%, and disagreement rows have 94.1% oracle
  accuracy. A targeted disagreement arbitrator is worth testing here.
- Housing is no longer a query-diversity problem in this snapshot. The
  conservative verifier is the best single method at 74.5%, while majority
  voting collapses to 63.5%. The controller should route Housing to statutory
  entailment verification, not vote over retrieval variants.
- CaseHOLD has 81.5% oracle headroom, but simple majority falls below the best
  single method. The hard slice is answer-option conversion under disagreement,
  consistent with the rule-frame replay rejection.
- LegalBench-SCALR is close to parity: the frontier route reaches 76.5% and
  majority voting does not improve it. The disagreement rows are still highly
  recoverable by oracle, so a better arbitrator could help, but broad reruns are
  unlikely to be efficient.

# Adaptive Portfolio Analysis: barexam

Rows joined: 200

## Single Methods

| Method | Correct | Accuracy | Avg calls | Gold retrieved | Route / policy |
|---|---:|---:|---:|---:|---|
| `rag_simple` | 160/200 | 80.0% | 1.00 | 5/200 |  |
| `adaptive_snap_hyre_v2` | 172/200 | 86.0% | 2.00 | 18/200 | barexam_option_grounding:200 |
| `adaptive_snap_hyre_frontier` | 168/200 | 84.0% | 2.00 | 17/200 | frontier_barexam_v2:200 |

## Headroom

- Best single method: `adaptive_snap_hyre_v2` at 172/200 = 86.0%.
- Any-method oracle: 184/200 = 92.0%.
- All methods correct: 145/200 = 72.5%.
- No method correct: 16/200 = 8.0%.
- Recoverable above best single method: 12 rows.

## Deterministic Portfolio Policies

| Policy | Correct | Accuracy | Avg calls counted | Delta vs best |
|---|---:|---:|---:|---:|
| `majority` | 173/200 | 86.5% | 5.00 | +1 |

## Agreement Buckets

Primary pair: `rag_simple` vs `adaptive_snap_hyre_v2`

| Bucket | Rows | Primary acc | Secondary acc | Oracle acc |
|---|---:|---:|---:|---:|
| agree | 166 | 150/166 = 90.4% | 150/166 = 90.4% | 152/166 = 91.6% |
| disagree | 34 | 10/34 = 29.4% | 22/34 = 64.7% | 32/34 = 94.1% |

## Disagreement Examples

- `qa_CONST. LAW_mbe_1041` gold=C | rag_simple=C* adaptive_snap_hyre_v2=D adaptive_snap_hyre_frontier=D
- `qa_CONST. LAW_mbe_599` gold=A | rag_simple=B adaptive_snap_hyre_v2=A* adaptive_snap_hyre_frontier=A*
- `qa_CONTRACTS_mbe_1015` gold=C | rag_simple=C* adaptive_snap_hyre_v2=A adaptive_snap_hyre_frontier=C*
- `qa_CONTRACTS_mbe_672` gold=B | rag_simple=A adaptive_snap_hyre_v2=B* adaptive_snap_hyre_frontier=B*
- `qa_CONTRACTS_mbe_824` gold=A | rag_simple=C adaptive_snap_hyre_v2=A* adaptive_snap_hyre_frontier=A*
- `qa_CRIM. LAW_mbe_1065` gold=B | rag_simple=C adaptive_snap_hyre_v2=B* adaptive_snap_hyre_frontier=B*
- `qa_CRIM. LAW_mbe_1107` gold=D | rag_simple=B adaptive_snap_hyre_v2=D* adaptive_snap_hyre_frontier=D*
- `qa_CRIM. LAW_mbe_565` gold=C | rag_simple=D adaptive_snap_hyre_v2=C* adaptive_snap_hyre_frontier=D
- `qa_CRIM. LAW_mbe_620` gold=C | rag_simple=B adaptive_snap_hyre_v2=D adaptive_snap_hyre_frontier=D
- `qa_EVIDENCE_mbe_1075` gold=B | rag_simple=C adaptive_snap_hyre_v2=B* adaptive_snap_hyre_frontier=B*


# Adaptive Portfolio Analysis: housing

Rows joined: 200

## Single Methods

| Method | Correct | Accuracy | Avg calls | Gold retrieved | Route / policy |
|---|---:|---:|---:|---:|---|
| `rag_state_filter` | 121/200 | 60.5% | 1.00 | 81/200 |  |
| `snap_hyre_state` | 126/200 | 63.0% | 2.00 | 81/200 | state_filter:200 |
| `adaptive_snap_hyre_diverse` | 127/200 | 63.5% | 2.00 | 92/200 | state_filter:200 |
| `adaptive_snap_hyre_frontier` | 124/200 | 62.0% | 2.00 | 89/200 | frontier_housing_diverse:200 |
| `adaptive_snap_hyre_housing_verifier` | 149/200 | 74.5% | 1.00 | 89/200 | housing_yes_no_verifier:200 |

## Headroom

- Best single method: `adaptive_snap_hyre_housing_verifier` at 149/200 = 74.5%.
- Any-method oracle: 169/200 = 84.5%.
- All methods correct: 94/200 = 47.0%.
- No method correct: 31/200 = 15.5%.
- Recoverable above best single method: 20 rows.

## Deterministic Portfolio Policies

| Policy | Correct | Accuracy | Avg calls counted | Delta vs best |
|---|---:|---:|---:|---:|
| `majority` | 127/200 | 63.5% | 8.00 | -22 |

## Agreement Buckets

Primary pair: `rag_state_filter` vs `snap_hyre_state`

| Bucket | Rows | Primary acc | Secondary acc | Oracle acc |
|---|---:|---:|---:|---:|
| agree | 171 | 109/171 = 63.7% | 109/171 = 63.7% | 140/171 = 81.9% |
| disagree | 29 | 12/29 = 41.4% | 17/29 = 58.6% | 29/29 = 100.0% |

## Disagreement Examples

- `hqa_Alabama_2544` gold=No | rag_state_filter=Yes snap_hyre_state=Yes adaptive_snap_hyre_diverse=Yes adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_Alaska_6609` gold=No | rag_state_filter=No* snap_hyre_state=No* adaptive_snap_hyre_diverse=No* adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_Arizona_3` gold=Yes | rag_state_filter=Yes* snap_hyre_state=Yes* adaptive_snap_hyre_diverse=Yes* adaptive_snap_hyre_frontier=Yes* adaptive_snap_hyre_housing_verifier=No
- `hqa_California_3980` gold=No | rag_state_filter=Yes snap_hyre_state=Yes adaptive_snap_hyre_diverse=Yes adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_California_3981` gold=No | rag_state_filter=Yes snap_hyre_state=Yes adaptive_snap_hyre_diverse=Yes adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_California_3984` gold=No | rag_state_filter=Yes snap_hyre_state=Yes adaptive_snap_hyre_diverse=Yes adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_California_6654` gold=No | rag_state_filter=Yes snap_hyre_state=Yes adaptive_snap_hyre_diverse=Yes adaptive_snap_hyre_frontier=Yes adaptive_snap_hyre_housing_verifier=No*
- `hqa_California_7572` gold=Yes | rag_state_filter=No snap_hyre_state=Yes* adaptive_snap_hyre_diverse=No adaptive_snap_hyre_frontier=No adaptive_snap_hyre_housing_verifier=No
- `hqa_California_913` gold=Yes | rag_state_filter=Yes* snap_hyre_state=Yes* adaptive_snap_hyre_diverse=Yes* adaptive_snap_hyre_frontier=Yes* adaptive_snap_hyre_housing_verifier=No
- `hqa_Colorado_6675` gold=Yes | rag_state_filter=Yes* snap_hyre_state=Yes* adaptive_snap_hyre_diverse=Yes* adaptive_snap_hyre_frontier=Yes* adaptive_snap_hyre_housing_verifier=No


# Adaptive Portfolio Analysis: casehold

Rows joined: 200

## Single Methods

| Method | Correct | Accuracy | Avg calls | Gold retrieved | Route / policy |
|---|---:|---:|---:|---:|---|
| `rag_simple` | 146/200 | 73.0% | 1.00 | 32/200 |  |
| `adaptive_snap_hyre_diverse` | 147/200 | 73.5% | 2.00 | 38/200 | option_grounding:200 |
| `adaptive_snap_hyre_frontier` | 141/200 | 70.5% | 2.00 | 42/200 | frontier_casehold_diverse:200 |

## Headroom

- Best single method: `adaptive_snap_hyre_diverse` at 147/200 = 73.5%.
- Any-method oracle: 163/200 = 81.5%.
- All methods correct: 127/200 = 63.5%.
- No method correct: 37/200 = 18.5%.
- Recoverable above best single method: 16 rows.

## Deterministic Portfolio Policies

| Policy | Correct | Accuracy | Avg calls counted | Delta vs best |
|---|---:|---:|---:|---:|
| `majority` | 145/200 | 72.5% | 5.00 | -2 |

## Agreement Buckets

Primary pair: `rag_simple` vs `adaptive_snap_hyre_diverse`

| Bucket | Rows | Primary acc | Secondary acc | Oracle acc |
|---|---:|---:|---:|---:|
| agree | 159 | 132/159 = 83.0% | 132/159 = 83.0% | 133/159 = 83.6% |
| disagree | 41 | 14/41 = 34.1% | 15/41 = 36.6% | 30/41 = 73.2% |

## Disagreement Examples

- `ch_ch_test_1174` gold=A | rag_simple=D adaptive_snap_hyre_diverse=A* adaptive_snap_hyre_frontier=E
- `ch_ch_test_1236` gold=E | rag_simple=C adaptive_snap_hyre_diverse=E* adaptive_snap_hyre_frontier=E*
- `ch_ch_test_1340` gold=D | rag_simple=A adaptive_snap_hyre_diverse=B adaptive_snap_hyre_frontier=C
- `ch_ch_test_1397` gold=A | rag_simple=C adaptive_snap_hyre_diverse=C adaptive_snap_hyre_frontier=B
- `ch_ch_test_1465` gold=E | rag_simple=D adaptive_snap_hyre_diverse=C adaptive_snap_hyre_frontier=C
- `ch_ch_test_1480` gold=D | rag_simple=C adaptive_snap_hyre_diverse=D* adaptive_snap_hyre_frontier=C
- `ch_ch_test_149` gold=E | rag_simple=E* adaptive_snap_hyre_diverse=D adaptive_snap_hyre_frontier=E*
- `ch_ch_test_1509` gold=A | rag_simple=A* adaptive_snap_hyre_diverse=B adaptive_snap_hyre_frontier=B
- `ch_ch_test_1539` gold=D | rag_simple=A adaptive_snap_hyre_diverse=B adaptive_snap_hyre_frontier=B
- `ch_ch_test_1554` gold=B | rag_simple=A adaptive_snap_hyre_diverse=D adaptive_snap_hyre_frontier=D


# Adaptive Portfolio Analysis: legalbench_scalr

Rows joined: 200

## Single Methods

| Method | Correct | Accuracy | Avg calls | Gold retrieved | Route / policy |
|---|---:|---:|---:|---:|---|
| `rag_simple` | 148/200 | 74.0% | 1.00 | 108/200 |  |
| `rag_snap_hyde_2call` | 152/200 | 76.0% | 2.00 | 146/200 |  |
| `adaptive_snap_hyre_frontier` | 153/200 | 76.5% | 2.00 | 153/200 | frontier_scalr_plain_snap_hyde:200 |

## Headroom

- Best single method: `adaptive_snap_hyre_frontier` at 153/200 = 76.5%.
- Any-method oracle: 167/200 = 83.5%.
- All methods correct: 133/200 = 66.5%.
- No method correct: 33/200 = 16.5%.
- Recoverable above best single method: 14 rows.

## Deterministic Portfolio Policies

| Policy | Correct | Accuracy | Avg calls counted | Delta vs best |
|---|---:|---:|---:|---:|
| `majority` | 153/200 | 76.5% | 5.00 | +0 |

## Agreement Buckets

Primary pair: `rag_simple` vs `rag_snap_hyde_2call`

| Bucket | Rows | Primary acc | Secondary acc | Oracle acc |
|---|---:|---:|---:|---:|
| agree | 172 | 137/172 = 79.7% | 137/172 = 79.7% | 140/172 = 81.4% |
| disagree | 28 | 11/28 = 39.3% | 15/28 = 53.6% | 27/28 = 96.4% |

## Disagreement Examples

- `qa_unknown_scalr_0` gold=D | rag_simple=C rag_snap_hyde_2call=D* adaptive_snap_hyre_frontier=D*
- `qa_unknown_scalr_10` gold=E | rag_simple=B rag_snap_hyde_2call=E* adaptive_snap_hyre_frontier=E*
- `qa_unknown_scalr_124` gold=D | rag_simple=D* rag_snap_hyde_2call=B adaptive_snap_hyre_frontier=B
- `qa_unknown_scalr_153` gold=C | rag_simple=C* rag_snap_hyde_2call=C* adaptive_snap_hyre_frontier=D
- `qa_unknown_scalr_172` gold=E | rag_simple=E* rag_snap_hyde_2call=C adaptive_snap_hyre_frontier=E*
- `qa_unknown_scalr_177` gold=B | rag_simple=A rag_snap_hyde_2call=B* adaptive_snap_hyre_frontier=A
- `qa_unknown_scalr_184` gold=C | rag_simple=C* rag_snap_hyde_2call=C* adaptive_snap_hyre_frontier=A
- `qa_unknown_scalr_263` gold=E | rag_simple=E* rag_snap_hyde_2call=D adaptive_snap_hyre_frontier=E*
- `qa_unknown_scalr_265` gold=D | rag_simple=B rag_snap_hyde_2call=D* adaptive_snap_hyre_frontier=D*
- `qa_unknown_scalr_272` gold=A | rag_simple=A* rag_snap_hyde_2call=D adaptive_snap_hyre_frontier=D
