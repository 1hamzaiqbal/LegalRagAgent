# Adaptive HyRE Housing Verifier N=200 - 2026-05-10

## Question

Can a task-specific final verifier turn fixed HyRE retrieval into a real HousingQA
lift, rather than another mixed Snap-HyDE result?

The cached frontier replay showed a concrete Housing failure mode: the model was
too willing to answer Yes. On the N=200 frontier run, false-positive Yes errors
were much more common than false-negative No errors.

## Method

Mode: `adaptive_snap_hyre_housing_verifier`

For HousingQA only, the method keeps the same state-filtered diverse HyRE
retrieval object as the frontier path, but changes the final answer prompt:

- say Yes only when retrieved statutes affirmatively support the proposition;
- say No when evidence contradicts it, leaves out a required condition, creates
  an exception, or leaves authorization uncertain;
- keep the final answer format as `Answer: Yes` or `Answer: No`.

The run used the fixed HyRE replay cache, so every row reused the same generated
snap/HyRE object and spent only one final-answer LLM call.

## Cluster Evidence

Job: `67373`

Detail log:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_housing_verifier_or-gemma4-26b_20260510_1037_housing_housing-verifier-cached-or-gemma4-26b-housing-n200-k5-adaptive_snap_hyre_housing_verifier_detail.jsonl`

Summary:
`/engrfs/tmp/jacobsn/hiqbal_legalrag/logs/adaptive_hyre_housing_verifier_cached_n200.md`

Health:

- rows: 200
- cache hits: 200/200
- average LLM calls: 1.00
- empty retrieval: 0
- missing predictions: 0
- parse failures: 0
- audit: PASS

## Result

| Method | N | Accuracy | Gold Retrieved | Calls | Health |
|---|---:|---:|---:|---:|---|
| `rag_state_filter` | 200 | 60.5% | 81/200 | 1.00 | prior control |
| `snap_hyre_state` | 200 | 63.0% | 81/200 | 2.00 | PASS |
| `adaptive_snap_hyre_diverse` | 200 | 63.5% | 92/200 | 2.00 | PASS |
| `adaptive_snap_hyre_frontier` | 200 | 62.0% | 89/200 | 2.00 | PASS |
| `adaptive_snap_hyre_housing_verifier` | 200 | 74.5% | 89/200 | 1.00 | PASS |

Paired comparisons from the postprocess summary:

| Baseline -> Verifier | Delta | b/c | p |
|---|---:|---:|---:|
| `rag_state_filter` -> verifier | +14.0pp | 41/13 | 0.0001751 |
| `snap_hyre_state` -> verifier | +11.5pp | 37/14 | 0.001769 |
| `adaptive_snap_hyre_frontier` -> verifier | +12.5pp | 36/11 | 0.000346 |

## Error Mechanism

Compared with the N=200 Housing frontier:

| Method | Pred Yes | Pred No | False Yes | False No |
|---|---:|---:|---:|---:|
| `adaptive_snap_hyre_frontier` | 119 | 81 | 62 | 14 |
| `adaptive_snap_hyre_housing_verifier` | 78 | 122 | 29 | 22 |

The verifier did not improve retrieval: gold retrieval stayed 89/200. The gain
comes from answer conversion. It sharply reduced unsupported Yes answers and
accepted a smaller increase in false-negative No errors.

## Interpretation

This is strong evidence for the adaptive-method framing. The useful intervention
was not "more HyDE" in the abstract; it was a bottleneck-specific final verifier
on top of fixed HyRE retrieval. For HousingQA, the controller should spend its
budget on conservative statutory entailment rather than query diversity alone.

This result should be treated as Housing-specific until analogous bottleneck
fixes are validated for CaseHOLD and SCALR.

