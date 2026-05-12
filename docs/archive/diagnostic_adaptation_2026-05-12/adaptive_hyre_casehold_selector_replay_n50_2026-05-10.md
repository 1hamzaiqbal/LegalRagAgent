# Adaptive HyRE CaseHOLD Selector Replay N=50 - 2026-05-10

## Question

Can CaseHOLD answer-option conversion be improved without fresh retrieval by
replaying the existing option-reranker evidence through a stricter final
selector?

This probe was launched after `adaptive_snap_hyre_option_table` failed before
the final LLM call on candidate-conditioned embedding. The replay path avoids
that failure mode entirely: it consumes the already-landed option-reranker detail
log and spends only one new final selector call per row.

## Method

Script: `scripts/replay_casehold_selector.py`

Variant: `minimal_rule`

Input detail log:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_reranker_or-gemma4-26b_20260510_1201_casehold_casehold-option-reranker-cached-or-gemma4-26b-casehold-n50-k5-adaptive_snap_hyre_option_reranker_detail.jsonl`

Output detail log:
`/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-adaptive-hyre/logs/eval_adaptive_snap_hyre_option_replay_minimal_rule_or-gemma4-26b_20260510_casehold_n50_detail.jsonl`

The replay selector:

- uses the original candidate evidence prompt from the source detail row;
- includes the cached snap answer and HyRE passage;
- asks the model to prefer the holding stated at the right level of generality;
- does no Chroma retrieval, no embedding, and no cross-encoder reranking.

Implementation note: the script uses raw HTTP requests instead of LangChain or
the OpenAI client because those imports hung in the cluster Gemma environment.

## Health

- rows: 50
- average LLM calls: 1.00
- errors: 0
- `analyze_detail_flags.py`: 31/50 correct, no top-level HyDE/report/knowledge
  artifacts, expected schema fields present.

## Result

| Method | N | Accuracy | Calls | Health |
|---|---:|---:|---:|---|
| `adaptive_snap_hyre_option_score` | 50 | 22.0% | 0.00 | PASS |
| `adaptive_snap_hyre_frontier` | 50 | 70.0% | 1.00 | PASS |
| `adaptive_snap_hyre_candidate_verifier` | 50 | 74.0% | 1.00 | PASS |
| `adaptive_snap_hyre_option_reranker` | 50 | 74.0% | 1.00 | PASS |
| `adaptive_snap_hyre_option_replay_minimal_rule` | 50 | 62.0% | 1.00 | PASS |

Paired comparisons from lightweight exact McNemar counts:

| Baseline -> Replay | Delta | b/c | p |
|---|---:|---:|---:|
| option reranker -> replay | -12.0pp | 1/7 | 0.07031 |
| candidate verifier -> replay | -12.0pp | 1/7 | 0.07031 |
| frontier -> replay | -8.0pp | 2/6 | 0.2891 |
| option score -> replay | +40.0pp | 23/3 | 8.798e-05 |

## Interpretation

The replay selector is a clean negative result. It confirms that CaseHOLD is not
fixed by raw score maximization, but it also shows that simply re-prompting the
same evidence with a "right level of generality" instruction hurts relative to
the original candidate verifier and option reranker.

The useful mechanism signal is narrower:

- score-only selection is badly miscalibrated;
- LLM answer conversion is necessary;
- prompt-only replay of existing evidence is not enough;
- the remaining promising path is a calibrated selector or a verifier trained or
  tuned on features from the successful and failed rows, not another generic
  final prompt.

## Decision

- Reject `minimal_rule` replay as a CaseHOLD deployment method.
- Keep `scripts/replay_casehold_selector.py` for cheap prompt-level selector
  probes that avoid fresh embedding.
- Next CaseHOLD work should analyze row-level disagreements between
  option-reranker, candidate-verifier, replay, and score-only outputs before
  spending more API calls.
