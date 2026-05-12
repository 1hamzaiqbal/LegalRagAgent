# Report Adversarial Pass - 2026-04-30

Purpose: capture the second subagent review pass over
`docs/final_class_report_2026-04-30.{tex,pdf}` and record which gaps were
patched, which runs were launched, and which runs should wait.

## Review Findings

Four reviewers were asked to act as adversarial readers:

- narrative/claim reviewer;
- quantitative substantiation reviewer;
- harness/run-feasibility reviewer;
- novelty/literature framing reviewer.

Consensus findings:

1. The main claim should be stated earlier and more directly:
   RAG is a bottleneck-targeted intervention, and top-k sensitivity is a cheap
   diagnostic for choosing interventions.
2. `snap_hyde_2call` should remain a probe/vehicle, not the novelty claim.
3. HousingQA should stay directional until state-filtered retrieval and
   metadata audits land.
4. CaseHOLD should remain answer-flat only until repaired Chroma runs produce
   meaningful gold-hit data.
5. SCALR and CaseHOLD should not be collapsed into one holding-selection
   bucket: SCALR is candidate-depth sensitive and saturates at top-5; CaseHOLD
   is flat under current logs and instrumentation-limited.
6. Signoff needed a 2026-04-30 overlay because the final report had moved past
   the older 2026-04-27 quick-reference rows.

## Targeted Run Launched

### MuSiQue `golden_passage` N=200

Reason: test whether the BarExam golden-passage paradox is a general property
of context injection or a task/model-prior interaction.

Result:

| Comparator | Baseline | Treatment | Delta | b/c | McNemar p |
|---|---:|---:|---:|---:|---:|
| `rag_simple` -> `golden_passage` | 27.5% | 56.5% | +29.0pp | 64/6 | 2.44e-13 |
| `snap_hyde_2call` -> `golden_passage` | 37.0% | 56.5% | +19.5pp | 47/8 | 8.07e-08 |

Source doc: `docs/musique_golden_passage_2026-04-30.md`.

Interpretation: gold context is strongly helpful on MuSiQue, where missing
facts and multi-hop evidence dominate. This sharpens the BarExam story:
`golden_passage` is not bad in general; it is a noisy single-passage control
when legal priors and distractor doctrines are already strong.

## Queued Runs Followup

2026-05-01 update: both jobs completed and are no longer "pending."

| Job | Name | Status at review | Script | Why it matters |
|---:|---|---|---|---|
| 58282 | `housing-state-filter` | completed, invalid-empty-retrieval | `scripts/hpc/slurm_housing_state_filter.sh` | Exposed a metadata-casing bug; fixed and resubmitted as SLURM `58799`. |
| 58283 | `embed-eval-casehold` | completed, usable | `scripts/hpc/slurm_embed_eval_casehold.sh` | Rebuilt repaired CaseHOLD holdings and landed `rag_simple` 69.5% vs `two_call` 72.0%, p=0.4421. |

## Patches Made

- `docs/final_class_report_2026-04-30.tex`: sharpened abstract, title, and
  claim language; separated depth-flat BarExam from full-corpus method result;
  added MuSiQue `golden_passage` mechanism contrast; softened causal wording.
- `docs/signoff_log.md`: prepended a 2026-04-30 overlay with current
  bottleneck-taxonomy citation gates and paired statistics.
- `docs/README.md`: added the new MuSiQue golden-passage control to the
  evidence-ledger map.

## Next Gates

1. Pull cluster logs for job 58282 and update HousingQA with state-filter
   results before claiming metadata-aware retrieval.
2. Pull cluster logs for job 58283 and rebuild the CaseHOLD row only after
   repaired `gold_retrieved` is meaningful.
3. If more local no-API work is needed, add a Housing yes/no rescued/harmed
   audit: the current top-10 gain may be class-bias correction rather than
   retrieval-depth alone.
4. Defer online SpecRAG-lite until Housing state-filter and router v2 evidence
   say draft verification is the active bottleneck.
