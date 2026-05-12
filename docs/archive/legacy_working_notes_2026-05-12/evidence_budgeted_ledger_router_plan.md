# Evidence-Budgeted Ledger Router Plan

Branch: `codex/evidence-ledger-router`

This branch combines the two strongest angles from
`docs/agentic_legal_rag_angles_2026-04-30.md`:

1. **Evidence-budgeted active legal RAG** - a controller decides which method
   arm deserves budget for a given question.
2. **Shared evidence ledger** - worker agents pass structured, auditable
   evidence state instead of lossy natural-language reports alone.

## Research Claim

Legal agentic RAG should be evaluated as **budget allocation over evidence
operations**, not as "more agents vs fewer agents." A legal agent must decide
whether to spend budget on retrieval, which retrieval/search strategy to use,
which role agent should inspect the result, what evidence state should be
shared, and when to stop.

The novelty target is therefore:

> A bottleneck-aware legal RAG controller that routes between retrieval and
> reasoning arms under a cost budget while preserving source-backed claims in a
> shared evidence ledger.

This is not a generic multi-agent workflow. The legal-specific hook is that
legal answers require auditable support: source identity, quoted support,
jurisdiction/date metadata, contradiction status, and role provenance.

## System Sketch

```text
question
  -> task profiler
  -> cheap retrieval probe
  -> bottleneck router
      -> direct / snap-only
      -> simple RAG k=1
      -> simple RAG k=5
      -> multi-query / HyDE
      -> ledger subagent path
  -> shared evidence ledger
  -> final answerer with ledger + compact evidence
  -> verifier / optional escalation
```

The router decides **which arm runs**. The ledger decides **what state survives**
from the arm into final synthesis.

## Why The Combination Matters

Routing alone can pick the right method but still let subagents summarize away
the evidence needed for legal auditability. A ledger alone can preserve evidence
but still waste budget by invoking expensive agents on tasks where direct
reasoning or top-1 retrieval is enough. Together they test the actual legal-agent
hypothesis: efficient orchestration plus auditable shared state.

## Files Added On This Branch

- `scripts/build_router_training_set.py` - joins completed detail logs by
  question ID, extracts cheap task features, records per-arm correctness/cost,
  and labels each row with oracle accuracy/reward arms.
- `scripts/train_router_baseline.py` - evaluates fixed arms, majority-oracle
  labels, and a reward-optimized one-rule router on random and
  leave-one-dataset-out splits.
- `eval/evidence_ledger.py` - minimal structured evidence-ledger contract.
- `tests/test_evidence_ledger.py` - regression tests for validation and prompt
  rendering.
- `docs/router_probe_findings_2026-04-30.md` - interpretation of the first
  offline router result.
- `docs/router_baseline_report_2026-04-30.md` - generated baseline table for
  `rag` vs `two_call` on four N=200 slices.

These are scaffolds, not an online eval mode. The branch deliberately starts
offline so we can answer whether routing is learnable before adding another
expensive harness mode.

## First Offline Probe Result

The first common-arm table tests `rag` vs `two_call` across MuSiQue, BarExam,
CaseHOLD, and LegalBench-SCALR. After adding evidence-state features, the
one-rule router reaches 72.0% on a random split at 1.30 calls/q, compared with
71.5% for fixed `two_call` at 2.00 calls/q and 69.0% for fixed `rag` at
1.00 calls/q. Leave-one-dataset-out behavior remains brittle:

- it partially recovers rare BarExam `two_call` flips;
- it fails to identify MuSiQue as a `two_call` regime under held-out training;
- it mostly collapses to cheap `rag` on CaseHOLD and LegalBench-SCALR;
- the oracle still has 4-10pp absolute headroom depending on the held-out
  dataset.

This is useful evidence for the combined story. Cheap prompt-shape routing is
not enough, and generic evidence-overlap features are not enough either. The
controller likely needs structured support, contradiction, and missing-link
state before it can decide when escalation is worth the cost.

## Next Implementation Steps

1. Add richer router features from cheap online probes:
   - direct-answer stability across two samples,
   - k=1 vs k=5 answer change,
   - answer-option/evidence overlap,
   - source disagreement or contradiction flags.
2. Add a stronger router baseline if we keep investing in routing:
   - gradient boosting or calibrated cost-sensitive models,
   - explicit per-regime training rather than one global model.
3. Add a ledger subagent eval mode:
   - retrieve per gap,
   - ask role agents to emit JSON ledger entries,
   - final answerer receives `ledger_to_prompt(...)` plus compact source text.
4. Compare:
   - `subagent_rag` report-only,
   - `subagent_rag_evidence`,
   - `ledger_subagent_rag`,
   - learned router with and without ledger arm.
5. Only after the offline router and ledger arm show signal, wire the online
   `bottleneck_router` state machine.

## Example Router Training Command

```bash
python3 scripts/build_router_training_set.py \
  --arm rag=logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl \
  --arm two_call=logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl \
  --arm mhd=logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl \
  --arm iter=logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl \
  --output /tmp/musique_router_training.csv
```

## Decision Gate

Proceed to online routing only if a cheap learned router beats the best fixed
arm on held-out questions or held-out datasets. Otherwise, keep the paper as a
diagnostic/negative result and invest in the ledger path for workshop impact.
