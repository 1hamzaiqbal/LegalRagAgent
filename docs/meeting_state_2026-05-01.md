# Meeting State Brief - 2026-05-01

Purpose: meeting-ready synthesis of the current research state across methods,
datasets, blockers, and the interpretation we can defend today.

## One-Slide Thesis

The project should not be framed as "a new RAG method wins." The defensible
claim is:

> RAG is an intervention, not one capability. Across legal and multi-hop QA,
> the useful intervention changes with the active bottleneck: retrieval depth,
> candidate-set size, query formulation, evidence use, answer-option anchoring,
> or metadata-constrained retrieval.

`snap_hyde_2call` is a useful probe and current MuSiQue method vehicle. The
novelty is the bottleneck diagnostic matrix and the evidence-budgeted routing
direction, not the pseudo-document primitive itself.

## Current Findings

| Dataset slice | Best current read | Evidence | Caveat |
|---|---|---|---|
| MuSiQue x Llama 70B | Retrieval-depth and query-formulation limited. `snap_hyde_2call` improves `rag_simple` 27.5% -> 37.0%, +9.5pp, p=0.007943. Top-1 collapses to 13.0%. | `docs/signoff_log.md`; `docs/evidence_matrix_2026-04-30.md`; `docs/musique_golden_passage_2026-04-30.md` | N=200 diagnostic slice, not full corpus. |
| BarExam x Gemma 4 | Legal MC is depth-flat on the N=200 top-k diagnostic, but full-corpus `rag_snap_hyde` is a signed legal-MC lift: 26B +3.09pp, E4B +3.68pp. | `docs/signoff_log.md` Section A; `docs/compiled_results.md` | Do not call this pure retrieval-depth improvement; better read is answer anchoring/evidence-use under strong legal priors. |
| LegalBench-SCALR x Llama 70B | Candidate-depth limited then saturated: top-1 59.5%, top-5 77.0%, top-10 77.0%. | `docs/scalr_depth_disagreement_2026-04-30.md` | Top-k changes candidate pool and context depth, so call it a retrieval-policy stress test. |
| CaseHOLD x Llama 70B | Repaired rerun now gives usable gold retrieval. `rag_snap_hyde_2call` improves gold-hit 16.0% -> 47.0% but answer accuracy only 69.5% -> 72.0%, p=0.4421. | `docs/casehold_repaired_rerun_2026-05-01.md` | Better retrieval recall is not converted into a reliable answer lift at N=200. Top-k still needs repaired rerun if we want a fresh depth claim. |
| HousingQA x Gemma 4 | Directional statutory depth signal: top-1 50.5%, top-10 58.0%, p=0.0722. | `docs/housing_speculative_metrics_2026-04-30.md`; `docs/housing_metadata_depth_audit_2026-04-30.md` | State-filter attempt exposed a casing bug and was resubmitted as job `58799`; no state-filter claim yet. |
| MLEB-SCALR | Retrieval-only calibration channel. `gte-large` beats MiniLM: Recall@5 65.0% vs 57.5%. | `docs/mleb_scalr_embedding_ab_2026-04-30.md` | Not an answer-quality benchmark; keep it out of EM tables. |

## Interpretation

The signal is the pattern, not any single row:

- Missing-fact multi-hop QA behaves as expected: MuSiQue collapses when
  retrieval depth is restricted, and privileged gold context jumps to 56.5%.
- Legal MC behaves differently: BarExam is high even with top-1 retrieval and
  benefits modestly from snap-style answer anchoring at full N.
- Holding-selection splits internally: SCALR needs a small candidate set;
  CaseHOLD now retrieves more gold options under two-call but still does not
  reliably lift answers.
- Housing is the best legal-domain test for metadata-constrained statutory
  retrieval, but the explicit state-filter result is still blocked pending the
  fixed rerun.

The meeting story should therefore be "we found a way to type bottlenecks,"
not "we found one method that wins."

## Blockers And Live Work

| Item | Status | What it blocks |
|---|---|---|
| Housing state-filter | Job `58282` failed with 100% empty retrieval due state metadata casing. Fixed locally and resubmitted as SLURM `58799`. | Whether Housing's top-10 lift is metadata/jurisdiction repair or generic deeper retrieval. |
| CaseHOLD top-k after repair | `rag_simple` and `two_call` repaired rerun landed; top-1/top-5 repaired rerun has not landed. | A fresh CaseHOLD depth-policy claim. |
| Local Chroma | Local `legal_passages` is empty; local Chroma only has SCALR/MLEB-SCALR. | Local BarExam/Housing/CaseHOLD RAG runs. Use cluster or rebuild locally. |
| Full-corpus MuSiQue Llama | Still rate-limit/provider constrained. | Turning the MuSiQue N=200 diagnostic into a full-corpus result. |
| SpecRAG-lite metrics | Answer drafts/verifier scores are not logged. | True Speculative-RAG draft/verifier metrics; current metrics are aligned proxies only. |
| Online router | Offline oracle headroom exists, but leave-one-dataset-out generalization is weak. | Prospective claim that a cheap router can select methods on the fly. |

## What To Say If Challenged

- "Is this just N=200?" For MuSiQue, SCALR, Housing, and CaseHOLD yes: treat as
  diagnostic slices. BarExam full-corpus is the strongest paper-grade legal MC
  result.
- "Does top-k prove the cause?" No. It is a cheap retrieval-policy stress test.
  The full diagnostic matrix is needed to distinguish query formulation,
  evidence use, option anchoring, and metadata effects.
- "Is snap-HyDE novel?" No. It is HyDE/active-retrieval adjacent. The useful
  contribution is using fixed probes to type bottlenecks.
- "Are agents bad?" No. Current unstructured subagent/gap-routing over-abstains
  on MuSiQue. The plausible agentic direction is shared evidence state and
  evidence-budgeted routing.

## Recommended Next Order

1. Monitor and pull SLURM `58799`; if non-empty, recompute Housing paired tests
   against top-5/top-10 and update `docs/housing_state_filter_followup_2026-05-01.md`.
2. Run a repaired CaseHOLD top-k pair if we need a clean CaseHOLD depth claim.
3. Add router v2 features from disagreement buckets and metadata hits before
   building an online controller.
4. Keep SpecRAG-lite deferred until a cell shows draft/verification is the
   active bottleneck.
5. For a broader research paper, prioritize Legal RAG Bench wiring over another
   method variant because it aligns with retrieval/generator error
   decomposition.
