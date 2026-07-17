---
title: Literature Coverage Audit — 2026-07-17
type: review
tags: [literature, coverage, novelty]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Literature coverage audit — 2026-07-17

## Coverage obtained

| Cluster | Primary sources read in this pass | What changed |
|---|---|---|
| Reader-conditioned utility | Predicting Retrieval Utility, CUE-R, Beyond Relevance | “Retrieval helpfulness” alone is not a novelty claim. The remaining wedge is paired causal, reader-conditioned **set** utility with harm/break-even and cost. |
| Effort control | BCAS | Generic search-depth/cost ablation is occupied. The experiment must predict marginal value and stop, conditioned on reader/evidence state. |
| Controllable reasoning | Arora-Zanette, L1/LCPO, Inkling; secondary scan of ALP, BudgetThinker, SelfBudgeter, Thinkless | Fixed-penalty and single-model prompt-controlled reasoning effort are occupied. “Vary lambda across rollouts” is a mechanism lead, not a novelty claim. |
| Efficient agent actions | OTC-PO, AutoSearch, Budget-Aware Tool-Use/BATS; Search-R1 and Agent Lightning infrastructure scan | Tool-call penalties, capability-aware minimal search depth, marginal intermediate-answer reward, and joint token/tool cost curves are occupied. The remaining candidate must use cross-reader counterfactual set utility and beat prompt-only budget awareness. |
| Conflict/sufficiency | ConflictRAG, SURE-RAG, ArbGraph | Generic conflict detection/arbitration is occupied. A legal contribution needs authority, jurisdiction, date, and precedent structure, plus abstention/sufficiency. |
| Skill internalization | SKILL0, SDAR, Skill1, OPSD, SDFT, SDPO | Broad skill internalization, same-policy co-evolution, and privileged solution/demonstration/feedback-to-weights self-distillation are occupied. Bare OPD is unsafe as the primary method; target-specific teacher quality, task RL, and gap-gated distillation are required gates. |
| Reviewer-named legal work | GuRE, KoBLEX and the existing legal-RAG pages | The old SCOPE method cannot be revived without directly engaging trained legal rewriting and provision-generation near-twins. |
| Migrated legacy legal RAG sources | Zheng et al., LRAGE, L-MARS | The old local literature folder is now represented in the persistent vault and wiki. Component grids and generic agentic search/sufficiency loops are occupied; benchmark retrieval-dependence remains a crucial evaluation axis. |

## Strongest defensible research opening

The project’s best under-owned object is not “utility,” “adaptive search,” or
“conflict resolution” in isolation. It is the **reader-conditioned marginal
utility of an evidence set**:

> Given a reader, question, current evidence set, and remaining budget, estimate
> whether another retrieval action will improve task success, leave it
> unchanged, or cause harm—and abstain/arbitrate when the set is insufficient
> or conflicting.

That object unifies the three dials already measured locally:

1. evidence exposure/quality;
2. reader ability to convert evidence into a correct answer;
3. cost and marginal search effort.

It also creates a precise relationship to the crowded areas: utility
prediction supplies prediction baselines, CUE-R supplies interventions,
SURE-RAG supplies set-level sufficiency, and budgeted search supplies fixed
policies. AutoSearch now supplies a particularly close learned baseline for
capability-aware minimal depth and marginal answer improvement. Our work must
beat or explain those baselines, not rename them.

The self-distillation update adds a separate boundary. OPSD, SDFT, and SDPO
show that privileged solutions, demonstrations, and rich feedback can be
distilled into no-context weights, but none estimates the reader-specific
causal value of forcing an external action or the target regret of importing a
teacher's action policy. Their cross-size failures make teacher quality a
measured gate, not an assumption. See
[[self-distillation-cluster-update-2026-07-17]].

One possible implementation is [[effort-conditioned-resource-allocation]]:
condition a single policy on reader, evidence-set state, hard budget, and a
resource-price vector, then let it choose among internal reasoning, retrieval,
verification, and stopping. The unique object would be the evidence action's
causal effect on a separate downstream reader. Controllable reasoning or
budget-aware tool use alone is not enough.

## Remaining high-priority gaps

1. Read and archive the primary papers/repos for RAGAS/ARES/RAGChecker and
   recent selective-RAG/abstention work; the present vault has partial coverage
   through citations and older source pages, but not a complete primary-source
   bundle.
2. Search legal authority-aware retrieval/conflict work specifically (court
   hierarchy, jurisdiction, recency, negative treatment). Generic RAG conflict
   papers are not enough to establish the legal wedge.
3. Add at least one non-legal reader×evidence replication before claiming the
   three-dial relation generalizes.
4. Keep a venue-date cutoff and rerun the literature search before submission;
   several decisive neighbors here appeared in April–May 2026.
5. Reproduce the closest effort-control baselines from the pinned AutoSearch,
   L1, and Efficient Reasoning repositories; source custody is now complete,
   but implementation comparability is not.
6. Finish full reads and code inspection for [[lrage]] and [[l-mars]]; archive
   migration only established their direct bearing and primary-source custody.

## Reproducibility

PDF hashes and repository commits are tracked in [[literature/index]]. The full
primary sources live in the persistent EIT vault, while these Markdown pages
remain local, searchable, and Obsidian-linked.
