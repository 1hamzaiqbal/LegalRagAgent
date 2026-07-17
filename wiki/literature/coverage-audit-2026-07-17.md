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
| Skill internalization | SKILL0, SDAR, Skill1 | Broad skill internalization and same-policy co-evolution are occupied. Bare OPD is unsafe as the primary method; task RL and gap-gated distillation are the evidence-backed next design. |
| Skill lifecycle and context-to-weight transfer | SkillOpt, SkillGen, MASA, SkillLens, SkillRevise, SkillMaster, SkillAudit, SkillsBench, Skill-Usage, AdaSkill, LifeSkill, SmartAD, Informative Alignment/RSR, PromptKD, LGTM, PerSyn, Distillation Traps and Guards, OPCD, Skill-SD, Skill0.5, SkillC, SEED, LatentSkill, eXTC, continual fact writing | The generic optimize-skill → internalize → distill/RL chain is occupied. Cross-reader contextual artifact matrices, student-conditioned teaching, and aggregate post-training skill withdrawal are also established. Skill-Usage makes frozen exposure/loading mandatory; AdaSkill makes the evaluation metric a frozen/indexed factor; SmartAD and RSR occupy capacity-aligned trajectory selection and cheap student-specific teaching-utility prediction. The remaining candidate is artifact-specific **placement transport**: whether the same fixed candidates preserve their ordering from one target's context to that target's matched post-withdrawal acquisition. It is a secondary, expensive question rather than the first experiment. |
| Compute elasticity and OPD | ElasticLM, BARD, Rational Metareasoning, CRISP, AdaCompute, PI Distillation, Rethinking OPD, RG-OPD, privileged-OPD negative result, TurnOPD, INTENT, MOC, CoRL, OPID | The broad claim is closed: elastic/budget-conditioned models, dynamic-price planning, preference-conditioned policies, and skill-conditioned OPD already exist. The remaining candidate is cross-scale **forced-action value transport and target regret from a transferred teacher oracle**. |
| Agent and strategy distillation | Agent Distillation with retrieval/code, SGPO, cost-aware skill rewriting | Full tool behavior, reusable strategy guidance, and quality/cost-aware external skills are occupied. The remaining intersection must condition deployment action choice on resource prices and remove teacher-only skills at inference. |
| Latent reasoning / intervention | J-lens/J-space, implicit-CoT KD, Coconut, LoRi | J-space is a white-box diagnostic/intervention substrate, not an API or cross-model method. Generic hidden-state/low-rank reasoning transfer is already occupied. |
| Small specialist versus large generalist | Distilling Step-by-Step, DeepSeek-R1 distillation, Thinking Machines expert judgment | Narrow students can exceed much larger or tested frontier models, but results are task-specific. Any new claim needs frozen held-out tasks, matched inference protocols/cost, and contamination controls. |
| Reviewer-named legal work | GuRE, KoBLEX and the existing legal-RAG pages | The old SCOPE method cannot be revived without directly engaging trained legal rewriting and provision-generation near-twins. |
| Migrated legacy legal RAG sources | Zheng et al., LRAGE, L-MARS | The old local literature folder is now represented in the persistent vault and wiki. Component grids and generic agentic search/sufficiency loops are occupied; benchmark retrieval-dependence remains a crucial evaluation axis. |
| Student-specific action value | Nagarajan et al.; Model-Adaptive Tool Necessity; Tool-Call Boundary Drift; RG-OPD; CRAFT; Token Teachability; SmartAD; Informative Alignment/RSR; LLM-Specific Utility | Generic disobedience, model-specific need, boundary drift, harmful teacher signal, signal incompatibility, student-specific trajectory teaching value, and reader-specific passage utility are occupied. The provisional opening is repeated, signed, cost-sensitive **teacher-to-target forced-action value transport** and held-out target regret from the forced-outcome teacher oracle. |

## Foundational three-dial opening: evidence-set action value

The project's foundational under-owned object is not “utility,” “adaptive
search,” or “conflict resolution” in isolation. It is the
**reader-conditioned marginal utility of an evidence set**:

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

One possible later implementation is [[effort-conditioned-resource-allocation]]:
condition a single policy on reader, evidence-set state, hard budget, and a
resource-price vector, then let it choose among internal reasoning, retrieval,
verification, and stopping. The unique object would be the evidence action's
causal effect on a separate downstream reader. Controllable reasoning or
budget-aware tool use alone is not enough. The cleaner first experiment strips
this down to one forced external action.

## Recommended primary opening: counterfactual action-value transport

The final recommendation in
[[research-question-recommendation-2026-07-17]] is:

> When does a strong teacher's optimal costly external action become
> suboptimal for the student that must execute or use it, and what structure
> in that action value transfers across model scale and family?

This starts as a repeated forced-action measurement panel, not OPD. Estimate
teacher and student advantages for the same canonical deterministic result or
fixed evidence payload, then examine sign, rank, threshold, difficulty, and
held-out target regret from the forced-outcome teacher oracle. Ex-post cost
weighting gives hypothetical oracle boundaries, not behavioral price response.
Training is downstream and justified only if teacher information predicts the
target's value or decision more sample-efficiently than student-only evidence.

The earlier “conditioned frontier distillation” and “cost-conditioned skill
internalization” statements were too broad.
ElasticLM already names compute elasticity in a distilled model, and BARD
already transfers a budget-conditioned reasoning frontier with exploration and
verification changes. Agent Distillation transfers retrieval/code behavior;
SGPO transfers strategies; ClawTrace distills cost-aware skills; OPID combines
on-policy RL with skill distillation; INTENT handles changing tool prices; MOC
generalizes to unseen preferences. The plausibly under-owned object is now the
paired measurement of whether cross-scale transfer preserves same-task forced-
action values and whether the teacher-derived oracle induces target-student
regret on held-out items. Teacher-only skills are an ablation, not the novelty
claim.

This direction should be domain-general. Begin with one fixed token cap, one
canonical deterministic Python/calculator result injected identically through
the harness, and one verifiable Reasoning Gym environment. Legal retrieval
can be a later stress test; it is not required and should not constrain the
benchmark surface.

### Full-paper correction: transport value, not behavioral price response

The focused [[action-value-transport-reading-packet-2026-07-17]] sharpens this
candidate further. A behavioral price-response curve is adjacent to existing
budget-conditioned policies and requires a separate free-choice experiment
with displayed randomized prices. The under-owned empirical object is whether the
**same forced external action** has different signed value for teacher and
target student, and whether the teacher's threshold creates target-student
regret. The strongest architecture-free hypothesis is that advantage rankings
partially transfer while cardinal values and zero-crossings require a small
target-student calibration set.

This is a candidate pattern, not a committed contribution. The first study
should ask broad causal questions and retain enough outcome, execution,
behavior, and cost information for alternative structures to emerge. A metric,
law, or calibration method should be claimed only if it earns that role on
untouched models/items rather than being chosen a priori.

This correction also separates four layers that current framing had mixed:
task/action value, teacher-policy transport, token-level absorbability, and
training credit. OPD should remain downstream of the causal measurement pass.

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
7. Rerun the venue-date search specifically for cross-scale forced-action
   value transport and held-out target regret from a transferred teacher
   oracle. Treat behavioral price response as a separate displayed-price
   experiment; adjacent ingredients are heavily occupied.
8. Reproduce prompt-only conditioning, direct conditioned RL, BARD, Agent
   Distillation, INTENT-style price planning, and OPID controls before
   implementing latent/J-space variants.
9. Before submission, recheck the currently unavailable artifacts for
   Tool-Call Boundary Drift, RG-OPD, and CRAFT, and resolve the anonymous
   successor of LLM-Specific Utility.
10. Before pursuing the secondary skill-lifecycle candidate, treat
    [[skillgen-verified]], [[masa]], [[skilllens]], [[skillrevise]],
    [[skillmaster]], [[skillsbench]], and [[skillaudit]] as direct contextual
    or withdrawal controls, in addition to [[smartad]],
    [[informative-alignment-rsr]], [[promptkd]],
    [[lgtm-student-level-kd]], [[personalized-teacher-selection]],
    [[distillation-traps-guards]], [[opcd]], [[seed-self-evolving-opd]],
    [[latent-skill]], and [[structured-prompt-optimization-extc]]. The full
    chain must beat direct target training, direct internalization, direct
    context-conditioned OPD, and external/modular skill use.

## Reproducibility

PDF hashes and repository commits are tracked in [[literature/index]]. The full
primary sources live in the persistent EIT vault, while these Markdown pages
remain local, searchable, and Obsidian-linked.
