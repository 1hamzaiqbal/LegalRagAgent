---
title: Recent Literature Matrix — July 2026
type: review
tags: [literature, novelty, agents, skill-distillation]
created: 2026-07-17
updated: 2026-07-17
status: maintained
---

# Recent literature matrix

## Closest work

| Work | What it already owns | Bearing on this project |
|---|---|---|
| [INTENT (2602.11541)](https://arxiv.org/abs/2602.11541) | Inference-time planning with priced and stochastic tools; cost-augmented StableToolBench; explicit budget feasibility | Direct dynamic-price baseline; no cross-scale distillation claim found |
| [BAVT (2603.12634)](https://arxiv.org/abs/2603.12634) | Budget-conditioned value-tree search and step-level pruning without training | Strong inference-time budget baseline |
| [One Model for All / MOC (2604.04497)](https://arxiv.org/abs/2604.04497) | Preference-conditioned multi-objective policy and unseen-preference generalization | Closes generic “one controllable policy over tradeoffs” novelty |
| [CoRL (2511.02755)](https://arxiv.org/abs/2511.02755) | Budget-conditioned multi-LLM controller trained with RL | Closes generic budget-conditioned routing novelty |
| [ClawTrace (2604.23853)](https://arxiv.org/abs/2604.23853) | Per-step agent cost tracing and cost-aware external-skill pruning/repair | Closest “cost + skill distillation”; external text skills, not weight-space price-response transfer |
| [OPID (2606.26790)](https://arxiv.org/abs/2606.26790) | On-policy hierarchical hindsight-skill extraction and skill-conditioned self-distillation | Closes broad “skill + OPD” idea; useful baseline/ablation |
| [SkillMOO (2604.09297)](https://arxiv.org/abs/2604.09297) | Multi-objective skill-bundle optimization over success, cost, and runtime | Closes cost-aware skill optimization novelty |
| [SkillOpt (2605.23904)](https://arxiv.org/abs/2605.23904) | Controlled text-space skill optimization with held-out validation gates | External-skill baseline; good experimental discipline to copy |
| [SkillGrad (2605.27760)](https://arxiv.org/abs/2605.27760) | Trajectory-derived textual gradients and momentum for skill packages | External-skill/harness optimization neighbor |
| [SkillAdaptor (2606.01311)](https://arxiv.org/abs/2606.01311) | Training-free step-level skill adaptation with failure attribution | Another reason not to claim generic adaptive skills |
| [Meta-Harness (2603.28052)](https://arxiv.org/abs/2603.28052) | Outer-loop search over harness code and traces | Harness optimization is research territory, not neutral novelty |
| [RHO (2606.05922)](https://arxiv.org/abs/2606.05922) | Self-supervised retrospective harness optimization from trajectories | Close to automatic harness improvement without labels |
| [AHE (2604.25850)](https://arxiv.org/abs/2604.25850) | Observability-driven agent-harness evolution | Further closes “evolve a better harness” novelty |
| [SkillOpt-Lite (2607.03451)](https://arxiv.org/abs/2607.03451) | Very recent lightweight skill optimization | Mandatory pre-submission near-neighbor check |

## Infrastructure and benchmark sources

| Work | Useful asset | Limitation for our core claim |
|---|---|---|
| [Reasoning Gym (2505.24760)](https://arxiv.org/abs/2505.24760) | 100+ procedural verifiable environments with adjustable complexity | Does not provide the priced-tool intervention itself |
| [Agent Lightning (2508.03680)](https://arxiv.org/abs/2508.03680) | Framework-neutral structured spans and agent/trainer decoupling | More machinery than Phase 0 requires |
| [VerlTool (2509.01055)](https://arxiv.org/abs/2509.01055) | Multi-turn tool RL on VERL with stateful tool environments | Heavy backend; defer until the phenomenon passes gates |
| [StableToolBench (2403.07714)](https://arxiv.org/abs/2403.07714) | Stable tool evaluation substrate used by INTENT's cost augmentation | External API/model dependencies and heavier judging |
| [MirrorAPI (2503.20527)](https://arxiv.org/abs/2503.20527) | Stable mirrored tool APIs | Large dependency; unnecessary for the one-tool pilot |
| [SkillsBench (2602.12670)](https://arxiv.org/abs/2602.12670) | 86 tasks, deterministic verifiers, curated/no/self skill comparisons | Skill/harness effects are heterogeneous; use later as external stress test |
| [Agent Distillation (2505.17612)](https://arxiv.org/abs/2505.17612) | Open code/retrieval trajectory distillation and public teacher traces | Fixed tool regime; downloaded traces are successful-only training data, not evaluation data |

## Related transfer and efficiency anchors already in the vault

- [[bard-budget-aware-reasoning-distillation]] is the decisive token-budget
  distillation neighbor.
- [[elastic-language-models]] closes the broad compute-elastic-model label.
- [[rational-metareasoning]] supplies the value-of-computation framing.
- [[agent-distillation-tools]] supplies the cross-scale tool-trajectory baseline.
- [[skill0]], [[sdar]], [[skill1]], [[reward-gated-opd]], and
  [[rethinking-privileged-opd]] define the skill/OPD controls and failure risks.

## Search conclusion

The literature is not sparse; it is rapidly converging on budget-conditioned
agents, external skill optimization, and trainable harnesses. The proposed
work should therefore be framed as a controlled study of **conditional policy
preservation under counterfactual prices**. A venue-date search must be rerun
immediately before submission.

All PDFs and repos listed here are recorded in [[09-source-and-asset-ledger]].
