---
title: SKILL0 — In-Context Agentic RL for Skill Internalization
type: source
tags: [agentic-rl, skills, internalization, curriculum, distillation-adjacent]
created: 2026-07-02
date: 2026-07-02
arxiv: "2604.02268"
code: https://github.com/ZJU-REAL/SkillZero
archived: EIT references/papers/skill0_2604.02268.pdf + references/repos/skillzero_repo.tgz
---

# SKILL0 (Lu et al., ZJU + Meituan LongCat + Tsinghua, arXiv 2604.02268v2, May 2026)

**One sentence**: RL framework that *internalizes* agent skills — the
Claude-Code-style markdown skill packages agents normally retrieve at
inference — into model parameters, by providing skills in-context during
training rollouts and progressively withdrawing them until the agent runs
fully zero-shot ("skills at training, zero at inference").

## Problem they attack
Inference-time skill augmentation has three costs: retrieval noise injects
irrelevant guidance, injected skill text imposes per-step token overhead, and
the model *follows* skills without ever *acquiring* them — competence lives
in the context, not the model.

## Method (the parts that matter for us)
1. **SkillBank**: hierarchical markdown skill library
   (`skills/{task}/{category}.md`), general + task-specific, following the
   agent-skills ecosystem (they cite Claude Code / OpenClaw skills directly).
2. **In-Context RL (ICRL)**: GRPO-style group-relative policy optimization
   (clipped importance ratio + KL to reference) where rollouts see a selected
   skill subset in context; inference sees none.
3. **Helpfulness-driven Dynamic Curriculum** — the key device. Per skill file
   S_k: Δ_k = Acc(π, T_k, with S_k) − Acc(π, T_k, without), measured
   on-policy every d steps on a matched validation sub-task T_k (offline
   relevance-driven grouping). Keep only Δ_k > 0, rank, take top-M under a
   **linearly decaying skill budget** M^(s) → 0. Skills exit exactly when
   the current policy stops benefiting — "adaptive internalization" instead
   of a rigid schedule.
4. **Context rendering**: interaction history + skills rendered to a compact
   RGB image consumed by the VLM's vision encoder, with a *self-generated*
   compression ratio c_t and a composite reward r + λ·ln(c_t) on success —
   this is why per-step context is <0.5k tokens.

## Results
Qwen2.5-VL 3B/7B, ≤180 steps on 4×H800. Over the AgentOCR RL baseline:
ALFWorld +9.7 (→87.9 at 3B), Search-QA +6.6 (→40.8; Search-R1 setup, E5
retriever, NQ/HotpotQA in-domain train + OOD evals), WebShop +10.1
(→78.6/66.4 score/acc). Beats memory-augmented baselines (ExpeL, Mem0,
MemRL) and search-RL baselines (Search-R1, ZeroSearch) in their tables.
**Helpfulness dynamics**: Δ_k rises early (policy learns to *use* skills),
then falls toward zero as internalization completes — skills as *transient
scaffolding*.

## Why we care ([[direction-2026-07]], 2026-07-02 mentor meeting)
- **The convergent measurement**: their skill-level Δ_k is the same
  construct as our evidence-level gold-present/gold-absent decomposition in
  [[judge-answer-conversion]] — "is this context *helpful to this policy*,
  measured on-policy" vs "is it merely relevant." Their curriculum retires
  context when Δ→0; our conversion dial says evidence pays only under
  parametric deficit. Same worldview, retrieval instead of skills.
- **The bridge idea** (primary meeting interest, [[skill-distillation-bridge]]):
  SKILL0 internalizes skills into the *same* model. The proposed twist is
  **cross-scale**: use a big model (or skill/context-augmented policy) as the
  teacher and internalize into a *smaller* model — distillation of agentic
  retrieval skills, not self-internalization. Their own follow-up **SDAR
  (Self-Distilled Agentic RL, ZJU-REAL, 2026-05)** and **SKILL1**
  (one unified policy, USTC) are adjacent. Both are now read: [[sdar]] shows
  standalone/naive on-policy self-distillation can collapse and motivates
  task RL plus gap gating; [[skill1]] occupies unified skill selection,
  utilization, and distillation. Broad novelty is therefore closed.
- Search-QA is their weakest domain (40.8 vs 87.9/78.6) — retrieval-heavy
  skills internalize worst. That gap is exactly where our three-dial
  machinery (what makes retrieval context helpful) has something to add.

## Caveats
VLM-specific context-rendering may not transfer to text-only pipelines;
skill libraries initialized from SkillRL (curated); 3B/7B only; benchmarks
are agent playgrounds (ALFWorld/WebShop), not professional domains like law.

## Links
[[skill-distillation-bridge]] · [[sdar]] · [[skill1]] · [[thinking-machines-expert-judgment]] ·
[[expert-judgment-replication]] · [[judge-answer-conversion]] ·
[[judge-mixed-legal]] · [[direction-2026-07]]
