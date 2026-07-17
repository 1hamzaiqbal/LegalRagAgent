---
title: SkillAdaptor — Failure-Attributed External Skill Adaptation
type: source
tags: [skills, adaptation, failure-attribution, contextual-utility, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.01311
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.01311.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SkillAdaptor
authors: Yu et al.
year: 2026
---

# SkillAdaptor

## TL;DR

SkillAdaptor is a training-free loop for repairing external skills from failed
agent trajectories. It localizes the first actionable fault, attributes
responsibility to retrieved skills, revises or generates a targeted skill,
and accepts updates only after held-out validation. Across Kimi-K2.5, GLM-5,
and GPT-5.2 experiments, it reports improvements on PinchBench, Claw-Eval, and
WebShop.

It does not cross a fixed candidate set over readers, update model weights, or
remove skills after training. It narrows generic novelty around
failure-grounded skill evolution, but not reader-conditioned ordering or
context-versus-weight placement.

## Method

The target backbone remains frozen. For a failed trajectory, the Localizer
identifies the earliest actionable bad step; the Linker scores responsibility
among candidate injected skills; the Reviser edits an implicated skill or the
Generator creates a missing one. A Validator reruns held-out tasks and adopts
the candidate only when measured outcomes improve under the method's
acceptance checks.

Skills remain external `SKILL.md` artifacts selected and injected at runtime.
Semantic retrieval and reranking can be model-specific, so separate runs do
not hold the deployed artifact/context fixed across readers.

## Evidence

The largest improvements highlighted in the paper are:

- +1.5 points in PinchBench average score percentage;
- +1.8 points in Claw-Eval average score; and
- +1.7 points in WebShop success rate.

The paper evaluates Kimi-K2.5, GLM-5, and GPT-5.2 and reports improvement over
its no-skill and adaptation baselines on all three benchmark families. These
are end-to-end per-model adaptation results, not cells in a common
artifact × reader matrix.

## Exact claim boundary

SkillAdaptor occupies training-free, step-level failure attribution,
responsibility-linked skill revision, and validation-gated external skill
adoption. It does not estimate how one fixed skill's utility changes across
readers, whether a source-selected candidate is target-optimal, or whether
contextual utility predicts acquisition into weights. There is no
post-withdrawal or matched direct-training arm.

## Artifact gaps and limitations

The official repository now contains the Python framework, benchmark adapters,
harness integrations, and an MIT license. At the pinned commit it does not
contain the paper's raw trajectories, evolved skill artifacts, benchmark
result tables, or run logs. The code can support new executions, but the
reported numbers cannot be reconstructed from deposited result artifacts
alone.

The per-model adaptation loops may produce different artifacts and retrieval
contexts. Without freezing and crossing those bytes, differences across
models cannot identify reader-conditioned utility or artifact ordering.

## Code custody

- Audited arXiv v1 PDF:
  `/engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.01311.pdf`.
- PDF SHA-256:
  `b2a4cd328ecff90db75854640de6b3ac39380a427f85fb367f24a18fd7a38eed`.
- Official repository: https://github.com/zjunlp/SkillAdaptor.
- EIT checkout pinned at
  `b26d1ab5a798f07e53048b5ff509e8535e9fa228` on 2026-07-17.
- Repository license: MIT.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] · [[lifeskill]]
