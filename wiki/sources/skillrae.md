---
title: SkillRAE — Skill-Based Context Compilation
type: source
tags: [skills, retrieval, context-compilation, agents]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2605.10114
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2605.10114.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/SkillRAE
authors: Meng et al.
year: 2026
---

# SkillRAE

## TL;DR

SkillRAE compiles retrieved skill evidence into a compact 384-token advisory
packet for an unchanged executor. Its cross-reader ablation shows that the
same context compiler has different aggregate value for Codex/GPT-5.2 and
Gemini/Gemini 3 Flash. That is useful evidence for a reader-by-presentation
interaction, but not a per-artifact ranking or context-to-weight result.

## Method

Offline, SkillRAE constructs a graph over skill communities, full `SKILL.md`
files, and deterministic procedural/file/constraint subunits. Online, it
combines top-down community evidence, bottom-up subunit evidence projected to
source skills, and skill name/description scores. It selects five skills,
rescues affiliated cues, and serializes the evidence and output contract into
a 384-token packet. No model weights are changed.

## Evidence

- SkillsBench contains 87 tasks, 207 skills, 14 communities, and 4,834
  subunits in this setup. AgentSkillOS contains 30 tasks, 200 skills, 14
  communities, and 718 subunits.
- SkillsBench reward is 29.26% for SkillRAE versus 26.20% with curated skills
  and 22.04% for SkillRouter: +3.06 absolute over curated skills, reported as
  11.7% relative.
- AgentSkillOS is 84.59% versus 83.50% native and 82.30% SkillRouter.
- On SkillsBench, context compilation changes vanilla retrieval from 19.44 to
  21.71, LLM retrieval from 16.34 to 23.36, and SkillRAE retrieval from 22.59
  to 29.26.
- Removing compilation costs 6.67 points for Codex CLI with GPT-5.2 (29.26 to
  22.59) but only 1.05 for Gemini CLI with Gemini 3 Flash (28.85 to 27.80).

## Novelty boundary

SkillRAE is a mandatory context-side baseline and shows that context
representation can confound an apparent reader effect. It does not estimate
individual skill utilities, compare per-skill rankings across readers, hold a
common paired task set in its cross-reader ablation, modify weights, or
withdraw context after training.

Any utility-transport study must freeze or cross the context compiler. Changing
the skill artifact, retrieval set, renderer, and reader together would make
rank reversals uninterpretable.

## Limits

The method assumes procedures and constraints are explicit in repository text;
hidden dependencies and runtime state weaken it. Only two benchmarks and two
hosted executor configurations are studied. The paper reports no repeated
runs, confidence intervals, significance tests, or token-budget sweep.

## Code custody

- Paper-linked public repository: https://github.com/honyeung1/SkillRAE.
  The arXiv source does not link it and authorship was not independently
  verified.
- EIT checkout pinned at
  `c38e962bb24a3a009b21ec79546067bf23c9553c` on 2026-07-17.
- The release omits baseline implementations, run logs, Git history, and some
  derived graph/embedding artifacts, so reported tables are not reconstructible
  from the release alone.
- PDF SHA-256:
  `93c908c9d06f28adade5e4a4a0a90e0a1027614a11d77706ff6699a68694a3fe`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillsinjector]] ·
[[ctx2skill]] · [[constant-context-skill-learning]]
