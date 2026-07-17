---
title: SkillAudit — Skill-Centered Assessment
type: source
tags: [skills, evaluation, utility, cost, safety, cross-model]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2606.22613
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2606.22613.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/skillaudit
authors: Yu et al.
year: 2026
---

# SkillAudit

## TL;DR

SkillAudit is the closest context-only measurement neighbor. It treats a skill
package as the evaluation object, generates capability-aligned scenarios, and
runs matched with/without-skill tasks across six model–harness configurations,
reporting utility, execution cost, and safety. A generic “evaluate arbitrary
skills across readers” contribution is closed.

It does not report per-artifact rank transport or source-selection regret and
has no training, withdrawal, adapter, or weight arm.

## Method and metrics

Each skill generates three scenarios, compiled into matched with-skill and
without-skill tasks intended to share instruction, input, environment, rubric,
and judge.

- Pass-Rate Gain: `(passed checklist items with skill - without skill) / N`.
- Efficiency/Cost Gain averages relative time and input-token savings clipped
  to `[-1,1]`. It is computed only when both conditions pass at least one judge
  item, so it is a selected conditional metric rather than unconditional cost
  effectiveness.

## Evidence

Across 226 skills and 23 occupational categories:

- Codex/GPT-5.4 has no-skill .763 and mean PRG .183; Codex/GPT-5.1 has
  .630/.248.
- Claude Code/Sonnet 4.6 has .799/.164; Claude Code/Sonnet 4 has
  .647/.207.
- OpenCode/GPT-5.4 has .718/.185; OpenCode/Sonnet 4.6 has .794/.130.
- The same Sonnet 4.6 backbone has office-administration PRG .425 in Claude
  Code versus .118 in OpenCode.
- Regression appears in every configuration: 3.8–10.9% of scenarios have
  negative PRG.
- Baseline headroom dominates under GPT-5.4: PRG is .648/.269/.034 for
  low/mid/high-baseline scenarios.
- Overall ECG is -.186, with efficiency gain -.133 and token-cost gain -.238.

## Novelty boundary

SkillAudit closes skill-centered, paired, cross-reader contextual assessment.
It reports category means and scenario distributions, not artifact-level rank
correlations, top-k overlap, rank reversals, or source-selection regret.
Valid scenario sets can differ across configurations because timeouts and judge
failures are excluded. There is one run per condition, no run-to-run
uncertainty, and tasks/rubrics are generated from each skill's claimed
competence gap. No weights are changed.

Any transport study should use the complete common candidate/task
intersection across readers, repeated paired outcomes, forced availability,
exact byte identity, a frozen renderer, and separately recorded model and
harness. SkillAudit is the context-only measurement baseline.

## Release/custody audit

- Project: https://skillaudit.github.io/.
- Official repository: https://github.com/SkillAudit/skillaudit.
- EIT checkout pinned at
  `2c46f770f823375dce6e956c505dcc4d5137cb35` on 2026-07-17; Apache-2.0.
- Release tag `v1.0.0-anon` exists. The paper used an internal unversioned
  Harbor; the public release was ported to Harbor 0.6.5.
- Public counts conflict: paper 226 skills/23 categories/six utility
  configurations; README references eight categories and a 227-skill x8-run
  sweep; `stats.json` has 217 skills and 1,806 reports; the per-skill directory
  has 216 files; aggregate CSV has 1,732 rows. Full 150–300GB trajectories are
  not released, and the observed Zenodo record exposes no files.
- PDF SHA-256:
  `72e58d8438b8881b33c704b35077b183aee3167430798bf5927fc7ba8dde293b`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillsbench]] ·
[[adaskill]] · [[skill-usage-in-the-wild]]
