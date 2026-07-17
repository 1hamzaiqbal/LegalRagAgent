---
title: SkillsBench — Benchmarking Agent Skills Across Models and Harnesses
type: source
tags: [skills, benchmark, contextual-utility, cross-model, harness]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2602.12670
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2602.12670.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/skillsbench
authors: SkillsBench contributors
year: 2026
---

# SkillsBench

## TL;DR

SkillsBench already estimates paired no-skill versus curated-skill contextual
lift for the same task bundle across 18 model–harness configurations. It
directly establishes that skill benefit depends on both reader and harness and
that some curated bundles harm performance.

It does not provide multiple candidate artifacts for the same task, a source
model that selects among them, source-selection regret, training, withdrawal,
or weight-space utility. The benchmark is therefore a mandatory frozen-context
control, not a collision with same-task artifact-ranking transport.

## Design

ArXiv v4 covers 87 tasks across eight domains and 18 model–harness
configurations, with three trials per arm and 9,396 main result files. For each
task-specific curated bundle, it estimates the paired contextual effect

`Delta_ctx(M, H, t, s_t) = P(pass | M, H, t, s_t) - P(pass | M, H, t, empty)`.

The harness index matters: a model is not a complete reader specification.
The appropriate notation for this project is `U_(M,H,placement)(s)`.

## Evidence

- Overall pass rate is 33.9% without skills and 50.5% with curated skills,
  a +16.6-point lift. All 18 configuration-level aggregate lifts are positive
  and range from +4.1 to +25.7, but 13 of 87 tasks have negative aggregate
  deltas.
- GPT-5.5 gains +15.8 with OpenHands (51.5 to 67.3) and +19.7 with Codex
  (46.8 to 66.5).
- Gemini 3.1 Pro gains +24.8 with Gemini CLI (36.0 to 60.8) and +19.0 with
  OpenHands (33.8 to 52.8).
- Opus 4.7 gains +18.2 with Claude Code (43.0 to 61.2) and +11.1 with
  OpenHands (42.1 to 53.1).
- Invocation varies from 99.2% for Codex/GPT-5.5 to 68.2% for Claude
  Code/Opus 4.7 and 46.4% for OpenHands/Gemini Flash Lite, so discovery and
  harness behavior mediate apparent utility.
- In the three-configuration diagnostic, self-generated skills score
  34.9/35.5/24.5 versus no-skill 43.0/46.8/36.0 and curated
  61.2/66.5/60.8. This is not cross-reader transport: creator and solver use
  the same configuration and generated-skill invocation is often absent.

## Novelty boundary

SkillsBench closes any claim that cross-reader frozen-context lift is
unmeasured. It has only one human-curated bundle per task, however. Comparing
effects across task rows changes both task and artifact, so it cannot identify
an ordering over same-task candidates or the regret of using the artifact a
source reader would select.

For [[skill-lifecycle-research-snapshot-2026-07-17]], Stage 0 is therefore a
replication/control. The candidate contribution begins with several fixed
same-task artifacts, explicit source selection, crossed target evaluation,
and then post-withdrawal target-weight outcomes.

## Limits

The +16.6-point result is optimistic efficacy evidence, not ecosystem-average
utility. Only 87 of roughly 400 submissions were accepted; tasks had to become
substantially easier with procedural skills; accepted skills came from the top
quality quartile. The benchmark has three temperature-zero trials per cell,
no length-matched irrelevant-context control, terminal/container-heavy tasks,
and interventions that may include scripts and resources rather than natural
language alone.

The repository's `website/public/skillsbench.pdf` is obsolete and reports
86 tasks, 11 domains, seven configurations, and +16.2 points. Use arXiv v4.
The repository also documents a prior skill-pollution incident affecting 15
no-skill trials; it is a provenance warning, not proof that those rows remain
in the final selected data.

## Code custody

- Official repository: https://github.com/benchflow-ai/skillsbench.
- EIT checkout pinned at
  `35661cdf113c52bb24e1644fbc200309a687ab15` on 2026-07-17.
- PDF SHA-256:
  `e987ebc3f0084a1ffc8acbca58259e33120b95fcef3876abfad060e169cf210b`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[ctx2skill]] ·
[[skillsinjector]] · [[skillrae]]
