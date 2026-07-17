---
title: Ctx2Skill — From Context to Skills
type: source
tags: [skills, context-learning, self-evolution, cross-model-transfer]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2604.27660
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2604.27660.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Ctx2Skill
authors: Si et al.
year: 2026
---

# Ctx2Skill

## TL;DR

Ctx2Skill turns a long context into an evolving Markdown skill set through a
Challenger–Reasoner–Judge self-play loop. It directly transfers the final
fixed skill set between GPT-4.1 and GPT-5.1 readers, so broad claims that
human-readable skills have not been moved across readers are no longer
defensible.

The surviving [[skill-lifecycle-research-snapshot-2026-07-17]] question is
narrower: whether the **ordering of several fixed artifacts** survives a
reader change and then a placement change from runtime context to
training-only exposure followed by withdrawal.

## Method

For each source context, five iterations alternate:

1. a Challenger generates five tasks with binary rubrics;
2. a Reasoner answers using its current skills;
3. a Judge scores the rubrics;
4. proposer/generator agents update separate Challenger and Reasoner skills
   from failures and easy successes; and
5. Cross-Time Replay evaluates historical skill sets and selects the one that
   best balances hard and easy probes.

The selected skill is prepended along with the original context. Reader
weights remain frozen. “No external feedback” means no ground-truth execution
feedback; the loop still depends on self-generated rubrics and an LLM judge.

## Evidence

- On CL-bench, GPT-4.1 rises from 11.1 to 16.5, GPT-5.1 from 21.1 to
  25.8 in Table 1, and GPT-5.2 from 18.2 to 21.4. The abstract instead gives
  21.2 to 25.8 for GPT-5.1; preserve that paper-internal discrepancy.
- The direct transfer table is the key collision. GPT-5.1-generated skills
  score 16.1 on GPT-4.1 versus 16.5 for GPT-4.1's own skills and 11.1 with no
  skill. GPT-4.1-generated skills score 23.1 on GPT-5.1 versus 25.8 for its
  own skills and 21.1 with no skill.
- Transfer is asymmetric: the stronger reader's artifact transfers almost
  completely to the weaker reader, while the reverse leaves more headroom.
- GPT-4.1's fixed-iteration scores fall from 15.9 at iteration 1 to 14.7 at
  iteration 5, while replay selection reaches 16.5. More textual optimization
  is not monotonically better.
- Selected median skill-set lengths are 705 words for GPT-4.1, 3,682 for
  GPT-5.1, and 1,458 for GPT-5.2. The paper reports roughly $30,000 in API
  cost.

## Novelty boundary

Ctx2Skill already owns aggregate cross-reader transfer of one final selected
natural-language skill set per context/model. It also shows that skill
construction can be reader-specific and that authoring asymmetry matters.

It does not rank several fixed candidates under both readers, estimate rank
agreement or source-selection regret, update reader weights, withdraw skills
after training, or compare execution-optimal with acquisition-optimal
artifacts. The original full context also remains available beside the skill.
Thus it occupies “do skills transfer?” but not the fixed-artifact utility
tensor `U(skill, reader, placement)`.

## Limits

The study uses proprietary GPT readers and an LLM judge, reports no independent
optimization reruns or confidence intervals, constructs skills separately per
context, and retains potentially large context artifacts at deployment. Its
repository HEAD postdates arXiv v3, so code/paper drift must be checked in any
reproduction.

## Code custody

- Official repository: https://github.com/S1s-Z/Ctx2Skill.
- Released skills: https://huggingface.co/datasets/ssz1111/Ctx2Skill-Skills.
- Released logs/data: https://huggingface.co/datasets/ssz1111/Ctx2Skill.
- EIT checkout pinned at
  `7776821017c42b0afff403647f28379b9fb54f96` on 2026-07-17.
- PDF SHA-256:
  `7085a7f5720d73b30b9c855e6608a89b8738f779ad514aa09d4f1104ecf0ffba`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[skillopt]] ·
[[skillsinjector]] · [[llm-specific-utility]]
