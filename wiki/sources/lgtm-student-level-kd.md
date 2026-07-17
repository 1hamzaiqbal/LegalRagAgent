---
title: LGTM — Tailoring Instructions to Student Learning Levels
type: source
tags: [knowledge-distillation, student-friendly, influence, teacher-training]
created: 2026-07-17
updated: 2026-07-17
status: maintained
url: https://arxiv.org/abs/2305.09651
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2305.09651.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/LGTM
authors: Ren et al.
year: 2023
---

# LGTM — Tailoring Instructions to Student Learning Levels

## TL;DR

Learning Good Teacher Matters (LGTM) starts from the established observation
that a higher-performing teacher need not produce a stronger student. It
estimates each training sample's effect on the exact student's held-out loss
and uses that “distillation influence” to train a more useful teacher.

Despite the paper title, its instructions are not natural-language procedures.
This is BERT classification and per-sample KD weighting. It closes the broad
claim that teacher standalone performance is a sufficient teaching objective,
but not artifact-level utility transport for reusable textual skills.

## Method

- Distillation influence is based on gradient similarity between one training
  sample's student KD loss and the updated student's validation loss.
- A finite-difference approximation avoids the full expensive influence
  calculation.
- The teacher update combines influence-weighted student generalization with
  an auxiliary teacher/online-distillation objective so the teacher does not
  simply collapse toward the student.

The optimized object is the teacher and its sample weighting, not an exported
skill, prompt, or agent policy.

## Evidence

- Across six GLUE tasks with BERT-base → BERT-6L, LGTM averages `83.4`, versus
  `82.5` for the strongest reported baseline and `81.6` for vanilla KD.
- For BERT-4L, LGTM averages `81.4` versus `80.7` for the strongest baseline.
- On MRPC, the finite-difference approximation reduces reported training time
  from 117 to 11 minutes while F1 changes from `90.7` to `90.4`.

## Novelty boundary

LGTM already estimates student-specific training influence and uses validation
generalization to improve teaching. Therefore neither “the best teacher may
not teach best” nor “use target-student feedback to improve teaching” is new.

LGTM does not compare a fixed natural-language artifact's source-context
utility, target-context utility, and post-withdrawal target gain; rank
artifacts across models; measure selection regret; or study agent actions and
tool cost. It is a conceptual baseline for
[[skill-lifecycle-research-snapshot-2026-07-17]], not an executable agent-skill
baseline.

## Limits and custody

One teacher/student architecture family, classification tasks, and no
generated procedural artifact or cross-student transfer study. The official
repository is https://github.com/twinkle0331/LGTM.

- Persistent EIT checkout pinned at
  `e9b615fb0cef1fddb17d693076dc54e9b804d110` on 2026-07-17.
- PDF SHA-256:
  `cc6b07e68f7029fc47eafb748ce8ba65473fd7aa699b8989668f7413a69dcde6`.

## Links

[[skill-lifecycle-research-snapshot-2026-07-17]] · [[promptkd]] ·
[[personalized-teacher-selection]] · [[distillation-traps-guards]]
