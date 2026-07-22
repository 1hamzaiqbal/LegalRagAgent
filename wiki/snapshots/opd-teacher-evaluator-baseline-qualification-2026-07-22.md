---
title: OPD teacher, evaluator, and baseline qualification
date: 2026-07-22
status: audit complete; length calibration specified; no new training authorized
tags: [opd, math, teacher, evaluation, truncation, baselines, contamination]
---

# OPD teacher, evaluator, and baseline qualification - 2026-07-22

## Bottom line

The O teacher's sealed held-out gain is real under the post-hoc
symbolic-eligible score-ledger estimand, but run `108609` did **not** establish
that the teacher was trained close to the capability of this recipe. It was a
low-signal pilot: only 16 of 100 nominal optimizer steps had nonzero gradient,
178 of 400 completions reached the 1,024-token cap, and only 3 capped
completions were correct.

The dominant prediction-parse failure is also not a parser implementation bug.
It is truncated generation. A new setup-only length calibration is therefore
the next legitimate action. Capped trajectories remain wrong/incomplete; they
will not be rescued with post-hoc answer extraction.

No existing artifact shows that OPD improved a student. The earlier task-RL
student runs are training pilots with absolute held-out scores, not valid
before/after improvements, because the raw student was not evaluated under the
identical held-out contract.

## What teacher run 108609 actually trained

The registered O teacher used Qwen3-8B, LoRA rank 16, four completions per
prompt, a 1,024-token completion cap, and 100 prompt groups. The implementation
correctly calls itself “GRPO with DAPO loss normalization; not the complete
DAPO recipe.”

| Measurement | Observed |
|---|---:|
| Declared O `teacher_train` rows | 55,033 |
| Matched selected pool | 4,322 |
| Unique prompt groups sampled | 100 (2.31% of selected pool) |
| Completion samples | 400 |
| Correct samples | 116 (29.0%) |
| All-zero groups | 61 |
| All-correct groups | 21 |
| Mixed-reward groups | 18 |
| Nonzero-gradient steps | 16 |
| Samples at 1,024-token cap | 178 (44.5%) |
| Correct capped samples | 3 |
| All-capped groups | 31 |
| All-capped and zero-reward groups | 29 |
| Intermediate checkpoints available | 0 |

For ordinary within-group GRPO, all-equal reward groups provide no relative
advantage. Two nominally mixed groups also produced no update because all four
completions were truncated and masked. Thus 100 optimizer steps materially
overstates the amount of learning signal: only 16 steps updated the adapter.
Because each step sampled a different problem, the reward trace is not a
learning curve, and the final-only save policy prevents checkpoint selection or
a duration/overtraining audit.

There is also an explicit objective/measurement mismatch. Teacher GRPO calls
the TRL-style accuracy contract (`NormalizationConfig(units=True)`), whereas
student training and evaluation use the repository's separate stricter
normalization/status path. The code preserves that distinction honestly, but
the campaign never measured the disagreement rate. A future teacher recipe
must either use one registered score-once reward contract end to end or freeze
both contracts and report a blinded disagreement matrix before training. It
must not choose between scorers after seeing which one yields the preferred
teacher gap.

Finally, `loss_type="dapo"` supplies DAPO loss normalization but not DAPO's
full dynamic-sampling recipe. With 82% all-equal groups, that omission is
material. A successor may increase generations per prompt or introduce a
preregistered sampler, but it may not describe this predecessor as complete
DAPO.

## Held-out O-teacher movement

The canonical all-record evaluation contained 4,585 records and four samples
per record. Before the score-ledger population restriction, the stored samples
showed:

| Surface | Stored accuracy | Parse-failure fraction | Mean completion tokens | Samples at cap |
|---|---:|---:|---:|---:|
| Raw Qwen3-8B | 26.4667% | 44.6619% | 811.16 | 8,819 / 18,340 |
| O-trained Qwen3-8B | 27.1101% | 44.3730% | 810.45 | 8,826 / 18,340 |

The paired stored-sample transitions were 974 incorrect-to-correct and 856
correct-to-incorrect, for a net 118 additional correct completions. Of the 974
improvements, 464 moved from prediction-parse failure to correct. The trained
model also changed 1,272 base-capped samples into non-capped samples. This does
not invalidate the gain; it shows that “learned to finish an answer under the
cap” and “learned more mathematics” are not yet separated.

The citeable result remains the stricter post-hoc symbolic-eligible result:
4,434 of 4,585 records eligible, 27.1876% to 27.8360%, paired +0.6484 percentage
points, 95% bootstrap CI [+0.1804, +1.1164] points. It is a small positive
teacher gap, not evidence of teacher sufficiency or OPD student improvement.

## Parse failures: structural diagnosis

| Surface | Samples | Accuracy | Parse-failure fraction | At cap | Parse failures at cap | Parse failures below cap |
|---|---:|---:|---:|---:|---:|---:|
| Raw student M support | 8,644 | 52.4294% | 39.2411% | 3,462 | 3,391 | 1 |
| Raw student O support | 8,644 | 10.4118% | 76.0065% | 6,785 | 6,520 | 50 |
| Raw O teacher gap | 18,340 | 26.4667% | 44.6619% | 8,819 | 8,053 | 138 |
| Trained O teacher gap | 18,340 | 27.1101% | 44.3730% | 8,826 | 8,002 | 136 |

The old M and O task-RL pilots used the same inadequate 512-token student cap.
Their 400 training samples yielded only about 20 and 12 mixed-reward groups,
respectively. The former student-support threshold asked only whether *some*
trainable variation existed; it did not certify a healthy generation setup.

This is consistent with the primary-source contracts. The Qwen3 model card
recommends up to 32,768 output tokens generally and 38,912 for complex math and
programming. The OpenR1-Math-220k card says its teacher traces were generated
with a 16K cap and only about 75% of problems were solved within 8K. Our 512 and
1,024 caps are therefore unusually restrictive for this source.

The registered successor plan is
`configs/opd_math/teacher_evaluator_qualification_plan.json`. It calibrates
2,048 and 4,096 tokens first on 64 setup-only O `student_opd` records with two
samples each, separately for the raw student and paired raw/trained teachers.
It escalates to 8,192 only if needed. The smallest cap with at most 5% capped
samples, at most 2% below-cap parse failures, and no material verifier failures
is frozen for the relevant model family. If 8,192 fails, the next action is a
new preregistered compact prompt, not an answer-recovery heuristic.

## Model/dataset exposure: what can and cannot be verified

| Question | Defensible status |
|---|---|
| Did Qwen3 see exact MATH items in training? | Unknown. MATH predates Qwen3 and Qwen3 reports MATH evaluation, but Qwen3 does not publish item-level pretraining/post-training membership. |
| Did Qwen3 see exact OpenR1-Math-220k items? | Unknown. OpenR1-Math-220k was public before the Qwen3 repository release, so exposure is temporally possible; no Qwen3 source discloses membership. |
| Do the Qwen3 model cards identify training datasets? | No dataset metadata are registered on the 8B or 1.7B cards. The Qwen3 report/blog disclose approximately 36T pretraining tokens, synthetic math/code data, long-CoT cold start, and reasoning RL, not item membership. |
| Does Qwen2.5-Math decontamination prove Qwen3 decontamination? | No. Qwen2.5-Math documents 13-gram MATH decontamination, but the Qwen3 sources do not state that the same item-level contract applies. |
| Can black-box memorization probes verify exposure? | No. They can flag suspicious behavior, but absence/presence of a memorized continuation is not proof of private training membership. |
| Can we add a cleaner external check? | Yes. AIME 2026 postdates the pinned July 2025 Qwen3 checkpoint and cannot be in those checkpoint weights. Its 30 problems are a small external check and require repeated samples, not a replacement for the main held-out sets. |

Primary sources:

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- [Qwen3 training overview](https://qwenlm.github.io/blog/qwen3/)
- [OpenR1-Math-220k dataset card](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [MATH-lighteval dataset](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)
- [Qwen2.5-Math decontamination disclosure](https://qwenlm.github.io/blog/qwen2.5-math/)
- [MathArena contamination-aware recurring evaluation](https://arxiv.org/abs/2505.23281)
- [AIME 2026 dataset](https://huggingface.co/datasets/MathArena/aime_2026)

The paper-level claim should therefore be conditional: controlled additional
post-training of a shared base checkpoint changes teaching utility. It should
not claim that either base model had or had not previously seen MATH/OpenR1.

## Baseline ledger reset

| Arm | Question answered | Current status |
|---|---|---|
| Raw student | What does the frozen 1.7B checkpoint do? | Exists on student-support training-role data, but not on the identical 370-row source holdout used for trained students. |
| Task-RL | Does task reward alone help at matched compute? | M and O pilots completed; absolute held-out scores were 52.432% and 11.149%, but no matched raw-heldout delta exists and truncation was severe. Re-run after hardening. |
| Offline distillation | Is on-policy collection better than fixed teacher traces/logits? | Required comparison; not yet implemented or run. |
| Bare OPD | Does teacher likelihood alone move the policy? | Plumbing-only smoke exists; no held-out improvement result. Treat as a collapse diagnostic. |
| Reward-gated OPD | Does teacher signal add value beyond task-RL? | Central arm; no scientific student result yet. |

Every successor arm must share initialization, seed, prompt order, optimizer
steps, rollout length, held-out records, and evaluation seeds within a
replicate. The primary effects are reward-gated OPD minus task-RL,
reward-gated OPD minus offline distillation, and each trained arm minus raw.
Report paired completion reward as primary and retain pass@k, cap rate,
below-cap parse failure, calls, tokens, latency, and verifier unknowns.

## Successor order of operations

1. Run and seal the setup-only length calibration. Select student and teacher
   caps without opening `teacher_gap_dev`, `source_holdout`, or external tests.
2. Align teacher optimization and evaluation reward/status contracts, or
   predeclare their disagreement analysis. Save intermediate checkpoints.
3. Qualify a teacher recipe with at least 100 nonzero-gradient updates, no more
   than 5% capped samples, at least 25% nonzero-gradient steps, and at least 25%
   mixed-reward groups. Select duration on disjoint tune-dev data.
4. Freeze the recipe, use three seeds for any paper claim, and open the final
   teacher gap once per seed.
5. Rebuild raw and task-RL student baselines under the same hardened contract,
   then run offline distillation, bare OPD, and reward-gated OPD.
6. Evaluate O source holdout, MATH as external transfer only, and AIME 2026 as
   a small post-checkpoint check. Do not resurrect the failed M teacher.

## Evidence custody

Large source artifacts remain on EIT under
`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/`. The read-only
reconstruction script is `scripts/opd_math/qualification_audit.py`. CPU Slurm
job `126821` ran it against the sealed artifacts without invoking the verifier.
The exact output lives at
`artifacts/legalrag/opd_math/qualification/teacher_evaluator_baseline_bd1ca8b_v1/audit.json`;
the tracked copy is
`evidence/july_2026/opd_teacher_evaluator_qualification_bd1ca8b_v1.json`
(SHA-256 `2255b2f7...400b9f1`).

## Links

[[opd-verifier-ledger-boundary-2026-07-22]] ·
[[opd-math-verifier-recovery-2026-07-20]] · [[opd-distillation]]
