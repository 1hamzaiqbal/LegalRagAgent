---
title: OPD gated campaign v2
date: 2026-07-22
status: terminal at setup-only length calibration
tags: [opd, math, campaign, gates, baselines, teacher, distillation]
---

# OPD gated campaign v2 - 2026-07-22

## Operating decision

Run the OPD program as a semantic-gated state machine. Scheduler success may
trigger an exact merge or read-only validator, but it may not trigger model
training. A new training stage requires a passing result artifact, independent
reconstruction, and an explicit rearm record.

The machine-readable source of truth is
`configs/opd_math/gated_campaign_v2.json`. This page explains why the sequence
is deliberately narrower than a fully prequeued grid.

## Terminal Stage 1 result

No scientific job is authorized in this campaign. Jobs `126824`--`126836`
completed the registered 2,048/4,096 calibration. The student qualified at
4,096, while the teacher required the conditional 8,192 candidate. Jobs
`126883` and `126884` evaluated only the raw and trained teacher arms at 8,192;
jobs `126885` and `126886` performed exact merges, and read-only selector job
`126887` evaluated all preserved candidates.

The raw teacher passed at 8,192 with 4/128 samples at cap (3.125%). The trained
teacher failed with 7/128 at cap (5.46875%), exactly one sample above the
immutable maximum of 5%. Both had zero parse failures below cap and zero
verifier errors. The official selector and a fresh independent rerun are
byte-identical at SHA-256 `044cbcae...d852`; teacher status is
`FAILED_ALL_CANDIDATES`, and scientific training authorization is false.

This is the registered terminal condition. Stage 2 and all model-training
stages remain closed. A compact-prompt qualification may exist only as a new
campaign version on fresh setup-only records; it cannot relax, overwrite, or
rescue this result. The compact tracked receipt is
`evidence/july_2026/opd_length_calibration_terminal_6d3be08_v1.json`.

## Why later jobs were not prequeued

`afterok` means that a program exited zero. It cannot establish any of the
facts on which the next scientific stage depends:

- trajectories usually terminate before their cap;
- parser failures below the cap are rare;
- teacher/student reward contracts agree or have a frozen disagreement rule;
- a teacher receives enough informative gradients and improves held-out task
  reward;
- task-RL improves the raw student under a matched contract;
- the teacher assigns higher likelihood and reward on student trajectories;
- OPD adds value beyond task-RL and offline distillation.

Prequeuing teacher or student training behind a zero exit code would violate
the user's instruction not to build on weak components. Instead, the EIT
conductor receives authority to launch only the next action written in a
passing semantic result.

## Campaign funnel

1. **Generation/evaluator health:** choose student and teacher caps on setup
   data. Parse failures caused by truncation are not rescued.
2. **Reward-contract alignment:** replace the teacher-TRL versus evaluator
   mismatch with a registered score-once contract. Verifier unknowns skip a
   whole prompt group and abort above 0.1%; they never become retry-until-pass.
3. **Teacher recipe qualification:** use setup/tune-dev data, periodic
   checkpoints, and at least 100 real gradient updates. Group size 4 versus 8
   is tested before extending from 250 to 500 steps.
4. **Teacher confirmation:** freeze the recipe, train three seeds, and open the
   complete teacher gap once. Negative seeds remain in the analysis.
5. **Student baseline qualification:** raw and task-RL share cap, prompt order,
   compute, reward contract, and three seeds. Task-RL must improve tune-dev
   before OPD is meaningful.
6. **Teacher/student interface:** verify exact tokenizer alignment,
   full-vocabulary K1 custody, teacher NLL advantage, teacher task advantage,
   and a finite one-step update. This is still plumbing, not improvement.
7. **Objective-family pilot:** compare task-RL, offline distillation, bare OPD,
   and reward-gated OPD on one frozen tune-dev seed. Reward-gated OPD must add
   value beyond both substantive baselines.
8. **Confirmatory matrix:** only after the pilot passes, train three frozen
   seeds and evaluate O source holdout, MATH external transfer, and AIME 2026.
9. **MOPD later:** only after single-teacher OPD works and a second non-M
   teacher independently qualifies. The failed M teacher remains excluded;
   DeepMath C requires a new qualification after its previous negative.

## Stop rules

- No M retraining, M merge, M_M, or M_O.
- No teacher training if length or reward-contract qualification fails.
- No OPD if the O teacher lacks a reproducible held-out gap or likelihood
  advantage on student trajectories.
- No confirmatory grid if reward-gated OPD fails to beat matched task-RL and
  offline distillation on the frozen development analysis.
- A changed prompt, scorer, dataset, cap, sampler, or optimizer recipe creates
  a new campaign version. It does not overwrite a failed result.

## Claim surface if the funnel succeeds

The primary result is not a new architecture. It is a controlled answer to:

> Under what teacher, student, data, and trajectory conditions does on-policy
> distillation add task value beyond reward-only training and ordinary
> distillation?

The extensible measurements—teacher likelihood advantage, task advantage,
mixed-group rate, truncation, reward-contract disagreement, and cost—leave room
for a transferable rule or metric to emerge without requiring that conclusion
in advance.

## Links

[[opd-teacher-evaluator-baseline-qualification-2026-07-22]] ·
[[opd-verifier-ledger-boundary-2026-07-22]] · [[opd-distillation]]
