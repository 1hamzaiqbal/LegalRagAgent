# OPD gated-campaign conductor handoff

Date: 2026-07-22

## Objective

Continue `opd_math_gated_campaign_v2` autonomously from immutable EIT
artifacts. Complete and independently audit each registered building block.
Launch only the next action explicitly authorized by a passing semantic result.
Do not stop merely because a job takes a long time, but stop a scientific
branch immediately when its registered terminal condition is met.

Read, in order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/OPERATIONS.md`
4. `configs/opd_math/gated_campaign_v2.json`
5. `configs/opd_math/teacher_evaluator_qualification_plan.json`
6. `wiki/snapshots/opd-gated-campaign-v2-2026-07-22.md`
7. `wiki/snapshots/opd-teacher-evaluator-baseline-qualification-2026-07-22.md`
8. `wiki/snapshots/opd-verifier-ledger-boundary-2026-07-22.md`

## Terminal resolution

This handoff was executed to its registered terminal condition on 2026-07-22.
Jobs `126824`--`126836` completed the initial calibration; teacher-only 8,192
jobs `126883`/`126884`, exact merges `126885`/`126886`, and selector `126887`
also completed. Independent reconstruction is byte-identical to the official
selector (SHA-256 `044cbcae...d852`). The student is `QUALIFIED` at 4,096, but
the teacher is `FAILED_ALL_CANDIDATES`: the trained arm has 7/128 samples at
the 8,192-token cap (5.46875%), one sample above the immutable 5% maximum.

The outgoing directive is `STOP`. No Stage 2 work or model training is
authorized in this campaign. A compact prompt requires a separate campaign
version and fresh setup-only records. The tracked receipt is
`evidence/july_2026/opd_length_calibration_terminal_6d3be08_v1.json`.

## Machine lanes

- Conductor/edit clone:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-conductor`
- Active job clone:
  `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math`
- Persistent artifacts:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math`
- Canonical data:
  `/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1_canonical_reviewed_19b24c2`

The active job clone was pinned to commit
`8feb248a095451abab00a40ba0d9f40897732e06` while jobs 126824--126836 were
live. After every job became terminal, it was cleanly fast-forwarded to
`6d3be084da3f7854ccde296ab20124c3d3acd5f3` for the registered 8,192 rearm.
Fresh commit-specific train/serve freezes were created and verified before
that launch. There is no currently authorized job launch.

## Initial Stage 1 jobs (terminal)

| Surface | Cap | GPU job | CPU merge |
|---|---:|---:|---:|
| trained O teacher | 4,096 | 126824 | 126835 |
| raw O teacher | 4,096 | 126825 | 126834 |
| raw O teacher | 2,048 | 126826 | 126830 |
| trained O teacher | 2,048 | 126827 | 126833 |
| raw student | 4,096 | 126828 | 126831 |
| raw student | 2,048 | 126829 | 126832 |

Read-only selector job 126836 depends on all six merges and writes:

`/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/qualification/length_calibration_8feb248_v1/calibration.json`

## Stage 1 decision procedure

1. Audit every GPU and merge job with `sacct`, exact stdout, output custody,
   summary/sample hashes, common sample identities, and clean commit/freeze
   bindings.
2. Independently rerun `scripts/opd_math/length_calibration.py` into a fresh
   directory. The result must be byte-identical or semantically identical with
   all source hashes equal.
3. If both student and teacher status are `QUALIFIED`, write and commit a
   compact tracked receipt and update the campaign snapshot/signoff log. Freeze
   each selected cap. Then proceed to Stage 2 only.
4. If either family is `NEEDS_NEXT_CANDIDATE`, preregister and launch 8,192
   only for the missing family/arms, with the same 64 records, two samples,
   seed, decoding, and model/adapter identities. Merge and run a fresh selector
   over all candidate artifacts. Do not rerun successful smaller surfaces.
5. If either family is `FAILED_ALL_CANDIDATES`, record the negative result and
   stop all teacher/student training. A compact-prompt campaign requires a new
   version and fresh setup-only records.

## Stage 2 boundary

Stage 2 is reward-contract implementation and validation, not model training.
The predecessor teacher optimized TRL-style normalization while evaluation
used a separate verifier. Implement the `S2_reward_contract_alignment`
requirements literally:

- define eligibility from gold only, before model outputs;
- score once and retain status/provenance;
- on training-side verifier unknown, discard the whole prompt group without an
  optimizer step and abort above 0.1%;
- on evaluation-side unknown, retain bounded uncertainty rather than retrying
  until favorable;
- build a blinded old-versus-successor disagreement report on setup-only data;
- add focused unit/integration tests and a CPU independent reconstruction;
- preserve all predecessor traces and gates unchanged.

Only a frozen, passing Stage 2 artifact can rearm Stage 3 teacher-recipe
qualification. Do not train a teacher while designing or debugging Stage 2.

## Later stages

Follow `configs/opd_math/gated_campaign_v2.json` exactly. In particular:

- qualify teacher recipe health and tune-dev gain before final teacher-gap
  confirmation;
- require all three teacher seeds and retain negative seeds;
- qualify raw/task-RL student baselines before OPD;
- require teacher likelihood and task advantage on student trajectories;
- compare task-RL, offline distillation, bare OPD, and reward-gated OPD on
  tune-dev before the confirmatory matrix;
- open O source holdout, MATH transfer, and AIME 2026 only for the frozen
  confirmatory analysis;
- defer MOPD until single-teacher OPD works and a second non-M teacher passes
  independently.

## Non-negotiable exclusions

- M remains permanently failed: no M retraining, M merge, M_M, or M_O.
- No threshold relaxation, outcome-adaptive rescoring, answer recovery from
  capped text, silent prompt changes, or retry-until-pass.
- No downstream scientific launch from Slurm `COMPLETED`, finite loss,
  checkpoint creation, or nonzero KL alone.
- No full experimental grid before its building blocks pass.
- Keep large outputs on EIT; track compact receipts, plans, and snapshots in
  Git with hashes.

## Reporting

Append immutable intent, submission, result, independent-audit, and rearm
records under the conductor artifact directory. Keep the Git branch clean and
push every durable code/plan/snapshot commit. When blocked by a scientific
failure, record it as the result rather than inventing a rescue. When user
input is genuinely required, leave one concise handoff naming the exact choice
and do not broaden authority.
