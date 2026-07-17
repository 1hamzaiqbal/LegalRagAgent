---
title: Execution Checklist
type: checklist
tags: [execution, experiments, handoff]
created: 2026-07-17
updated: 2026-07-17
status: proposed
---

# Execution checklist

## Main-thread decision

- [ ] Accept, revise, or reject the resource-response preservation question.
- [ ] Decide whether the first paper target is an evaluation/finding paper or
      a method paper only if a baseline failure appears.
- [ ] Approve the one-tool Reasoning Gym pilot and non-legal starting domain.

## Environment and data

- [x] Archive close papers and repositories on EIT.
- [x] Download both Agent Distillation trajectory files and record EDA caveats.
- [x] Add deterministic Reasoning Gym pilot generator.
- [x] Detect and handle deterministic `countdown` generation failures while
      recording their rate.
- [x] Complete Reasoning Gym EIT generator smoke using project-scoped uv
      cache/Python paths.
- [x] Regenerate the pilot independently and verify byte-identical JSONL and
      summary hashes.
- [x] Run all oracle answers plus empty/non-prime-factor scorer audits.
- [x] Manually inspect one example from each of the 12 family/tier cells.
- [ ] Expand the manual review and add shortest-path/countdown malformed-answer
      tests before freezing the final benchmark.
- [ ] Freeze final train/validation/test generator configs and seed ranges.

## Harness

- [x] Specify the project-owned trace schema and add a dependency-free
      validator/rescoring self-test.
- [ ] Integrate the trace validator with a real model/harness episode.
- [ ] Implement scripted always/never/difficulty policies.
- [ ] Wrap Reasoning Gym + Python in Verifiers.
- [ ] Add Inspect AI adapter only after the canonical smoke passes.
- [ ] Decide NeMo Gym fallback based on actual composition friction.

## Teacher gate

- [ ] Select one open-weight teacher checkpoint and revision.
- [ ] Draft one concise tool/verification skill and length-matched placebo.
- [ ] Run tiny deterministic paired-price evaluation.
- [ ] Run stochastic skill/no-skill switching surface.
- [ ] Evaluate switching prevalence, monotonicity, and baseline frontier.
- [ ] Stop if the predeclared gate fails.

## Transfer

- [ ] Conditional trace SFT overfit smoke.
- [ ] Fixed-price and mixed-price SFT baselines.
- [ ] Direct price-conditioned task RL baseline.
- [ ] Reward/gap-gated OPD only after compatible-tokenizer scoring tests.
- [ ] OPID/external skill baselines only if needed for the paper claim.

## Analysis

- [ ] Implement paired task-level metrics and clustered bootstrap.
- [ ] Separate seen, interpolated, and extrapolated prices.
- [ ] Plot raw success/calls before scalarized utility.
- [ ] Audit exact success versus native shaped reward.
- [ ] Inspect every action reversal in the smoke set.

## External validity

- [ ] Add one cost-augmented multi-tool benchmark after Phase 1.
- [ ] Add one skill/harness benchmark if the skill factor is retained.
- [ ] Add legal retrieval only with a valid corpus and authority-aware verifier.
- [ ] Run a final July-2026-forward literature search before submission.

## Documentation contract

- [ ] Every run appends a source log and experiment-ledger row.
- [ ] Every citable result gets a completion audit mapping cells to JSONL.
- [ ] Update the literature matrix for any paper that changes the claim.
- [ ] Preserve old Snap-HyRE/SCOPE artifacts as historical branches; do not
      rewrite their claims into this new project.
