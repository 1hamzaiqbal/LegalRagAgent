---
title: DeepMath Candidate C Negative Qualification
type: snapshot
tags: [opd, deepmath, data-quality, preregistration, negative-result]
created: 2026-07-20
updated: 2026-07-20
status: sealed negative qualification; no C training authorization
---

# DeepMath candidate C negative qualification

DeepMath-103K does **not** qualify as source `C` under the frozen
`deepmath_C_global_collision_label_prompt_v1` data gate. Slurm job `108534`
completed the global scan over all `1,237,750` inventory rows, and an
independent replay rehashed the inventory manifest, every source receipt, the
scan manifest, and all eight registered scan outputs.

The terminal blocker is the preregistered prompt contract. Candidate C was
allowed zero prompts over the shared `1,536`-token bound. Two rows require
`1,546` and `1,692` tokens. The scan therefore reports
`data_gate_passed=false`. The gold surface otherwise reaches
`103,020 / 103,022 = 99.9981%` parseability, with zero missing candidate
problems and `92,835` preliminary eligible clusters.

The semantic scan also leaves `15,224` review-required edges unresolved out of
`53,035` review edges. That is a second blocker, but completing the review
cannot repair the already-failed prompt gate. We therefore do not run the
finalizer merely to produce a second failure, and we do not inspect raw-model
feasibility, freeze C roles, train a C teacher, or launch `C_C`, `C_O`, `O_C`,
or routed O/C arms. Changing the prompt length or dropping the two records now
would be a post-outcome rescue and is outside this campaign.

This result does not affect the separately defined O-teacher objective-family
campaign. `O_M` and `O_O` remain the active scientific cells. MATH remains a
student/evaluation distribution, while the failed M-trained teacher remains
permanently excluded.

## Immutable custody

- complete scan root:
  `/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_audit_1a6dd77_scan_v2`
- scan manifest SHA-256:
  `6a81cb47c61a6f6e2d3f6678d4a75cc8d9c11a853593be8c4be3ec768be77383`
- accepted inventory manifest SHA-256:
  `abd30c1079e663f38a2c8af21c43009800eec904cc0cf1c60267c5024fe06c59`
- persistent compact receipt:
  `/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math/campaigns/deepmath_negative_qualification_1a6dd77_108534`
- tracked decision evidence:
  `evidence/july_2026/deepmath_negative_qualification_108534.json`

## Links

[[opd-program-goal-2026-07-20]] - [[deepmath-103k]] -
[[opd-math-source-transfer]] - [[mopd-multi-teacher]]
