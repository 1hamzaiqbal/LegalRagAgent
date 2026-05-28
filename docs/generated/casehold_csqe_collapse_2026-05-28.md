# CaseHOLD CSQE Collapse - 2026-05-28

Verdict: **supported** for H-collapse-2nd. CaseHOLD CSQE reaches 19.6% Hit@5, only 0.016 RI over raw, while the existing Llama-70B HyDE/SCOPE caches are 51.2% and 45.0% Hit@5. The mechanism sub-check is **mixed**: CSQE moves toward gold in CE space, but much less than HyDE/SCOPE and not enough to create a meaningful retrieval lift.

## Main Table

| Arm | Generator/source | N | Hit@5 | Hits | RI vs raw | Help | Hurt | Mean CE(gold) | Mean CE delta vs raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw question | raw question | 3600 | 17.9% | 646 | 0.000 | 0 | 0 | -5.769 | 0.000 |
| Llama-70B HyDE | groq-llama70b | 3600 | 51.2% | 1844 | 0.333 | 1345 | 147 | 1.715 | 7.484 |
| Llama-70B SCOPE | groq-llama70b | 3600 | 45.0% | 1619 | 0.270 | 1190 | 217 | 0.789 | 6.558 |
| Gemma-26B CSQE | or-gemma4-26b | 3600 | 19.6% | 704 | 0.016 | 157 | 99 | -2.342 | 3.427 |

## Reading

- CaseHOLD is an intermediate-weak point, not a BarExam clone: raw Hit@5 is 17.9%, versus the prior BarExam raw Hit@5 of 1.4%.
- CSQE barely improves retrieval exposure here: 19.6% vs raw 17.9%; help=157 and hurt=99.
- The parametric expansion arms remain much stronger on the same gold-labeled set: HyDE RI=0.333 and SCOPE RI=0.270.
- The pre-stated near-zero/negative CE-delta mechanism is not literally met: CSQE's mean CE gold-affinity delta is positive at 3.427. The useful distinction is magnitude: HyDE is 7.484 and SCOPE is 6.558, so CSQE shifts toward gold but remains much weaker.
- Aggregated with the prior BarExam CSQE sweep, this gives two weak-query legal sets where CSQE is not the winning expansion arm. BarExam is the extreme weak point; CaseHOLD shows the gradient point between BarExam and HousingQA.

## Caveats

- CSQE was generated with `or-gemma4-26b`, while the HyDE/SCOPE rows are existing `groq-llama70b` signed caches. Treat this as a mechanism test for CSQE's reliance on raw-retrieved text, not as a strict model head-to-head.
- Metrics are retrieval exposure only; no downstream CaseHOLD answer cells were run.

## Sources

- Raw question retrieval: `caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl`
- Llama-70B HyDE retrieval: `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl`
- Llama-70B HyDE generation: `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_rag_hyde.jsonl`
- Llama-70B SCOPE retrieval: `caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- Llama-70B SCOPE generation: `caches/hyre/full/casehold_qfull_seed42_groq-llama70b_snap_hyre.jsonl`
- Gemma-26B CSQE retrieval: `caches/retrieval/full/casehold_qfull_seed42_or-gemma4-26b_csqe_k10.jsonl`
- Gemma-26B CSQE generation: `caches/generation/full/casehold_qfull_seed42_or-gemma4-26b_csqe.jsonl`
- Row-level points: `docs/generated/casehold_csqe_collapse_2026-05-28_points.jsonl`
- Prior BarExam/Housing CSQE context: `docs/generated/csqe_regime_sweep_2026-05-26.md`
