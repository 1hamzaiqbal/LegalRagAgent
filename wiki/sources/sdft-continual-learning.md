---
title: SDFT - Self-Distillation Enables Continual Learning
type: source
tags: [self-distillation, continual-learning, context-distillation, forgetting, skills]
created: 2026-07-17
updated: 2026-07-17
status: maintained with method-custody warning
url: https://arxiv.org/abs/2601.19897
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2601.19897.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Self-Distillation
authors: Shenfeld et al.
year: 2026
venue: ICML 2026 Spotlight
---

# Self-Distillation Enables Continual Learning - SDFT

## TL;DR

Self-Distillation Fine-Tuning (SDFT) samples a student's own rollout, then has
an EMA copy of the same model re-score the student prefixes while conditioned
on an instance-specific expert demonstration. It internalizes skills and new
facts while preserving prior capabilities substantially better than SFT in the
reported Qwen2.5-7B experiments.

This is a strong collision with any broad claim that privileged context can be
distilled into weights, removed at deployment, and used for continual skill or
knowledge acquisition. It does **not** compare several fixed procedural
artifacts as runtime context and as matched post-withdrawal curricula, and it
does not study forced external-action value or teacher-to-student action-policy
regret.

## Critical method-custody correction

The paper and the empirical code disagree about the objective that produced
the headline results:

- arXiv v1 and the ICML/OpenReview manuscript describe reverse KL, present a
  full-vocabulary analytic reverse-KL estimator, and build an IRL-style
  interpretation around it;
- the official repository states that **all paper results were actually
  produced with student-prefix/on-policy sampling but per-token
  full-vocabulary forward KL**; and
- the code confirms that forward KL is the default and `main.py` does not
  override it.

As of 2026-07-17, arXiv still exposes only v1 despite the repository's promise
of a correction. The demonstrated recipe should therefore be described as:

> student-prefix/on-policy sampling followed by full-vocabulary forward-KL
> matching to a demonstration-conditioned EMA teacher.

Do not cite the paper's reverse-KL derivation or estimator discussion as an
empirical explanation of the reported numbers.

## Method and setting

- Student input: task question only.
- Teacher input: the same question plus an instance-specific correct response,
  rationale, article/answer pair, or tool call.
- The student generates one on-policy rollout.
- An EMA teacher scores the full vocabulary at the student-generated prefixes
  while seeing the privileged demonstration.
- Full-parameter fine-tuning; generally three seeds with mean and 95%
  confidence intervals.
- Main reader: Qwen2.5-7B-Instruct. Scaling uses 3B, 7B, and 14B; an additional
  reasoning test uses OLMo-3-7B-Think.
- Reported hardware: one H200.

Skill tasks are SciKnowEval Chemistry L3, ToolAlpaca API-call generation, and
HuatuoGPT-o1-derived medical reasoning. Knowledge acquisition uses nine 2025
natural-disaster Wikipedia articles (about 200,000 source tokens), generated
questions, and direct/OOD tests. Prior-capability retention averages
HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, and HumanEval.

## Main evidence

Target accuracy / prior-capability average:

| Task | Base | SFT | SFT + re-invoke | DFT | SDFT |
|---|---:|---:|---:|---:|---:|
| Science | 32.1 / 65.5 | 66.2 / 53.4 | 66.0 / 60.2 | 54.8 / 60.2 | 70.2 / 64.5 |
| Tool use | 42.9 / 65.5 | 63.2 / 56.0 | 63.1 / 63.7 | 64.2 / 60.8 | 70.6 / 65.4 |
| Medical | 30.1 / 65.5 | 35.5 / 60.2 | 35.6 / 62.6 | 36.2 / 64.0 | 40.2 / 65.4 |

SDFT improves target accuracy over SFT by 4.0, 7.4, and 4.7 points while
retaining within 0.1-1.0 points of the base prior-capability average. This is
evidence of **substantially reduced**, not literally zero, forgetting.

Knowledge acquisition, strict / lenient / OOD accuracy:

| Method | Result |
|---|---:|
| Base | 0 / 0 / 0 |
| Oracle RAG | 91 / 100 / 100 |
| Continued pretraining | 9 / 37 / 7 |
| SFT | 80 / 95 / 80 |
| SDFT | 89 / 100 / 98 |

Further anchors:

- Science SDFT-minus-SFT gap by size: **-3.3 at 3B, +4.0 at 7B, +6.9 at
  14B**. Self-distillation is not uniformly useful for weaker readers.
- OLMo medical accuracy / average output tokens: base 31.2 / 4,612; SFT
  23.5 / 3,273; SDFT 43.7 / 4,180.
- Knowledge-context ablation: answer only 37% strict accuracy, article only
  75%, article plus answer 89%. The paper's prose says answer-only outperforms
  article-only, the opposite of the plotted values; preserve the figure values
  and treat the accompanying interpretation as internally inconsistent.
- The paper estimates about 2.5x SFT FLOPs and 4x SFT wall-clock time.
- More than one student trajectory per prompt yielded negligible reported
  benefit.

## Continual-learning claim boundary

The sequential experiment trains one Qwen2.5-7B model in one fixed order:
`Tool Use -> Science Q&A -> Medical`. Its figure normalizes each task so zero
is base accuracy and one is the maximum reached by either method. SDFT visually
retains earlier tasks better than SFT, but the paper does not provide raw
sequential endpoints, standard forgetting/BWT/FWT metrics, task-order
permutations, or full uncertainty at every boundary. The separate raw
prior-capability table comes from single-task runs, not the sequential run.

The defensible statement is that SDFT reduces forgetting in this fixed-order
three-task experiment. “Continual learning without performance regression” is
broader than the evidence.

## Exact boundary for our questions

SDFT occupies:

- privileged demonstration context present during training and absent at
  inference;
- same-model on-policy context distillation into weights;
- skill and factual-knowledge acquisition after context withdrawal;
- reduced general-capability forgetting versus SFT; and
- scale-dependent self-teaching success.

It does not contain:

- forced internal/external potential outcomes or action cost;
- a canonical external action/payload crossed across readers;
- cross-family teacher-to-student action-value transport;
- reusable, independently versioned `SKILL.md` artifacts;
- several competing artifacts for one task;
- a target's runtime-context ranking versus the same artifacts'
  post-withdrawal causal acquisition ranking; or
- matched-cost artifact removal, rollback, or update experiments.

SDFT is therefore a mandatory internalization baseline for the secondary
[[skill-lifecycle-research-snapshot-2026-07-17]] study. It is not a collision
with the primary [[research-question-recommendation-2026-07-17|forced-action
value]] estimand.

## Design lessons

1. Gate training on a directly measured target-specific contextual-teacher
   gap; the negative 3B result makes this non-negotiable.
2. Verify teacher correctness independently rather than treating privileged
   conditioning as correctness.
3. Compare forward and reverse KL explicitly; do not inherit the paper/code
   label mismatch.
4. Include frozen, current-student, and EMA teacher controls.
5. Report raw old-task endpoints, multiple task orders, standard continual-
   learning metrics, and unrelated-capability retention.
6. Audit context-marker leakage: the paper reports phrases such as “Based on
   the text,” and the code heuristically masks the first three completion
   tokens.
7. Match or amortize compute against SFT/direct training; deployment context
   removal does not make training free.

## Code and artifact custody

- Audited paper: arXiv v1, 27 January 2026; CC BY 4.0.
- PDF SHA-256:
  `c27949d1b7888b128bf40f6b031d4e84a82b8ba40d6e03d671491f8e414a516b`.
- Project page: https://self-distillation.github.io/SDFT/.
- Official repository: https://github.com/idanshen/Self-Distillation.
- EIT checkout pinned at
  `d77573212fa0a3ae2eeb64b9b44db1c251f75e3e` on 2026-07-17.
- ICML 2026 Spotlight record: https://openreview.net/forum?id=qA6FgH0nnZ.

The repository is partial reproduction support: it contains Science and Tool
Use plumbing but not the Medical/Wikipedia experiments, sequential-training
runner, full baseline/sweep suite, checkpoints, result logs, or plots. Its
README clone command points to a different/non-resolving organization path,
one evaluation filename is stale, and no top-level license was present at the
pinned commit.

## Links

[[self-distillation-cluster-update-2026-07-17]] ·
[[opsd-self-distilled-reasoner]] · [[sdpo-rich-feedback]] ·
[[continual-facts-in-weights]] · [[opcd]] ·
[[skill-lifecycle-research-snapshot-2026-07-17]]
