---
title: Big-Math and Big-Math-RL-Verified
type: source
tags: [opd, math, dataset, rlvr, difficulty, decontamination]
created: 2026-07-20
updated: 2026-07-20
status: scoped read; dataset access gated on EIT
url: https://arxiv.org/abs/2502.17387
dataset: https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified
local: /engrfs/project/jacobsn/hiqbal/literature/legalrag/papers/arxiv_2502.17387.pdf
repo: /engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/Big-Math
authors: Albalak et al.
year: 2025
---

# Big-Math

## TL;DR

Big-Math-RL-Verified is the strongest candidate found for a future second
teacher source because it combines 251,122 closed-form, open-ended problems
with source/domain metadata and a 64-rollout Llama-3.1-8B solve rate. Those
fields support a model- and outcome-blind curriculum or difficulty match before
our Qwen teachers train.

It is not the active replacement today. An authenticated Hugging Face dry run
from EIT returned gated-access `403`, and the corpus includes MATH, NuminaMath,
AoPS, and competition-derived sources. Access alone would not make it eligible:
we would still exclude the MATH component and build one collision graph against
O, M, and every frozen evaluation set before preregistration.

## What the paper establishes

The paper starts with 643,374 problems from HARP, Omni-MATH, and retained
NuminaMath sources, then reports 251,122 after source-specific and global
filtering. The final count includes 47,010 multiple-choice questions
reformulated into open-ended problems. Filters target proof, multipart,
yes/no, true/false, hyperlink-dependent, and non-closed-form content.

Deduplication is not a guarantee of independence for our study. The paper uses
exact matching and SemDeDup with all-MiniLM-L6-v2, and decontaminates the named
500-problem MATH and Omni-MATH test sets by string matching. Our O/M source
lineage and later evaluation inventory are different and require a fresh,
auditable collision pass.

For difficulty, the authors generate 64 Llama-3.1-8B rollouts per problem. The
paper reports 91,647 problems below 20% solve rate and 71,926 above 80%. This
is valuable selection metadata, but it cannot substitute for the registered
raw Qwen3-8B and Qwen3-1.7B feasibility surface: difficulty is reader-specific.

Anchors read: dataset construction and Table 1 on PDF pp. 3-4; filtering and
decontamination on pp. 5-6; difficulty analysis and Figure 3 on pp. 10-11;
limitations/future directions on pp. 15-16.

## Role in LegalRagAgent

- Best scientific candidate if access is granted before candidate selection is
  frozen.
- Not needed for the current O-teacher objective-family campaign, which already
  uses MATH and OpenR1 as two student/evaluation distributions.
- Potential `C` source for a later O/C teacher-source x student-source matrix
  or multi-teacher routed OPD study.
- Never a post-failure rescue: once another candidate teacher outcome is
  observed, Big-Math cannot be substituted into that same campaign.

## Version and custody

- Paper PDF SHA-256:
  `3cedf59056a5ff9ec8984c33c61f12e0dcba20e09e8b33434f29a6a606d39831`.
- Official MIT-licensed filtering/reformulation repository:
  https://github.com/SynthLabsAI/Big-Math.
- EIT repository checkout pinned at
  `420b9a771a7e97a85b81cbdcbd573b1b0d56f522`.
- Dataset card reports Apache-2.0 and 251,122 rows; its gated data bytes were
  not available to this audit.

## Links

[[deepmath-103k]] - [[opd-m-teacher-clarification-and-source-options-2026-07-20]] -
[[opd-objective-family-expansion-2026-07-20]] - [[opd-math-source-transfer]]
