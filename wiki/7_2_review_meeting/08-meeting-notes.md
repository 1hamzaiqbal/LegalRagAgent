---
title: Meeting Notes — 2026-07-02 (post-rejection direction discussion)
type: meeting-doc
tags: [meeting, notes, direction, distillation, retrieval-control, benchmark]
created: 2026-07-02
date: 2026-07-02
---

# Meeting notes — 2026-07-02

Attendees: HI + PhD-student mentor (+ HL, LH contributions noted below).
Raw notes recorded verbatim-ish; **[evidence] annotations** added afterward
link claims to existing results so threads can be picked up without
re-deriving. Priority per HI: **the skill-internalization + Tinker +
small-model distillation thread is the most interesting** — the others are
recorded for later.

## Opening state (agreed)
SCOPE/HyRE was not significantly better than HyDE and not very novel — the
rejection's core is accepted ([[02-critique-analysis]]). Context point:
Claude Code / Cursor-era **agentic models** change what a retrieval paper
should look like.

## Ideas 1 + 2 — consider jointly: retrieval control for agentic models

**Idea 1 — search effort control.** Don't retrieve 200 passages OR 1
passage; make effort *flexible*. Not strictly one-by-one retrieval —
parallel retrieval? How much effort per query? Retrieval always has a cost:
the retrieval itself PLUS the model then attending to more evidence. Frame as
cost + retrieval-accuracy joint optimization. Action: survey the literature
on adaptive retrieval-effort control.
- **[evidence we hold]**: the three dials are exactly the decision inputs an
  effort controller needs — expansion margin says *whether to expand*
  ([[affinity-margin-mechanism]]), pool confusability says *how much
  selection effort* ([[judge-pilot-v0-results]]), parametric deficit says
  *whether to retrieve at all* (one llm_only run, [[judge-answer-conversion]]).
  [[qpp-routing-negative]] bounds what cheap per-query signals can do (weak) —
  an honest constraint for any controller design.
- **[adjacent lit already ingested]**: [[adaptive-rag-mallen]] (popularity-
  gated retrieval), [[qe-survey-2025]] (selective QE as folklore).

**Idea 2 — retrieval information conflict arbitration.** What happens when
retrieved information conflicts (doc vs doc)? When the model *doesn't
believe* the evidence (doc vs parametric prior)? Does one verify, or fetch
additional evidence?
- **[evidence we hold]**: the model-vs-evidence case is measured — on
  parametric-strong MC the reader overrides/ignores evidence (BarExam 70B:
  gold-present only +2.4pp; 8× exposure → +0.6pp answers). The
  [[power-noise-lostmiddle]] source covers distractor sensitivity. Doc-vs-doc
  conflict is genuinely unmeasured in our stack — new axis.

## Idea 3 — RAG benchmarking: "is the retrieved document HELPFUL to the LLM?"

RAG benchmarks reverse directions across metrics; F1/AUC measure retrieval,
but the real question is whether the document *helps the reader*. If the
correct document is retrieved but the LLM can't use it, was it useful?
Possible contribution: a new evaluation metric for the agentic era — not
"did we retrieve gold" but "did retrieval help solve the task," akin to an
agentic **cost-per-task** metric. Noted as possibly the *easiest* paper
since we have existing experiments and data.

Meeting asked to FIND these experiments — **found, all already run**:
- *"Does golden passage not improve accuracy?"* — reader-dependent, exactly
  dial 3: golden vs llm_only is **+25.7pp** on CaseHOLD/70B (97.5 vs 71.8),
  **+19.1pp** SCALR/70B, **+22.5pp** Housing/70B — but **+0.5pp ns** on
  BarExam/70B (79.2 vs 78.7) and **−2.3pp** on BarExam/Gemma-26B (78.6 vs
  80.8, gold *hurts*). Sources: CLAUDE.md signed rows +
  [signoff log](../../docs/signoff_log.md).
- *"Golden + neighbors vs golden alone"* (neighbor dilution): CaseHOLD/70B
  **97.5 → 79.4 (−18.1pp, p=1e-187)**; SCALR/70B 93.5 → 83.0 (−10.5pp);
  SCALR/8B-class 93.2 → 77.1 (−16.1pp); Housing/70B 67.3 → 66.0 (−1.3pp);
  BarExam/70B 79.2 → 77.8; but BarExam/Gemma-26B *inverts*: 78.6 → 80.7
  (+2.1pp). Adding related-but-non-gold context can cost up to 18 points even
  with gold present.
- *"Do 5 similar-but-not-golden passages help?"* — the gold-absent
  decomposition answers per-row: BarExam/70B **−3.8pp** (distractor tax),
  Housing/70B **+12.0pp** (neighboring provisions carry value), BarExam/8B
  **+7.3pp** (weak reader helped by topical context) —
  [[judge-answer-conversion]]. Arm-level: rag_simple vs llm_only = −4.1pp
  BarExam/70B, −1.1pp CaseHOLD, −1.6pp SCALR, **+2.5pp** Housing.
- The metric skeleton exists: per-row helpful/harmful evidence effect +
  break-even model + cost columns (calls/tokens/latency already logged per
  row in the harness). **Convergence note**: SKILL0's per-skill on-policy
  helpfulness Δ_k ([[skill0]]) is the same construct at the skill level —
  "helpful to this policy," not "relevant."

## HL's suggestions (search-effort mechanics)
- **Tree-based index** built on passage similarity; prune similar passages
  to cut search effort.
- **Metadata tags to shrink search space**: an SLM tags the query →
  tag-filtered retrieval. *[We have a validated special case: the Housing
  jurisdiction state filter, Hit@5 2.8 → 36.9 — the largest single retrieval
  lever in the whole project ([[01-scope-submission]] Table 10). Generalizing
  filter-first retrieval is a real thread.]*
- **Legal query decomposition**: if one question has multiple aspects, is
  that *why* multi-evidence sometimes beats single gold? Decompose into
  subtopics then search per-subtopic. *[Fits the BarExam/Gemma-26B
  golden+neighbors inversion above.]*

## LH's point (benchmark construction)
Benchmarks shouldn't contain irrelevant evidence; if multiple gold passages
are needed, the benchmark should supply *sufficient* gold. Audit whether
multi-evidence questions are constructed with that in mind. *[Related:
Housing multi-gold pools already handled group-level; Zheng single-gold
Hit@k pessimism caveat ([[zheng-cslaw]]) is the same concern from the
scoring side.]*

## Idea 4 — proactive LLM
Model autonomously decides to search while "thinking about" a question,
rather than search-on-demand. (Recorded; not developed in the meeting.)

## The priority thread — skill internalization × distillation ([[skill-distillation-bridge]])

Ingredients discussed:
- **[[skill0]]** (arXiv 2604.02268, archived + repo cloned): internalize
  in-context skills into weights via helpfulness-curriculum RL — but into
  the *same* model.
- **[[thinking-machines-expert-judgment]]**: small model + expert outcome
  labels replicates expert judgment, beating prompted frontier models.
- **Our Tinker/EIT battery** as existing proof of the pattern: trained 9B
  judge > prompted 235B (+5.3pp, [[judge-capacity-dial]]); training = label
  quality × headroom ([[judge-pilot-fiqa]], [[judge-pilot-scidocs]]);
  mixed-label judge generalizes across legal tasks for $0
  ([[judge-mixed-legal]]).

The twist proposed in the meeting: **instead of SKILL0's large model
absorbing skills into itself, have the big model's skills distilled into a
smaller model** — e.g. distill the reasoning-heavy abilities from Ideas 1+2
(search-effort control, conflict arbitration) into a small policy.

Technical constraint raised: **on-policy distillation (OPD) needs the
teacher's token distribution / top-k logprobs** — fine for open teachers,
problematic for closed-source (GPT/Claude). Action item: check the
literature for how people do distillation from closed models (sequence-level
KD on traces, on-policy student sampling with teacher scoring / GKD-style,
rejection-sampling SFT). Details + first-experiment options in
[[skill-distillation-bridge]].

## Recorded decisions / next actions
1. **Primary**: develop the skill-distillation bridge — read SDAR + SKILL1
   (novelty check), survey closed-teacher distillation, sketch a v0 on the
   free EIT lane ([[skill-distillation-bridge]]).
2. Record Ideas 1–3 for later; Idea 3 (helpfulness-metric benchmark paper)
   flagged as lowest-cost because the experiments largely exist (see found
   evidence above).
3. Lit surveys queued: retrieval-effort control; conflict arbitration;
   distillation-from-closed-models.

## Links
[[direction-2026-07]] · [[skill0]] · [[skill-distillation-bridge]] ·
[[judge-answer-conversion]] · [[judge-mixed-legal]] · [[thesis-v2]] ·
[[06-results-and-open-questions]]
