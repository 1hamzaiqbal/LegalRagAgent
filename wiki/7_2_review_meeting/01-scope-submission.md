---
title: SCOPE Submission - What Was Submitted
type: meeting-doc
tags: [meeting, scope, submission, rejection, ai4law]
created: 2026-07-02
date: 2026-07-02
---

# SCOPE submission - what was actually submitted

This page is a standalone reconstruction of the submitted workshop paper,
based on the copied primary PDF in `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`.
It summarizes the paper as submitted, not the post-rejection pivot.

Canonical wiki links: [[scope-paper-2026]] for the source-page recap and
[[icml-ai4law-2026-rejection]] for the critique inventory.

Primary sources: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf` and
`assets/primary/feedback.md`.

## Submission identity

The paper was titled *SCOPE: When Generated Legal Queries Help Legal RAG* and
was submitted as submission `97` to AI4Law, with authors Hamza Iqbal, Hanxun
Li, Mingzheng Li, Langlin Huang, and Jiaxin Huang
[source: `assets/primary/feedback.md`, OpenReview header].

The submission framed itself as preliminary ICML work for a legal RAG workshop:
a legal-domain retrieval problem, a generated-query method, and evaluation on
legal QA benchmarks [source: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`,
title block and Section 1].

## Core claim as submitted

The submitted claim was a legal-specific generated-query claim:

- Standard RAG fails when everyday fact-pattern language does not match formal
  legal authority language [source: PDF, Abstract and Section 1].
- HyDE-style pseudo-document retrieval is useful in general RAG but is
  brittle in legal contexts because the generated passage can miss the right
  legal frame [source: PDF, Abstract and Section 2].
- SCOPE first creates a private snap answer, then a formal legal
  pseudo-document conditioned on that snap answer, and embeds only the
  pseudo-document for retrieval [source: PDF, Section 3 and Algorithm 1].
- The final answer call receives the original question and retrieved evidence,
  but not the snap answer or pseudo-document [source: PDF, Figure 1 and
  Algorithm 1].
- The headline empirical claim was that BarExamQA gold-passage Hit@5 rose
  from `1.4%` raw-question retrieval to `9.5%`-`12.1%` under SCOPE across
  the submitted model rows, with final answer accuracy improving over
  raw-question RAG [source: PDF, Abstract, Table 1, Table 11].

The paper's strongest self-description was therefore: SCOPE is a two-call,
snap-conditioned, generated-query legal RAG method for bridging colloquial
legal questions and formal legal corpora [source: PDF, Abstract and Conclusion].

## Method as submitted

SCOPE was presented as a fixed pipeline, not a learned reranker, router, or agent.

1. Input: a formatted legal question `q`.
2. First model call: the same model generates a private draft answer `a0` and
   a separate legal search passage `p`.
3. Validation: the draft answer must have a parseable answer label and the
   passage must not contain an answer marker.
4. Retrieval: only `p` is embedded; dense retrieval returns top candidates
   from the corpus.
5. Reranking: a cross-encoder reranks the retrieved set using `p`.
6. Final answer: a second call answers from the original question plus the
   retrieved evidence; `a0` is withheld.

Source: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Algorithm 1 and
Sections 3.1-3.3.

The method's comparison to HyDE was narrow. HyDE generates a hypothetical
document directly from the question. SCOPE instead makes the model commit to a
tentative answer frame first, then writes the retrieval passage from that
frame [source: PDF, Section 3.2].

The submitted paper called the withheld snap answer a guardrail against
anchoring the final answer on an unverified guess, but it did not include an
ablation that passed the snap answer into the second call
[source: PDF, Section 3.3; `assets/primary/feedback.md`, reviewer oSUu C12].

## Datasets

The paper selected two Stanford RegLab legal retrieval benchmarks as opposite
ends of a query-corpus vocabulary-gap axis [source: PDF, Section 4.2].

| Dataset | Submitted role | QA format | Submitted size | Retrieval interface |
|---|---|---:|---:|---|
| BarExamQA | weak-query regime: everyday fact patterns vs doctrinal rule language | multiple choice | `1,195` questions | shared fact pattern plus question stem; generated methods may see options without correctness marker |
| HousingQA | strong-query regime: statutory-entailment questions already use corpus-like terms | yes/no | `6,853` questions | jurisdiction state filter applied to every main retrieval row |

Sources: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Section 4.2
and Section 4.3.

The paper did not use CaseHOLD or LegalBench-SCALR as active main-matrix
benchmarks in the submitted exact-scored comparison [source: PDF, Appendix A.1].

## Models and retrieval stack

The submitted model rows used the same model as query generator and final
answerer, so improvements could not be attributed to a stronger reader
[source: PDF, Section 4.3].

| Role | Submitted choice |
|---|---|
| Models | Llama 3.1 8B Instruct, Gemma 4 26B, Llama 3.3 70B Versatile |
| Retrieval depth | `k = 5` retrieved passages for answer rows |
| Embeddings | `Alibaba-NLP/gte-large-en-v1.5` |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder |
| Decoding controls | seed `42`, temperature `0`, answer cap `2048` tokens |
| Main modes | LLM-only, raw-question RAG, HyDE, SCOPE, Gold Evidence |

Sources: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Sections 4.3,
4.4, and Appendix A.1.

## Main answer table

The submitted headline answer table reported exact answer accuracy in percent.
Bold in the original marked the best deployable method per column; the table
below reproduces the submitted values without adding any post-submission rows.

| Method | BarExam 8B | BarExam 26B | BarExam 70B | BarExam Avg. | Housing 8B | Housing 70B | Housing Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-only | `57.3` | `80.8` | `78.7` | `72.3` | `55.4` | `44.8` | `50.2` |
| Raw-question RAG | `54.5` | `78.0` | `74.6` | `69.0` | `62.3` | `62.1` | `62.2` |
| HyDE | `56.1` | `80.3` | `80.2` | `72.2` | `59.1` | `62.2` | `60.7` |
| SCOPE | `56.9` | `82.0` | `79.7` | `72.9` | `59.0` | `59.6` | `59.3` |
| Gold Evidence | `60.0` | `78.6` | `79.2` | `72.6` | `64.3` | `67.3` | `65.8` |

Source: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Table 1.

The paper foregrounded SCOPE's gains over raw-question RAG on BarExamQA:
`+2.4`, `+4.0`, and `+5.1` percentage points for the 8B, 26B, and 70B rows,
with the 26B row described as paired-significant at McNemar `p < 0.001`
[source: PDF, Section 5.1 and Table 7].

It did not make LLM-only the primary downstream baseline in the prose. Read
against LLM-only, the BarExamQA SCOPE deltas were `-0.4`, `+1.2`, and `+1.0`
percentage points [source: PDF, Table 1; calculation from the table].

## Main retrieval table

The core retrieval table in the main body focused on Gemma 4 26B at `k = 5`.

| Method | BarExam Hit@5 | BarExam MRR@5 | Housing Hit@5 | Housing MRR@5 |
|---|---:|---:|---:|---:|
| Raw-question RAG | `1.4` | `0.7` | `36.9` | `23.3` |
| HyDE | `11.4` | `5.4` | `30.6` | `19.6` |
| SCOPE | `12.1` | `6.0` | `38.1` | `24.5` |

Source: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Table 4.

The appendix expanded that retrieval view across top-k caches. For BarExamQA,
the submitted mean over three generated-query model caches was:

| Method | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Raw-question RAG | `0.9` | `1.4` | `2.2` | `0.8` |
| HyDE | `6.4` | `10.0` | `16.7` | `5.9` |
| SCOPE | `7.1` | `10.9` | `17.3` | `6.3` |

Source: `assets/primary/97_SCOPE_When_Generated_Legal_.pdf`, Table 11.

The model-specific BarExamQA SCOPE Hit@5 values behind the abstract headline
were `9.5` for Llama 3.1 8B, `12.1` for Gemma 4 26B, and `11.0` for Llama
3.3 70B [source: PDF, Table 11].

## Supplemental controls included in the submission

The submitted appendix reported descriptive deltas, not a causal mechanism:
BarExamQA Snap-vs-raw was `+3.8pp` with `n=3`, HousingQA was `-2.9pp` with
`n=2`, and pooled HyDE-vs-Snap was `+0.1pp` with `n=5` [source: PDF, Table 6].
It also disclosed `31/42` signed cells: BarExamQA `21/21`, HousingQA `10/21`
[source: PDF, Appendix A.1].

## Other submitted claims

- Gold comparison: SCOPE matched or exceeded Gold Evidence on the two larger
  BarExamQA models, `82.0%` vs `78.6%` for Gemma 4 26B and `79.7%` vs `79.2%`
  for Llama 3.3 70B [source: PDF, Section 5.4 and Table 8].
- Efficiency: SCOPE had `268.3` correct answers per million answer-stage
  tokens, vs HyDE `244.5` and raw-question RAG `241.5`; first-stage generation
  tokens were excluded [source: PDF, Table 3].
- Exemplar probe: in Gemma 4 26B `N = 500` retrieval probes, the exemplar
  changed Hit@5/MRR@5 by `+0.6/-0.1`, `+3.0/+2.2`, and `+7.6/+7.0` on
  BarExamQA, HousingQA, and Legal-Link-EU [source: PDF, Table 5].
- Housing filter: national-corpus raw Hit@5 `2.8` became jurisdiction-filtered
  Hit@5 `36.9`, with MRR@10 `1.8` vs `24.8` [source: PDF, Table 10].

## Framing and venue positioning

The venue-facing story was legal RAG and generated legal queries. The paper
positioned BarExamQA as the weak-query case where legal fact patterns need
translation into doctrinal language, and HousingQA as the strong-query case
where statutory question text already overlaps the corpus [source: PDF,
Section 4.2].

The paper did cite general RAG and generated-query work such as RAG, HyDE,
Query2doc, FLARE, Self-RAG, Corrective RAG, LegalBench, LexGLUE, Pile of Law,
LegalBench-RAG, Legal RAG Bench, and Zheng et al.'s benchmark paper
[source: PDF, Related Work and References].

It did not cite the reviewer-named legal generated-query prior art KoBLEX /
ParSeR or GuRE [source: `assets/primary/feedback.md`, reviewer oSUu comments;
absence verified from PDF References].

## What the submitted paper did not do

These are omissions in the submitted PDF, not judgments about later work:

- No mechanism account: the paper asserted a vocabulary-gap story, but did not
  test per-query geometric affinity, query drift, factuality, leakage, or
  margin mechanisms [source: PDF, Sections 5-7; contrast with
  [[affinity-margin-mechanism]] and [[factuality-falsification]]].
- No leakage audit: the paper did not NLI-match generated passages against gold
  passages or split retrieval gains by matched vs unmatched generations
  [source: PDF, Sections 5-7; later page [[leakage-audit-barexam]]].
- No selector story: the paper used the standard cross-encoder reranker, but
  did not analyze whether candidate pools contained gold that the reranker
  buried [source: PDF, Section 4.3; later pages [[judge-pilot-v0-results]] and
  [[judge-pilot-housing]]].
- No trained legal rewriter baseline such as GuRE, and no ParSeR-style
  parametric-provision baseline [source: PDF References; `assets/primary/feedback.md`].
- No significance-tested SCOPE-vs-HyDE answer or retrieval contrast in the
  paper, even though SCOPE-vs-HyDE was the comparison that isolated the snap
  conditioning step [source: PDF, Table 6 and Section 5; reviewer oSUu in
  `assets/primary/feedback.md`].
- No confidence intervals on answer accuracy, and no bootstrap interval table
  for the headline answer matrix [source: PDF, Table 1 and reviewer oSUu in
  `assets/primary/feedback.md`].
- No total-pipeline token-cost table including first-stage query generation
  tokens [source: PDF, Table 3].
- No ablation passing the snap answer into the final answer call [source: PDF,
  Section 3.3 and reviewer oSUu in `assets/primary/feedback.md`].

## Bottom line for the meeting

What was submitted was a compact method paper: SCOPE as a two-call,
snap-conditioned pseudo-document retriever for legal RAG. Its best evidence
was the BarExamQA retrieval lift from `1.4%` raw Hit@5 to `9.5%`-`12.1%`
SCOPE Hit@5 across the submitted model rows [source: PDF, Abstract and
Table 11].

The fragile part was the framing. The paper treated raw-question RAG as the
main answer baseline, described HousingQA as parity, and presented
snap-conditioning as meaningful without proving SCOPE beat HyDE. Those are
the exact pressure points picked up in [[icml-ai4law-2026-rejection]].
