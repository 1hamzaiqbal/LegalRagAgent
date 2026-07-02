---
title: LegalBench (NeurIPS 2023)
type: source
tags: [legal-reasoning, benchmark, irac, legal-nlp, llm-evaluation]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2308.11462
local: references/guha2023legalbench.pdf
authors: Guha et al. (Guha, Nyarko, Ho, Re, + 36 legal/CS coauthors)
year: 2023
venue: NeurIPS 2023 Datasets and Benchmarks
code: https://github.com/HazyResearch/legalbench
---

# TL;DR

LegalBench is the legal-NLP community's canonical LLM benchmark: 162 tasks drawn from 36 corpora, designed and hand-crafted by ~40 legal professionals who are coauthors, and organized into a six-type taxonomy of legal reasoning (issue-spotting, rule-recall, rule-application, rule-conclusion, interpretation, rhetorical-understanding) explicitly derived from the IRAC framework lawyers use. It is the paper that gives lawyers and LLM developers "a common vocabulary," and it is precisely the vocabulary reviewer C1 accuses SCOPE of not speaking.

# Key claims / numbers

- 162 tasks from 36 distinct corpora; task breakdown by reasoning type: issue-spotting 16-17, rule-recall 5, rule-application 16, rule-conclusion 16, interpretation 118-119, rhetorical-understanding 10; formats: 35 MC, 112 binary classification, 8 multi-class, 7 open generation; average task size 563 samples. *our-relevance:* this is the taxonomy a revised SCOPE paper should use to say what our benchmarks measure, directly answering C1/C2.
- The typology is explicitly built on IRAC (Issue, Rule, Application, Conclusion), described as the common framework American lawyers use to execute legal reasoning; the paper walks a worked diversity-jurisdiction example through all four steps. *our-relevance:* C1's "practitioners' analytical processes" has an operational definition in the literature; SCOPE can be located inside it (see below) instead of ignoring it.
- Rule-recall tasks are singled out as "particularly useful for measuring hallucinations" because legal rules are jurisdiction-anchored and models must generate the rule text. *our-relevance:* SCOPE's pseudo-document p is functionally a rule-recall generation whose output is verified against a real corpus by retrieval; this reframing directly engages C3 and C4 in community terms.
- Evaluation of 20 LLMs from 11 families (3B-13B open models plus GPT-3.5/GPT-4/Claude-1). GPT-4 category averages: issue 82.9, rule-recall 59.2, rule-conclusion 89.9, interpretation 75.2, rhetorical 79.4; rule-application graded manually by a law-trained annotator on correctness/analysis (GPT-4: 82.2/79.7). *our-relevance:* rule-recall is the weakest GPT-4 category — the community's own numbers say parametric rule knowledge is the bottleneck, which is the gap retrieval (and SCOPE's generate-then-verify loop) is supposed to close; useful against C9's "answers are driven by parametric knowledge" as a framing, not a bug.
- Stated limitations: tasks have objectively correct answers only; skew toward American law and contracts; and "LegalBench evaluates IRAC abilities independently, while law exams and other legal work require... multi-hop" IRAC. *our-relevance:* SCALR (our LegalBench-SCALR benchmark) is one of the 10 rhetorical-understanding tasks, so our historical SCALR rows already sit inside this taxonomy without our paper ever saying so — an easy C2 fix.

# Bearing on the review

Grounds C1 and C2 most directly, and gives vocabulary for C3/C9. A revised paper must: (1) cite LegalBench and classify each of our four benchmarks in its taxonomy — BarExamQA is rule-recall + rule-application over fact patterns, HousingQA is jurisdiction-anchored rule-recall/entailment, SCALR is rhetorical-understanding (it is literally a LegalBench task), CaseHOLD is holding-selection; (2) state explicitly which IRAC step SCOPE intervenes on: the pseudo-document is a parametric *rule-recall* draft, and retrieval is a verification step that swaps the parametric rule for authoritative text before rule-application — this converts "HyDE applied to law" into an IRAC-legible mechanism claim; (3) use the GPT-4 rule-recall weakness (59.2) to motivate why rule-recall, not issue-spotting, is where a retrieval intervention belongs.

# Differentiation

We are not pre-empted: LegalBench measures reasoning, contains no retrieval component, and evaluates closed-book prompting only. But we are exposed on the reviewer's exact point — LegalBench proves the legal-NLP community had a lawyer-built framework and vocabulary in 2023, and our submission used none of it despite evaluating on one of its tasks (SCALR). Honest posture: SCOPE is a retrieval method whose evaluation should be *expressed in* this taxonomy, not a contribution to the taxonomy. We should also concede that LegalBench's "IRAC steps evaluated independently" limitation applies to us too: our answer-accuracy metric collapses rule-recall quality and rule-application quality into one number, which is exactly the [[answer-conversion-gap]] we keep observing.

# Links

[[scope]], [[legal-rag-benchmarks]], [[expert-judgment-replication]], [[vocabulary-gap]], [[answer-conversion-gap]], [[icml-ai4law-2026-rejection]]; siblings: [[jiang2023syllogism]], [[magesh2024hallucinationfree]].

# Raw source

references/guha2023legalbench.pdf (read pp. 1-14 of 143; task appendices F/G not read in detail).
