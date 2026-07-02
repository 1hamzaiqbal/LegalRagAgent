---
title: Legal Syllogism Prompting (ICAIL 2023)
type: source
tags: [legal-reasoning, prompting, syllogism, irac, judgment-prediction]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2307.08321
local: references/jiang2023syllogism.pdf
authors: Jiang & Yang (Cong Jiang, Xiaolei Yang; Peking University Law School)
year: 2023
venue: ICAIL 2023
code: https://github.com/jiangcongpku/Legal-Syllogism-Prompting
---

# TL;DR

The exemplar of "legal reasoning principles" prompting that reviewer C1 invokes: a zero-shot prompt that tells the model the legal syllogism structure (major premise = law, minor premise = facts, conclusion = judgment) and asks it to reason in that form for legal judgment prediction. On a sampled CAIL2018 Chinese criminal-charge set with GPT-3, syllogism prompting beats both direct prompting and zero-shot CoT, and — crucially for us — its error analysis shows the model fabricates law-article content and numbers, which the authors say should be fixed by "introducing external legal knowledge."

# Key claims / numbers

- Method (LoT): prepend "In the legal syllogism, the major premise is the law article, the minor premise is the facts of the case, and the conclusion is the judgment of case... use legal syllogism to think and output the judgment," reformulating LJP from classification to generation. Zero-shot, no exemplars. *our-relevance:* this is the concrete IRAC/syllogism prompting tradition C1 says we ignore; it operates on the answering call, whereas SCOPE operates on the retrieval query — orthogonal, and citable as such.
- On CAIL2018 (800 sampled cases, 100 each of 8 high-frequency charges), text-davinci-003: baseline 0.6450, zero-shot CoT 0.5875, LoT 0.6850 (micro accuracy). Zero-shot CoT is *below* baseline because its free-form intermediate steps "do not conform to legal reasoning." *our-relevance:* structure-of-reasoning interventions can beat generic CoT in law — supports the C1-facing claim that legal-process structure matters, and mirrors our finding that generic expansion (HyDE) is not automatically right for legal queries.
- On text-davinci-002 all methods are unstable (baseline 0.1313); smaller GPT models could not complete the task at all. *our-relevance:* the method is model-capability-gated, like our small-model SCOPE rows (Llama 8B), so any "incorporate legal reasoning structure" revision must be tested across our three model sizes, not assumed.
- Error analysis: the model outputs wrong major premises (e.g., "negligent causing serious injury constitutes intentional injury") and wrong statute numbers (cites Criminal Law Article 266 for robbery; the true article is 263); authors propose post-processing with external legal knowledge as future work. *our-relevance:* the fabricated-parametric-law failure mode is exactly C4's worry about SCOPE's pseudo-documents; the difference is SCOPE never shows generated law to the user or the answer call — p is only an embedding probe, and the answer is conditioned on retrieved authentic text. This paper lets us cite the risk from the legal-NLP tradition and then state the architectural mitigation.
- Discussion claims LoT gives selectivity (minor premise restricted to facts relevant to the recalled law) and sensitivity (aggregating acts under one charge), plus explainability via the explicit premise structure; notes IRAC as the "syllogism-like approach" in prior explainable legal-AI work. *our-relevance:* gives us the standard citation chain (syllogism -> IRAC -> practitioner process) for one paragraph of related work answering C1 without redoing our method.

# Bearing on the review

Grounds C1 (legal reasoning principles) and partially C3/C4. A revised paper must: (1) cite this line (LoT, IRAC prompting, chain-of-logic) as the reasoning-side tradition and position SCOPE as the retrieval-side complement — the syllogism's *major premise acquisition* step is where SCOPE lives: generate a candidate major premise parametrically, then use it to retrieve the authoritative one; (2) use their Article-266/263 fabrication finding to concede C4's premise (LLM-drafted law is unreliable) while showing our design point (p is discarded as content, kept only as a retrieval key; the answerer sees retrieved statute text); (3) note their CoT-below-baseline result as precedent that unstructured generation can hurt in law, connecting to our [[query-drift]] findings on strong queries.

# Differentiation

No overlap in mechanism: LoT is a zero-shot answer-side prompt with no retrieval, evaluated on Chinese criminal LJP; SCOPE is a two-call retrieval method on US bar/housing corpora with the generated text never surfaced. We are not pre-empted, but we also cannot claim to have "incorporated legal expertise" (C3) at their level: their prompt encodes an actual doctrinal form; our pseudo-document instruction ("write in the style of formal legal authority") is a stylistic prior, not a legal-theoretic one. The honest revision either adopts an explicit syllogism/IRAC-structured generation prompt as an ablation arm or scopes the paper as retrieval-mechanics work that deliberately does not model practitioner reasoning.

# Links

[[scope]], [[hyde]], [[generated-query-family]], [[query-drift]], [[vocabulary-gap]], [[answer-conversion-gap]], [[icml-ai4law-2026-rejection]]; siblings: [[guha2023legalbench]], [[magesh2024hallucinationfree]].

# Raw source

references/jiang2023syllogism.pdf (read in full, 5 pages).
