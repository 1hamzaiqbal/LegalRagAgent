---
title: Vocabulary Gap (colloquial ↔ statutory)
type: concept
tags: [retrieval, legal-ir, lexical-mismatch]
created: 2026-07-02
updated: 2026-07-02
status: maintained
---

# The vocabulary gap

**Definition.** The lexical/register mismatch between how users state legal
problems (narrative fact patterns: "crutch slipped on a banana peel") and how
authoritative corpora state answers (doctrinal abstractions: "foreseeable
intervening negligent act", "proximate cause"). The motivating premise of
[[scope]] and of legal passage retrieval generally.

**Why it matters.** It is *the* reason raw-question dense retrieval gets 1.4%
Hit@5 on BarExamQA while reaching 36.9% (state-filtered) on HousingQA whose
questions already speak statute. The gap defines the
[[weak-vs-strong-query-regime]] axis.

**Established names / prior art (post-review correction).** We are not the
first to target this: GuRE ([[gure]]) *trains* a generative query rewriter for
exactly this mismatch in legal passage retrieval (criticism C8); classic IR
calls the general phenomenon term mismatch, addressed by PRF/expansion; the
[[generated-query-family]] (HyDE/Query2doc/LameR/GAR) attacks it zero-shot.
Our measurement contribution is treating the gap as a *measurable, per-query
geometric quantity* (CE affinity of query vs gold/distractors) rather than a
dataset-level vibe — see [[geometry-vs-factuality]].

**Important negative**: per-query unigram perplexity and OOV rate do **not**
measure this gap in a useful way (corr ≈ 0 with expansion benefit); dataset-level
perplexity is only a weak regime separator (was still useful for the MedQA
pre-screen). "Vocabulary gap" as actually operative is *embedding-geometric*,
not surface-lexical — which is why we renamed the mechanism to affinity margin.

## Links
[[weak-vs-strong-query-regime]] · [[gure]] · [[generated-query-family]] ·
[[geometry-vs-factuality]] · [[scope]] · [[zheng-cslaw]]
