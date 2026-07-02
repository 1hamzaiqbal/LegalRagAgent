---
title: LegalBench-RAG (arXiv 2024) + Legal RAG Bench (Isaacus 2026)
type: source
tags: [legal-rag, benchmarks, retrieval-evaluation, error-decomposition, legal-nlp, span-retrieval]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2408.10343 ; https://arxiv.org/abs/2603.01710
local: references/legalbench-rag.pdf ; references/legal-rag-bench.pdf
authors: Pipitone et al. (ZeroEntropy); Butler et al. (Isaacus)
year: 2024; 2026
venue: arXiv (both; Legal RAG Bench dated 2 March 2026)
code: https://github.com/zeroentropy-cc/legalbenchrag ; https://github.com/isaacus-dev/legal-rag-bench
---

# Legal RAG benchmark landscape: LegalBench-RAG + Legal RAG Bench

## TL;DR
Two benchmarks that define what the legal-NLP community currently means by "evaluating legal RAG." **LegalBench-RAG** (Pipitone & Houir Alami, ZeroEntropy, Aug 2024) is the first retrieval-step benchmark for legal RAG: 6,858 expert-annotated queries traced from four LegalBench source datasets back to character-level spans in a 79.7M-char corpus, scored with span-level Precision/Recall@k. **Legal RAG Bench** (Butler & Butler, Isaacus, Mar 2026) is an end-to-end benchmark: 100 expert-crafted Victorian criminal-law questions over 4,876 Charge Book passages, scored on correctness/groundedness/retrieval-accuracy with a hierarchical error decomposition; it finds embedding-model choice (not LLM choice) dominates end-to-end legal RAG, and that most hallucinations are triggered by retrieval failures. Neither benchmark includes any generated-query (HyDE-family) baseline.

## Key claims / numbers

**LegalBench-RAG (2408.10343):**
- Measures the *retrieval step only*, at character-span granularity. 6,858 QA pairs (Table 1; Table 2 says 6,889 — internal inconsistency), 714 documents, 79,704,214 corpus characters, from PrivacyQA (194 q), CUAD (4,042), MAUD (1,676), ContractNLI (946). Human/expert annotations inherited from source datasets, plus manual QC. LegalBench-RAG-mini = 776 queries / 72 docs. *our-relevance:* this is exactly the "benchmarks the legal-NLP community recognizes" gap in C2 — SCOPE used BarExamQA/HousingQA and never engaged span-level legal retrieval benchmarks.
- Queries are templated: "Consider (document_description); (interrogative)" — the document description (GPT-4o-mini generated, manually inspected) routes to the right document, the interrogative comes from annotation-category mappings. *our-relevance:* these are NOT colloquial fact-pattern queries; the vocabulary gap is milder than BarExamQA's, so it probes a different point on our [[weak-vs-strong-query-regime]] axis.
- Baselines are pipeline ablations only: OpenAI text-embedding-3-large + SQLite Vec, naive 500-char chunking vs Recursive Character Text Splitter (RCTS), with/without Cohere rerank-english-v3.0. Best = RCTS + *no* reranker. All-dataset P@1: naive 2.40 vs RCTS 6.41. *our-relevance:* the general-purpose Cohere reranker *hurt* legal retrieval — direct precedent for questioning our ms-marco-MiniLM cross-encoder choice (C2/C8 adjacent).
- Difficulty spread: PrivacyQA easiest (RCTS P@1 14.38, R@64 84.19); MAUD hardest (P@1 2.65, R@64 28.28) — attributed to specialized M&A jargon. Absolute precision is low everywhere. *our-relevance:* low raw dense-retrieval numbers on specialized legal text corroborate our BarExamQA raw Hit@5 1.4% story ([[vocabulary-gap]]); a revised paper can cite MAUD as community evidence that legal jargon breaks general-purpose retrievers.
- No generated-query, query-rewrite, or HyDE baseline anywhere; limitation section concedes single-document answerability and no multi-hop. *our-relevance:* SCOPE could be dropped onto LegalBench-RAG(-mini) with zero harness changes to retrieval metrics — a concrete C2 remedy.

**Legal RAG Bench (2603.01710):**
- End-to-end: 100 hand-crafted expert questions + long-form answers + supporting passage (question-answer-evidence triplets) over 4,876 semchunk-chunked (≤512 legal-tokenizer tokens) passages from the Victorian Criminal Charge Book. *our-relevance:* this is what "reflecting practitioners' analytical processes" (C1) looks like operationally — realistic scenarios, expert-written long-form answers, criminal-procedure domain.
- Questions were deliberately written "as lexically dissimilar from relevant passages as possible in order to stress test the semantic understanding of evaluated models." *our-relevance:* a benchmark that *engineers* the vocabulary gap — the community-sanctioned version of our weak-query regime; strong citation for motivating [[scope]] on legitimate grounds.
- Full factorial 3 embedders × 2 LLMs (Kanon 2 Embedder, Gemini Embedding 001, OpenAI Text Embedding 3 Large × Gemini 3.1 Pro, GPT-5.2), GPT-5.2-as-judge (99% agreement on internal review). Embedder averages: Kanon 2 = 94.0 correct / 96.0 grounded / 86.0 retrieval acc; Text Emb 3 Large = 76.5 / 91.5 / 52.0; Gemini Emb 001 = 74.0 / 87.0 / 53.0. Kanon 2 lifts avg correctness +17.5pp, groundedness +4.5pp, retrieval accuracy +34pp. *our-relevance:* a domain-adapted embedder solves much of the same vocabulary gap SCOPE targets — an alternative (train the encoder) that a revised paper must position against, alongside GuRE's trained rewriter (C8).
- Hierarchical error decomposition: Hallucination (ungrounded) → Retrieval error (grounded, wrong, gold not retrieved) → Reasoning error (grounded, wrong, gold retrieved). Combined error: 8% (Kanon 2 × Gemini 3.1 Pro) up to 35% (Gemini Emb 001 × GPT-5.2). Poor retrieval raises hallucination (Kanon 2 = −6.75pp hallucinations vs general-purpose embedders); "retrieval sets the ceiling." *our-relevance:* directly grounds C4 — they measure fabrication risk instead of asserting a guardrail; also the mirror-image of our [[answer-conversion-gap]] (see Differentiation).
- Statistical rigor: linear probability model with question fixed effects, cluster-robust SEs, ANOVA-style Wald tests, and explicit embedder×LLM interaction tests. Embedder main effect p<0.001 for correctness; LLM main effect NS for correctness (p=0.499). *our-relevance:* this is the significance-testing standard C7/C11 demanded of us (we reported SCOPE-vs-HyDE Hit@5 deltas of +0.5–1.2pp with no test at all).
- Related-work section explicitly attacks LegalBench and LegalBench-RAG as "low-value, relatively trivial text classification" tasks, and criticizes Zheng et al.'s BarExamQA/HousingQA for being closed-ended MC/yes-no — unable to surface hallucinated-but-correct answers — quoting the authors' own limitation. *our-relevance:* the community is already litigating our exact benchmark choices (C2); note the Isaacus conflict of interest (they built Kanon 2 Embedder and sponsored both this bench and MLEB).

## Bearing on the review
- **C2 (benchmarks/baselines don't reflect legal-NLP prior work):** these two papers *are* the landscape a revised paper must cite and ideally run on. Minimum fix: cite both plus MLEB; strong fix: report SCOPE vs raw vs HyDE on LegalBench-RAG-mini (776 queries, span-level P/R@k — retrieval-only, cheap) and, if feasible, on Legal RAG Bench (100 questions end-to-end with groundedness).
- **C7/C11 (no significance tests, no CIs):** Legal RAG Bench's fixed-effects LPM + clustered SEs + interaction Wald tests is the concrete template; adopt it (or McNemar + CIs at minimum) for every SCOPE-vs-HyDE and SCOPE-vs-raw contrast.
- **C4 (fabricated legal content risk):** Legal RAG Bench operationalizes hallucination as ungroundedness and *measures* it per pipeline. A revised SCOPE paper should score pseudo-document factuality/groundedness (we already have the [[geometry-vs-factuality]] machinery) rather than hand-waving the discarded-a0 guardrail; note their finding that retrieval failures *cause* hallucinations reframes C4 productively — better retrieval via SCOPE could plausibly *reduce* downstream fabrication, which is testable.
- **C9/C10 (framing against weak baselines):** their error decomposition (hallucination vs retrieval vs reasoning error) is a cleaner way to report where SCOPE helps than raw accuracy deltas; it would make the BarExamQA "retrieval up 8x, answers flat" honest by construction.
- **C1 (practitioner realism):** Legal RAG Bench's expert-crafted long-form QA is the community's answer; citing it and acknowledging MC-format limitations (which its authors, quoting Zheng et al., already flagged) defuses the criticism better than defending MC evals.

## Differentiation
- Neither benchmark tests generated-query methods: both vary embedder / chunking / reranker / LLM with the raw query fixed. SCOPE (and HyDE-family expansion generally) is orthogonal to and unexamined in both — we are not pre-empted on method, only under-benchmarked (C2). Running SCOPE on these suites is an open, low-cost differentiator.
- Legal RAG Bench's headline ("retrieval sets the ceiling"; +34pp retrieval → +17.5pp correctness) appears to conflict with our answer-conversion wall (8x retrieval lift → 72.3→72.9 answers). The settings differ: they use frontier LLMs on long-form expert questions engineered to be lexically dissimilar (retrieval genuinely load-bearing); we used small/mid open models on MC tasks where parametric knowledge often suffices and options anchor the answer. Honest read: our answer-flat result is partly a property of MC benchmarks + parametric-knowledge-saturated models, exactly the format criticism Legal RAG Bench levels at BarExamQA/HousingQA. A revised paper should say this rather than defend the wall as universal.
- Their solutions to the vocabulary gap are *trained* components (Kanon 2 domain embedder; cf. GuRE's trained rewriter per C8); SCOPE is zero-training, inference-time query-side expansion. That is our niche claim — but it now needs a head-to-head or at least an explicit cost/deployment argument against domain-adapted embedders.
- LegalBench-RAG's span-level granularity is stricter than our passage-ID Hit@k; adopting span or at least passage-level P/R with their k-sweep (1–64) would strengthen the retrieval-exposure story.
- Conflict-of-interest note for citation hygiene: Legal RAG Bench is authored and sponsored by Isaacus, whose Kanon 2 Embedder is the benchmark's winning model; its harsh assessment of LegalBench-RAG/BarExamQA should be cited as a position, not settled consensus.

## Links
- [[scope]], [[hyde]], [[generated-query-family]] — methods absent from both benchmarks' baseline sets
- [[vocabulary-gap]], [[weak-vs-strong-query-regime]] — Legal RAG Bench engineers lexical dissimilarity; MAUD shows jargon-driven retrieval collapse
- [[answer-conversion-gap]] — tension with their "retrieval sets the ceiling" finding (format + model-strength dependent)
- [[geometry-vs-factuality]] — our machinery for the groundedness/hallucination measurement they demand (C4)
- [[qpp]], [[query-drift]], [[regime-routing]] — where routing could be tested on span-level legal retrieval
- [[icml-ai4law-2026-rejection]] — grounds C1, C2, C4, C7, C9, C10, C11
- Sibling sources: [[gure]], [[koblex-parser]], [[guha2023legalbench]], [[zheng-cslaw]], [[magesh2024hallucinationfree]], [[scope-paper-2026]]

## Raw source
- `references/legalbench-rag.pdf` (arXiv 2408.10343v1, 12 pp., read in full)
- `references/legal-rag-bench.pdf` (arXiv 2603.01710v1, 13 pp., read in full)
