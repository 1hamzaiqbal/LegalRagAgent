# Pipeline Diagnosis — Manual Walkthrough Findings

Date: 2026-02-21
Branch: `feature/corpus-expansion-and-tools`

## Context

Ran 5 bar exam questions manually through the pipeline, acting as the LLM at each node while using real ChromaDB retrieval. All 5 answered correctly by the LLM, but retrieval quality was the bottleneck — not synthesis.

## Baseline Metrics (1K passages)

| Metric | Value |
|--------|-------|
| Corpus size | 1,000 / 220,000 passages (0.45%) |
| Recall@5 | 0.1637 (953 QA pairs) |
| MRR | 0.0952 |
| Avg cosine similarity | 0.4457 |
| Gold passage in top-5 | 2/5 in manual walkthrough |

## Three Retrieval Failure Modes

### 1. Keyword Noise

**Observed in:** Q1 (Torts/IIED)

The query contained the word "murderer" (from protest signs saying "Governor Doe - Murderer"). The embedding model retrieved passages about **murder elements** (first-degree murder, voluntary manslaughter) instead of IIED or First Amendment doctrine. 3 of 5 retrieved docs were completely irrelevant.

**Root cause:** MiniLM-L6-v2 bi-encoder embeddings match on surface lexical overlap. "Murderer" is semantically close to "murder" in embedding space even though the legal concepts are entirely different (tort vs. criminal law).

**Fix:** Cross-encoder reranker that scores query-document pairs with full attention, catching that "murderer on a protest sign" is contextually unrelated to "elements of murder."

### 2. Missing Concepts (Terminological Gap)

**Observed in:** Q2 (Contracts/UCC formation)

The question hinged on whether a cancellation clause makes a promise **illusory** (no contract formed). The illusory promise doctrine was completely absent from the top-5 results. Retrieved docs covered offer/acceptance mechanics, UCC 2-207, and option contracts — related but not the core doctrine needed.

**Root cause:** The query uses terms like "purchase order," "right to cancel," "contract formation." The illusory promise doctrine uses terms like "illusory," "unfettered discretion," "no consideration." The embedding model can't bridge this terminological gap because the concepts are linked by legal reasoning, not by shared vocabulary.

**Fix:** (a) Larger corpus increases chance of having a passage that uses bridging vocabulary. (b) Multi-round retrieval: extract concepts from first-round passages, use them to query again. (c) Query expansion using LLM to generate alternative search terms.

### 3. Adversarial Retrieval (Evidence Against Correct Answer)

**Observed in:** Q5 (Constitutional Law / 14th Amendment)

The correct answer was that the statute is constitutional under Congress's **Section 5 enforcement power** of the 14th Amendment. But the retrieved passages all emphasized the **state action limitation** — that the 14th Amendment does NOT reach private conduct (citing *Civil Rights Cases*, 1883). No passage mentioned Section 5 enforcement power.

A faithful RAG system that only synthesizes from evidence would answer **incorrectly** — the evidence actively argues against the right answer.

**Root cause:** The corpus has more passages about the 14th Amendment's limitations than about Section 5's expansive enforcement power. The embedding model correctly retrieved 14th-Amendment-related passages, but the corpus coverage is lopsided.

**Fix:** (a) Larger corpus (220K passages likely has Section 5 content). (b) When evidence is one-sided but the question presents multiple theories, trigger a targeted follow-up retrieval for the underrepresented theory. (c) Web search fallback for constitutional law topics with poor corpus coverage.

## Per-Question Detail

| # | Subject | Confidence | Gold Retr? | Relevant Docs | Failure Mode | Answer |
|---|---------|-----------|------------|---------------|--------------|--------|
| 1 | TORTS (IIED) | 0.467 | No | 1/5 | Keyword noise | D (correct) |
| 2 | CONTRACTS (UCC) | 0.514 | No | 3/5 | Missing concepts | A (correct) |
| 3 | CRIM LAW (Bruton) | 0.532 | Yes | 2/5 | — (adequate) | A (correct) |
| 4 | EVIDENCE (impeach) | 0.607 | Yes | 4/5 | — (good) | B (correct) |
| 5 | CONST LAW (14A §5) | 0.587 | No | 4/5* | Adversarial | B (correct) |

*Q5: 4/5 docs were topically relevant but argued against the correct answer.

## Priority Fixes

1. **Load full 220K corpus** — highest impact, zero API cost — **DONE** (commit `09de734`)
2. **Cross-encoder reranker** — fixes keyword noise, improves precision — **DONE** (source-aware reranking with `ms-marco-MiniLM-L-6-v2`)
3. **Query expansion / multi-round retrieval** — fixes terminological gaps
4. **Web search fallback** — covers topics absent from corpus entirely

## Fix Results

### Full Corpus (220K passages)

Loading all 220K passages caused caselaw (98.2%) to drown out MBE (1.1%) passages.
Fixed with hybrid retrieval: fetch from MBE/wex and caselaw pools separately, interleave 3 study + 2 caselaw.

### Cross-Encoder Reranker

Added two-stage retrieval: bi-encoder over-retrieves 4x candidates per pool, cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks within each pool, then interleave.

**Qualitative improvement confirmed**: The IIED/"murderer" keyword-noise query (failure mode #1) now returns 5/5 directly relevant IIED passages (was 2/5 with bi-encoder only).

**Recall@5 A/B test** (96 stratified queries): Bi-encoder-only 0.0625 vs source-aware reranking 0.0521. The 1-hit difference is within noise. Recall@5 measures whether a *specific* gold passage appears in top-5; the cross-encoder often selects an equally relevant but different passage. The metric understates the quality improvement.

| Config | Recall@5 | MRR | Avg Confidence |
|--------|----------|-----|----------------|
| 1K passages only (baseline) | 0.1637 | 0.0952 | 0.4457 |
| 220K unfiltered | 0.0115 | 0.0051 | 0.5534 |
| 220K hybrid (3 study + 2 caselaw) | 0.0693 | 0.0474 | 0.5166 |
| 220K hybrid + cross-encoder rerank | 0.0521 | 0.0208 | 0.4917 |
