---
title: Overlapping Related Work Dossier
type: meeting-doc
tags: [meeting, related-work, scope, ai4law, rejection]
created: 2026-07-02
date: 2026-07-02
---

# Overlapping related work dossier

Grounding note: this page is based only on the official reviews in
`official_paper_and_review_icml_ai_4_law/feedback.md`, the extracted submitted
PDF text from `official_paper_and_review_icml_ai_4_law/97_SCOPE_When_Generated_Legal_.pdf`,
the existing [[wiki/sources]] pages, and
`paper/submission/RELATED_WORK_GROUNDING.md`. I did not download new PDFs for
this pass. "Submission cited?" means I checked the submitted PDF text; absence
is reported as "not found in extracted PDF text" rather than as a bibliographic
proof.

Canonical links: [[scope-paper-2026]], [[icml-ai4law-2026-rejection]],
[[thesis-v2]], [[generated-query-family]], [[vocabulary-gap]],
[[weak-vs-strong-query-regime]], [[query-drift]], [[qpp]],
[[answer-conversion-gap]].

## What the reviewers pointed at

| Work or family | How the reviews pointed at it | One-line description | Submission cited? | Immediate implication |
|---|---|---|---|---|
| [[koblex-parser]] / ParSeR | Explicit in Reviewer oSUu W1 and references | KoBLEX introduces ParSeR: generate statute-style "parametric provisions" from LLM knowledge, retrieve, rerank, select provisions, then answer Korean multi-hop legal QA. | **No.** No KoBLEX/ParSeR hit found in the extracted submission text. | Closest uncited near-twin to SCOPE's legal pseudo-document retrieval. |
| [[gure]] | Explicit in Reviewer oSUu W2 and references | GuRE fine-tunes a legal LLM to generate the target legal passage from drafting context and uses it as the retrieval query; it directly targets legal vocabulary mismatch. | **No.** No GuRE or "Generative Query REwriter" hit found. | The obvious supervised legal-rewriter skyline was absent. |
| [[hyde]] | Explicit in both reviews and oSUu references | HyDE generates a hypothetical document from the question and embeds that generated text for zero-shot retrieval. | **Yes.** HyDE and Gao et al. appear throughout the method, contrast, and references. | Citation was present; the problem was claiming novelty without proving the SCOPE-specific delta. |
| [[zheng-cslaw]] | Explicit in oSUu summary and references | Source paper for BarExamQA/HousingQA; quantifies legal query-gold lexical gaps and already runs legal structured generative expansion on these benchmarks. | **Yes.** Zheng et al. and CSLAW '25 appear in dataset and references. | Cited as benchmark source, but not engaged as prior expansion, ceiling, and conversion-gap evidence. |
| Legal query rewriters | Implied by "any legal generated-query method" and explicit "legal query rewriter such as GuRE" | Supervised or trained methods that rewrite legal queries/passages rather than prompt zero-shot pseudo-documents. | **Partly no.** GuRE not found; `rag_rewrite` appears only as our supplemental mode. | Need a zero-shot-vs-supervised framing and at least a scoped skyline experiment. |
| [[query2doc]] and generic LLM QE | Implied by "essentially HyDE" and "generated-query method of this kind" | Query2Doc concatenates generated pseudo-documents with the original query; its ablation says pseudo-doc-only can be worse than raw query on sparse retrieval. | **Yes for Query2Doc.** It appears in related work and references. | Citation was too thin; keep-vs-discard needed to be an ablation, not an assumption. |
| [[lamer-gar]] answer-conditioned expansion | Implied by the snap-answer novelty and guardrail critique | GAR and LameR generate answers or answer-bearing text to improve retrieval, usually keeping the original query. | **No.** I found no LameR/GAR hits in the extracted PDF text. | Snap-answer conditioning was not a new primitive; only private-use-and-discard remains distinctive. |
| [[csqe]] corpus-grounded QE | Implied by the fabricated-pseudo-document concern | CSQE expands queries with real sentences extracted from first-pass retrieved corpus text plus smaller parametric expansion. | **No.** No CSQE/Corpus-Steered hit found. | Strongest "avoid hallucinated expansion" baseline family was absent. |
| [[weller-drift]] | Implied by HousingQA regression, strong-query harm, and "marginal gains" | Weller et al. show generative expansion helps weak retrievers/format-shift regimes and hurts strong ones by blurring relevance signals. | **No.** No Weller/query-drift hit found. | The reviewer-read "parity" problem should have been framed as known expansion drift. |
| [[qe-survey-2025]] | Implied by missing IR taxonomy for QE/gating | Survey names zero-grounding generative QE, grounding-only QE, learning/alignment QE, and states selective use on hard queries as deployment guidance. | **No.** No survey hit found. | Our regime law should be positioned as measuring a known selective-QE prescription. |
| [[yoon2025leakage]] | Implied by "LLMs may inherit limitations" and parametric-knowledge concerns | Tests whether HyDE/Query2Doc gains come from generated text leaking benchmark gold evidence. | **No.** No Yoon/leakage hit found. | Forces a leakage audit before BarExamQA gains can be trusted as non-memorization. |
| [[reuter2025sac]] | Implied by legal retrieval mismatch and chunk/corpus-side failure | Defines document-level retrieval mismatch in legal RAG and fixes part of it with summary-augmented chunking. | **No.** No Reuter/SAC hit found. | Shows an indexing-side alternative to query-side generation. |
| [[legal-rag-benchmarks-src]] | Implied by "benchmarks/baselines do not reflect legal NLP" | LegalBench-RAG and Legal RAG Bench define span-level and end-to-end legal RAG evaluation norms. | **Yes.** Pipitone & Alami and Butler & Butler appear in related work/references. | Cited, but not run or used as evaluation-design constraints. |
| [[magesh2024hallucinationfree]] | Implied by practitioner-process and fabricated-legal-content critiques | Lawyer-graded commercial legal RAG evaluation with correctness-groundedness typology. | **Yes.** Magesh et al. appears in related work/references. | Cited, but not used to audit pseudo-documents or final groundedness. |

## The dangerous overlaps, ranked

### 1. [[koblex-parser]] / ParSeR

What it actually does: KoBLEX builds a bilingual Korean legal QA benchmark and
introduces ParSeR. The retrieval method prompts the LLM to generate one or more
statute-style "parametric provisions" from its parametric knowledge, then runs
Retrieve -> Rerank -> LLM Selection before final answer generation. The source
page reports large provision-retrieval and answer-quality gains, including
retrieval F1 59.41 versus 21.50 for one-time retrieval under GPT-4o.

Claim-level overlap: this directly overlaps the SCOPE claim "generate formal
legal authority-like text from the model and use it as the retrieval handle."
The reviewer mapping from parametric provision to pseudo-document is basically
right. The selection stage even makes their pipeline more legal-RAG-specific
than ours.

What still differentiates us: SCOPE uses a private snap answer to condition a
single pseudo-document, discards that answer, and targets US BarExamQA/HousingQA
rather than Korean civil-law multi-hop QA. That is a narrow distinction, and it
is not yet a positive contribution because the submitted SCOPE-vs-HyDE deltas
were small and untested.

What we must cite/engage: cite KoBLEX as closest prior art, not as background.
State clearly that pseudo-statute/provision generation is not ours. Compare
SCOPE to a ParSeR-style conclusion-banned provision prompt and report whether
snap conditioning adds anything.

Experiment forced: a C12-style ablation: no-snap statute-style passage,
snap-conditioned passage, conclusion-banned ParSeR-style passage, and final
answer with/without `a0`. Report retrieval and answer outcomes.

### 2. [[gure]]

What it actually does: GuRE trains SaulLM-7B with LoRA to generate a cited legal
passage from legal drafting context, then uses the generated passage as a
replacement query. On LePaRD, the source page reports BM25 nDCG@10 improving
from 15.33 to 47.69, with paired t-test significance, and shows a long-tail
citation-frequency analysis of target passages.

Claim-level overlap: GuRE owns the legal vocabulary-mismatch motivation in a
legal-IR venue. It is the supervised/trained sibling of SCOPE's zero-shot
pseudo-document generation, and it directly attacks the same "query language
does not match legal passage language" premise.

What still differentiates us: SCOPE is training-free and question-answering
end-to-end; GuRE is retrieval-only and requires supervised passage-generation
pairs. But "training-free" is not a strong claim unless we show settings where
supervision is unavailable or where zero-shot expansion gives useful diagnostic
mechanism beyond performance.

What we must cite/engage: cite GuRE as the supervised legal rewriter skyline.
Report passage-frequency or jurisdiction-stratified analyses in our corpora,
because GuRE makes distributional characterization part of the legal-IR story.

Experiment forced: run a GuRE-style supervised skyline on BarExamQA question
and gold-passage pairs if enough train pairs exist, or explicitly scope it as a
future supervised skyline with an honest data-volume constraint. At minimum,
run a trained/learned rewriter baseline on the same retrieval metric.

### 3. [[zheng-cslaw]]

What it actually does: Zheng et al. introduce BarExamQA and HousingQA. The
source page says they quantify extreme query-gold lexical gaps, report weak raw
retrieval, run GPT-3.5 generative expansion variants including structured legal
reasoning, and warn that retrieval gains translate weakly to answer gains.
They also show rollout-as-pseudo-passage can hurt BarExamQA answers.

Claim-level overlap: this is not merely the dataset citation. It already
contains the benchmark-native vocabulary-gap analysis and a generated legal
reasoning expansion baseline. It also contains the gold-ceiling/conversion
caveat that inverts our submitted C5/C9 framing.

What still differentiates us: same-model generator/answerer, pseudo-doc-only
embedding, k=5 answer pipelines, cross-encoder reranking, and later mechanism
work on affinity margin. The submitted paper did not make these differences
carry the claim.

What we must cite/engage: cite as prior work on legal generative expansion and
retrieval-to-answer limits, not only as "benchmark source." Use their lexical
similarity and bootstrap/CI norms as the reporting baseline.

Experiment forced: reproduce their structured legal reasoning rollout as a
baseline on our exact BarExamQA/HousingQA retrieval stack, then test SCOPE-vs-
HyDE-vs-structured-rollout with paired retrieval and answer tests.

### 4. [[lexpath2026]]

What it actually does: LexPath is a 2026 Chinese legal article retrieval system.
It combines IRAC-guided sparse expansion, structure-guided dense retrieval with
hierarchy/citation hard negatives, and intent-consistency reranking. Its source
page reports IRAC-Exp beating HyDE and Query2Doc on three Chinese legal
benchmarks, while downstream QA still shows answer-conversion failures for
frontier models on LexRAG.

Claim-level overlap: LexPath is the strongest current legal expansion rival:
it uses legal reasoning structure rather than generic pseudo-document style,
and it tests directly against HyDE/Query2Doc. It also reproduces the
retrieval-up/answer-not-always-up pattern that we want to make central.

What still differentiates us: LexPath is an always-on engineering system for
Chinese statute/article retrieval. It does not offer a per-query affinity
mechanism, leakage/factuality falsification, or a trained selector based on
free outcome labels.

What we must cite/engage: cite it in any July-forward related work, but note it
appeared after the May 22 submission date, so it was not a fair strict
submission-bibliography absence. It is still dangerous for any revision.

Experiment forced: run an IRAC-structured SCOPE prompt variant on BarExamQA.
If IRAC beats generic SCOPE, the method story shifts further from SCOPE to
"legal-structured expansion as one instrument."

### 5. [[weller-drift]] and [[qe-survey-2025]]

What they actually do: Weller et al. show a broad macro law: generative query
and document expansion helps weak retrievers/format-shift settings and hurts
strong ones by blurring relevance. The 2025 QE survey provides the taxonomy:
SCOPE is zero-grounding, non-interactive generative QE; deployment guidance is
to use such methods selectively on hard or underspecified queries.

Claim-level overlap: these works own the regime intuition. HousingQA regression
and BarExamQA lift should not be framed as surprising; they are exactly the
weak-query/strong-query pattern predicted by this literature.

What still differentiates us: Weller is retrieval-only, macro, and label-hungry;
the survey is a map, not an experiment. Our white space is a quantitative,
per-query affinity-margin account plus legal/medical evidence and answer
conversion modeling.

What we must cite/engage: stop calling the regime split a discovery. Cite the
macro law and the taxonomy, then claim our measured mechanism and failures of
no-gold QPP as the contribution.

Experiment forced: report expansion gain as a function of raw-query margin and
baseline retrieval strength. Include query drift examples and paired tests for
SCOPE-vs-HyDE on strong-query settings.

### 6. [[query2doc]] plus [[lamer-gar]]

What they actually do: Query2Doc generates pseudo-documents and concatenates
them with the original query; its source page highlights a keep-vs-discard
ablation where pseudo-doc-only retrieval is worse than query+pseudo-doc and
even worse than raw query in sparse retrieval. GAR and LameR make answer-
conditioned expansion explicit: generate answers or answer-bearing passages,
usually while retaining the original query.

Claim-level overlap: these works pre-empt both "generated pseudo-documents help
retrieval" and "answer attempts can condition expansion." They also attack the
submitted design choice of discarding the raw question during retrieval.

What still differentiates us: SCOPE's latent answer-conditioning and discard
policy may be useful in weak legal regimes where the raw query is lexically
misleading. But this is a regime-conditional reversal of Query2Doc/GAR, not a
general method novelty claim.

What we must cite/engage: cite the keep-query family and answer-conditioned
family. The generated-query taxonomy should lead the section before SCOPE is
introduced.

Experiment forced: a 2x2 retrieval ablation: raw query, pseudo-doc only,
query+pseudo-doc concat/pool, and answer-conditioned variants. Report where
weak-query BarExamQA reverses Query2Doc's keep-query result, if it really does.

### 7. [[yoon2025leakage]]

What it actually does: Yoon et al. test whether HyDE/Query2Doc gains on fact
verification come from generated documents that NLI-match gold evidence, i.e.
knowledge leakage from public benchmarks. The source page reports that gains
concentrate on matched claims and that unmatched claims often drop below the
no-expansion baseline.

Claim-level overlap: this is the strongest rival explanation for our BarExamQA
retrieval lift. A reviewer can say generated legal pseudo-documents retrieve
well because the LLM memorized bar-exam or benchmark-adjacent gold evidence.

What still differentiates us: our post-rejection leakage audit, linked from
[[thesis-v2]], reportedly rejects leakage as the sole explanation for BarExamQA.
That was not in the submitted paper and must be front-and-center if the result
is reused.

What we must cite/engage: cite the leakage paper whenever claiming generated
query retrieval gains on public benchmarks. Do not claim non-leakage without an
audit.

Experiment forced: NLI-match generated pseudo-documents against gold passages,
then report Hit@k separately for matched and unmatched generations for SCOPE,
HyDE, Query2Doc, and structured legal prompts.

### 8. [[reuter2025sac]]

What it actually does: Reuter et al. define Document-Level Retrieval Mismatch
in legal RAG and show that summary-augmented chunking can roughly halve wrong-
document retrieval in boilerplate-heavy legal corpora. The intervention is
indexing-side: add short document-level summaries to chunks before embedding.

Claim-level overlap: it attacks the same semantic/context mismatch from the
corpus side rather than the query side. It makes "legal retrieval fails because
chunks are confusable" a published quantity, not just our intuition.

What still differentiates us: SCOPE is online query-side generation and can be
applied without re-indexing. Reuter is retrieval-only and does not ask when
query expansion helps or how answer accuracy converts.

What we must cite/engage: cite it as a complementary legal retrieval-mismatch
diagnostic and as a competing deployment path when index rebuilds are feasible.

Experiment forced: audit DRM-style wrong-document/wrong-jurisdiction retrieval
in BarExamQA/HousingQA, and test whether query-side expansion still helps on a
summary-augmented or document-context-enriched index. This interaction is
currently **UNVERIFIED**.

## Coverage map

| Thesis-v2 piece | Prior work that touches it | What prior work already owns | White space that survives |
|---|---|---|---|
| P1: margin-governed expansion | [[weller-drift]], [[qe-survey-2025]], [[hyde]], [[query2doc]], [[csqe]], [[gure]], [[zheng-cslaw]], [[lexpath2026]], [[yoon2025leakage]] | Generated expansion is a known family; expansion helps weak/format-shift regimes and can drift; legal vocabulary mismatch is already measured; trained rewriters and IRAC expansion are stronger legal baselines; leakage is a known confound. | A quantitative per-query gold-affinity/margin mechanism, legal plus non-legal replication, and falsification that factuality/leakage alone explains the effect. |
| P2: snap-vs-HyDE drift damping | [[hyde]], [[lamer-gar]], [[koblex-parser]], [[weller-drift]], [[lexpath2026]], [[query2doc]] | HyDE is the parent method; answer-conditioned expansion exists; ParSeR generates legal provision-like text; IRAC expansion can beat HyDE; keep-vs-discard is already a live axis. | Snap conditioning as drift dampener on strong queries, not as weak-query retrieval booster. Still needs C12 ablations to show whether the snap answer, frame commitment, or prompt style is the active ingredient. |
| P3: trained-judge selection | [[koblex-parser]], [[lexpath2026]], [[reuter2025sac]], [[thinking-machines-expert-judgment]], [[magesh2024hallucinationfree]], [[legal-rag-benchmarks-src]] | ParSeR has an LLM selection stage; LexPath has intent-consistency reranking; Reuter names reranking as future work; practitioner and benchmark papers show legal relevance/groundedness needs expert judgment; Thinking Machines argues expert labels beat prompts. | Training a small legal evidence selector on free outcome labels, measuring pool confusability, and showing selector quality rather than expansion is the bottleneck. Needs lawyer-label validation before legal-practice claims. |
| Conversion law: exposure-to-answer | [[power-noise-lostmiddle]], [[emami-qpp-variant]], [[tian-right-track]], [[zheng-cslaw]], [[koblex-parser]], [[lexpath2026]], [[magesh2024hallucinationfree]], [[legal-rag-benchmarks-src]] | Distractor harm and position effects are known; QPP can correlate with retrieval but not answer quality; Zheng already warns retrieval gains have limited answer ceiling; legal RAG benchmarks evaluate groundedness and error decomposition. | A task/reader-deficit law for when better evidence converts, with break-even Hit@k estimates and explicit separation of retrieval, selection, and answer use. This is the least-owned part if measured rigorously. |

## Meeting implications

1. The old novelty sentence is untenable: "SCOPE generates formal legal
   pseudo-documents for retrieval" belongs to the generated-query family and
   has legal near-twins.
2. The defensible revision is not a stronger SCOPE sales pitch. It is:
   expansion margin, drift damping, selector bottleneck, and conversion law.
3. Legal venues will expect KoBLEX, GuRE, Zheng, LegalBench/LegalBench-RAG,
   Magesh, and practitioner/IRAC vocabulary. IR venues will expect HyDE,
   Query2Doc, CSQE, Weller, QPP, and leakage.
4. Any "legal reasoning" claim must be tied to an actual legal construct:
   IRAC/syllogism prompting, statute hierarchy, jurisdiction filters, authority
   relevance, or lawyer-graded groundedness. "Formal legal style" is not enough.
5. Every table needs the comparator that matches the claim: LLM-only for answer
   value, raw retrieval for query policy, HyDE/Query2Doc/ParSeR-style for
   generated-query novelty, GuRE/LegalMALR/LexPath for legal retrieval skyline,
   and gold/neighbor controls for conversion.

## The three absences that most likely triggered the strong rejects

1. **[[koblex-parser]] / ParSeR.** Strictly uncited, explicitly named by the
   reviewer, and closest to the submitted mechanism.
2. **[[gure]].** Strictly uncited, explicitly named by the reviewer, and owns
   the legal vocabulary-mismatch/trained-rewriter lane.
3. **[[zheng-cslaw]] as substantive prior work.** This was cited, so the failure
   was engagement rather than bibliography. But the submission did not use its
   structured legal expansion, lexical-gap measurement, bootstrap norms, or
   retrieval-to-answer ceiling as prior work; that omission made our benchmark
   story look self-contained when it was not.

Strict uncited-only answer: KoBLEX/ParSeR and GuRE are the two clear misses.
The third most damaging issue was a cited-but-not-engaged source, Zheng et al.

## Source-page expansions in this pass

None. The existing source pages were sufficient for this meeting dossier.
