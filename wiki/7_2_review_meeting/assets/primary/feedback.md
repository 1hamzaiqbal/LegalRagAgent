SCOPE: When Generated Legal Queries Help Legal RAG
Download PDF
Hamza Iqbal, Hanxun Li, Mingzheng Li, Langlin Huang, Jiaxin Huang
22 May 2026 (modified: 17 Jun 2026)
Submitted to AI4Law
AI4Law, Authors
Revisions
BibTeX
CC BY 4.0
Keywords: retrieval-augmented generation, legal RAG, HyDE, generated queries, legal evaluation
TL;DR: Retrieval with hypothetical legal passages leads to increased retrieval and final accuracy on legal benchmarks
Abstract:
Retrieval-Augmented Generation (RAG) is an effective approach for legal question answering, yet standard RAG struggles with the lexical gap between colloquial fact patterns and formal statutory corpora. While retrieving with pseudo-documents (e.g., HyDE) proves successful in general-domain RAG, it fails in legal contexts due to this lexical mismatch. To address this, we introduce SCOPE (Snap-answer COnditioned Pseudo-document Embedding), which first generates a direct ”snap answer” reflecting immediate legal intuition. A pseudo-document is then generated in the style of formal legal authority to support this snap answer. Crucially, only the pseudo-document is embedded for retrieval. During the final generation phase, the snap answer and pseudo-document are discarded, ensuring the ultimate response is grounded strictly in the retrieved evidence. We evaluate SCOPE across two extremes of the question-corpus lexical gap. On BarExamQA, with low lexical similarity, SCOPE significantly bridges this gap, lifting gold passage retrieval (Hit@5) from a 1.4% baseline to 9.5%–12.1% across 8B, 26B, and 70B parameter models, while consistently improving final answer accuracy. These results demonstrate that SCOPE is a robust and tailored solution for bridging the colloquial-to-statutory vocabulary gap in complex Legal RAG challenges.

Submission Number: 97
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
3 / 3 replies shown
Add:
Paper Decision
Decisionby Program Chairs12 Jun 2026, 08:51 (modified: 19 Jun 2026, 18:12)Program Chairs, Reviewers, AuthorsRevisions
Decision: Reject
SCOPE: When Generated Legal Queries Help Legal RAG
Official Reviewby Reviewer 7gGP11 Jun 2026, 07:08 (modified: 14 Jun 2026, 16:39)Program Chairs, Reviewers, AuthorsRevisions
Review:
The proposed approach focuses primarily on retrieval optimization techniques rather than legal reasoning principles or legal practitioners' actual analytical processes.
The literature review is insufficient for a Legal NLP paper. Neither the benchmarks they used nor the baselines reflect prior research in the legal NLP community. The method is essentially an application of HyDE to the legal domain. However, it does not convincingly incorporate legal expertise, legal reasoning patterns, or practitioner perspectives.
The paper highlights the limitations of LLMs in legal tasks, yet relies on the LLMs to generate hypothetical documents. This creates a inconsistency at the core of the approach.
The reported performance gains are marginal. Given the difficulty of generating reliable and legally grounded hypothetical documents from short queries, the effectiveness of HyDE in this setting remains questionable. The generated hypothetical documents themselves appear likely to inherit the limitations of the underlying LLM, including superficial reasoning and potentially unsupported or fabricated legal content. The observed improvements appear limited to minor variations.
Rating: 2: Strong rejection
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
Official Review from Reviewer oSUu
Official Reviewby Reviewer oSUu02 Jun 2026, 02:59 (modified: 14 Jun 2026, 16:39)Program Chairs, Reviewers, AuthorsRevisions
Review:
Summary
SCOPE is a two-call retrieval-augmented generation pipeline for legal question answering. The first model call (Eq. 3, Algorithm 1 line 1) produces a private "snap answer" a0 and a separate pseudo-document p written in the formal style of legal authority; only p is embedded for dense retrieval (Eq. 4), after which a cross-encoder reranks the top-k. The second call (Eq. 5) answers the original question from the retrieved passages with a0 withheld, framed as a confirmation-bias guardrail (Eq. 7). The only structural difference from HyDE is that HyDE generates the pseudo-document directly from the question (Eq. 6), whereas SCOPE conditions it on the intermediate snap answer (Section 5.3). Evaluation covers two Stanford RegLab benchmarks (Zheng et al., 2025): BarExamQA (1,195 questions, low question-corpus lexical overlap) and HousingQA (6,853 questions, high overlap), with Llama 3.1 8B, Gemma 4 26B, and Llama 3.3 70B as both query generator and answer model. The headline claim is that on BarExamQA, SCOPE lifts gold-passage Hit@5 from a 1.4% raw baseline to 9.5%-12.1% (Table 4, Table 11) and improves answer accuracy, while on HousingQA it reaches parity.

Strengths
The control design isolates query formation correctly. The paper separates four roles (Section 3.2, lines 134-139): LLM-only, raw-question RAG, HyDE (generated query without the snap-answer signal), and Gold Evidence (the labeled passage supplied directly). Using the same model for both the query-generation and answer calls (lines 259-262) means any lift cannot be attributed to a stronger reader. Table 1 plus Tables 7 and 8 report all four against the gold upper bound.

Reporting is traceable and largely honest about limits. Appendix A.1 discloses the inclusion policy and the incomplete cell grid, Table 8 gives the Gold Evidence ceiling, line 308 concedes the gold passage appears in only about one in nine top-five lists, and Table 10 shows the HousingQA jurisdiction filter, not the query method, drives most of that benchmark's retrieval (Hit@5 2.8% to 36.9%). The worked example (Table 2, Figure 2, item mbe_1175) concretely illustrates the colloquial-to-doctrinal gap.

Weaknesses
W1. The core method overlaps closely with uncited prior work, and the one novel component does nothing. SCOPE's mechanism is to generate text in the style of real statutory authority from the model's parametric knowledge and use it as the retrieval query (Section 3.1, Eq. 3-4). This is the same idea as ParSeR in KoBLEX (EMNLP 2025), which generates "parametric provisions," statute-style passages built from the LLM's parametric knowledge, as query scaffolds and then runs a Retrieve-Rerank-Selection pipeline. The mapping is close to one-to-one: parametric provision to pseudo-document, and Retrieve-Rerank-Selection to retrieve-rerank-answer. The honest differences are that ParSeR is multi-hop Korean statutory QA with an explicit selection stage and no discarded snap-answer step, whereas SCOPE targets US benchmarks and adds the snap answer. But the paper cites neither KoBLEX nor any legal generated-query method of this kind, so the novelty claim is untested against the closest prior art. The novelty that does remain, the snap-answer conditioning that separates SCOPE from HyDE (Eq. 3 vs Eq. 6), shows no measurable benefit: Table 11 gives SCOPE-over-HyDE BarExamQA Hit@5 deltas of +1.2, +0.7, +0.5pp (8B, 26B, 70B), Table 1 gives answer deltas of +0.8, +1.7, -0.5pp, and pooled over five slices HyDE versus Snap is +0.1pp (Table 6). No significance test is reported for any SCOPE-versus-HyDE comparison; the single test in the paper (Gemma 26B, McNemar p < 0.001, line 301) is against raw-question RAG. The method is thus squeezed between a near-identical uncited legal method and a general baseline it does not improve on.

W2. The legal-retrieval literature on the motivating vocabulary gap is absent, including the analysis that would test the premise. The paper's entire motivation is the colloquial-to-statutory lexical gap (Section 1, lines 96-106), but it engages none of the legal-IR work that targets exactly this. GuRE (Kim et al., NLLP 2025) trains a generative query rewriter to mitigate vocabulary mismatch in legal passage retrieval, the same problem framed the same way, and analyzes the strongly long-tailed distribution of legal passages, showing that frequent and highly cited targets behave differently from rare ones under contrastive retrieval. SCOPE neither compares against a trained query rewriter nor offers any distributional characterization of its corpora. Without that, there is no evidence on whether the snap-conditioned passage helps uniformly or only on a slice of the corpus, and the method cannot be situated against the obvious legal-domain alternative.

W3. The reported benefit is not holistic, and the headline is framed against the weakest baseline. Absolute retrieval stays low: BarExamQA Hit@5 tops out near 12% (Tables 4 and 11), so the gold passage is missed roughly nine times in ten (line 308), and the paper offers no corpus-level breakdown of where retrieval succeeds or fails. The answer-side framing compounds this. The abstract and Section 5.1 (line 299) report gains of "+2.4, +4.0, +5.1pp over raw-question RAG," but raw-question RAG is the worst method on BarExamQA (average 69.0, Table 1), below LLM-only (72.3); against LLM-only the SCOPE deltas are -0.4, +1.2, +1.0pp, negative at 8B. An eightfold Hit@5 improvement (1.4% to about 12%) moves average answer accuracy only from 72.3 to 72.9, indicating answers are driven by parametric knowledge, not retrieved evidence. On HousingQA the method regresses rather than reaches "parity": SCOPE is 59.0 and 59.6 versus raw-question RAG 62.3 and 62.1 (Table 1; pooled -2.9pp, Table 6), and is the weakest non-LLM method in that column.

Comments
The decisive missing pieces are positioning and a fair comparison. The paper should cite and compare against ParSeR (KoBLEX) and a legal query rewriter such as GuRE, since these define the actual prior art for generated legal queries, and should run a significance-tested SCOPE-versus-HyDE contrast on both retrieval and answers, because that is the only comparison that isolates the snap-answer mechanism. It should also report against LLM-only as the primary BarExamQA baseline and add a corpus-level analysis (for example by passage frequency or jurisdiction) so the reader can see whether the gain is holistic. Smaller points: the evaluation matrix is one-quarter empty (31 of 42 cells, HousingQA 10 of 21, Appendix A.1 line 568, with Gemma 4 26B absent from the HousingQA column of Table 1) and confidence intervals are reported on no answer-accuracy number, so per-model trends such as "the lift grows with model size" (lines 301-302) are unsupported and in fact contradicted by the -0.5pp 70B cell; and the "leading answer-token efficiency" claim (Section 5.5, Table 3) excludes first-stage query-generation tokens (lines 279-280), which flatters SCOPE because its first call emits both a snap answer and a passage. Finally, the confirmation-bias guardrail (Eq. 7) is asserted rather than tested; an ablation that passes the snap answer into the answer call and shows it hurts would convert the rationale into evidence.

References
KoBLEX: Open Legal Question Answering with Multi-hop Reasoning (introduces the ParSeR retrieval method; EMNLP 2025): https://aclanthology.org/2025.emnlp-main.200/
GuRE: Generative Query REwriter for Legal Passage Retrieval (Kim et al., NLLP 2025): https://aclanthology.org/2025.nllp-1.31/
Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE; Gao et al., ACL 2023): https://aclanthology.org/2023.acl-long.99/
A Reasoning-Focused Legal Retrieval Benchmark (BarExamQA and HousingQA; Zheng et al., CSLAW 2025): https://dx.doi.org/10.1145/3709025.3712219
Rating: 2: Strong rejection
Confidence: 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature