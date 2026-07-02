---
title: When do Generative Query and Document Expansions Fail? (Findings of EACL 2024)
type: source
tags: [query-expansion, hyde, information-retrieval, distribution-shift, regime, beir]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://arxiv.org/abs/2309.08541
local: references/weller-drift.pdf
authors: Weller et al. (Orion Weller, Kyle Lo, David Wadden, Dawn Lawrie, Benjamin Van Durme, Arman Cohan, Luca Soldaini)
year: 2024
venue: Findings of EACL 2024
code: https://github.com/orionw/LM-expansions
---

# When do Generative Query and Document Expansions Fail? (Weller et al., EACL Findings 2024)

## TL;DR
First comprehensive study of LM-based query/document expansion (11 techniques including HyDE, CoT expansion, LM-PRF, Doc2Query; 24 retrievers; 12 datasets grouped by distribution-shift type). Headline: a strong **negative correlation between a ranker's baseline nDCG@10 and its gain from expansion** — expansion helps weak models, hurts strong ones, independent of architecture and model scale. Their proposed mechanism (qualitative only): expansions add terms that "blur the relevance signal" of the original text, introducing false positives. Their recipe: expand for weak models or severe format shift (esp. long queries); otherwise don't.

## Key claims / numbers
- Negative correlation between baseline score and expansion delta across 24 retrievers (bi-encoder, late-interaction, sparse, cross-encoder) and 11 expansion methods; e.g. TREC DL19: DPR (base 38.4) gains +18.8 nDCG@10 from HyDE (+21.9 HyDE+Doc2Query), while MonoT5-3B (base 71.7) loses −4.0 from HyDE and −23.6 from D-LM PRF (Table 3). *our-relevance:* this is the macro law behind SCOPE's BarExamQA-vs-HousingQA split; grounds C9/C10 framing.
- Trend holds for in-domain, domain shift, relevance shift, and short-doc shift; the **one exception is long-query format shift** (Tip-of-My-Tongue, TREC CT 2021, ArguAna), where expansion helps most-to-all models; on short-doc Quora all models are harmed. *our-relevance:* BarExamQA's long colloquial fact patterns match their expansion-helps exception regime; HousingQA's short jurisdiction-filtered strong-retrieval setting matches the expansion-hurts regime — the SCOPE results are the predicted pattern, not an anomaly (C9, C10).
- Scale is not the cause: larger MonoT5 gains less, but all E5 sizes are equally impacted; base score, not parameter count, is the moderator (Fig. 5). *our-relevance:* supports framing regimes by retrieval strength/vocabulary gap rather than model size.
- Mechanism evidence is qualitative: 30 annotated failure cases from three datasets; 2 were unjudged false negatives, the remaining 28 all attributed to expansion terms diluting the relevance signal or making irrelevant docs look relevant (HELOC example, Fig. 3); plus violin plots of per-query rank-position change of relevant docs (Figs. 4/6). **No per-query predictor, no quantitative mechanism metric, no significance tests or confidence intervals anywhere in the paper.** *our-relevance:* this is exactly the gap our CE-affinity-movement mechanism (Spearman ~0.44, 5 BEIR sets, 3 retrievers) fills; also tempers C11 hypocrisy risk — breadth was their robustness argument, ours must be paired stats.
- The word "drift" **never appears in the paper.** Their wording: expansions "blur the relevance signal", "add additional noise that makes it difficult to discern between the top relevant documents (thus introducing false positives)", failures caused by "expansion weakening the original relevance signal". *our-relevance:* cite Weller for the phenomenon and this wording; attribute the term "query drift" to classical PRF literature (e.g. Mitra et al. 1998), not to Weller — our memory note "query drift (Weller'24)" is a mislabel.
- Their Limitations concede the recipe is not deployable without labels: "deciding whether to use augmentation requires having access to evaluation data for the target domain." *our-relevance:* our label-free QPP/geometric routing is positioned precisely against this stated gap.
- Ethics section notes "LMs may generate factually incorrect text, which could affect ranking," but their setup never puts generated text in the corpus, and their error analysis blames term noise, not factuality. *our-relevance:* consistent with our geometry-beats-factuality falsification (AUC 0.79–0.94 vs ~0.55) and a partial answer to C4 (SCOPE's pseudo-doc is query-side only, like theirs).
- Setup details: expansions from gpt-3.5-turbo-0613 (GPT-4, Claude v2, Llama2-70B-Chat replicate trends, App. A); reranking top-100 BM25 candidates (1k/10k same trends, App. C); zero-shot; nDCG@10; prepend/append/replace placement doesn't matter (App. B). Retrieval-only — **no answer/generation stage is ever evaluated.** *our-relevance:* SCOPE's downstream answer accuracy and answer-conversion gap are outside their scope entirely.

## Bearing on the review
- **C9/C10**: Weller'24 is the citation that turns our HousingQA regression from an embarrassment into the literature-predicted strong-regime outcome, and BarExamQA's 8x Hit@5 lift into the predicted weak-regime outcome. A revised paper must lead with this conditional framing ("when", not "that"), present LLM-only as a primary comparator, and label HousingQA a regression, not parity.
- **C3/C7**: they already treat HyDE as one member of a generated-expansion family whose value is regime-dependent; a revision cannot sell SCOPE as a new method, but can sell the *per-query, label-free* characterization Weller explicitly lacks — with significance tests for SCOPE-vs-HyDE, since neither we nor they had any.
- **C11**: their robustness argument is breadth (24x12x11), not statistics. Our matrix is small, so we need McNemar/CIs — no cover here.
- **C4**: cite their ethics note plus our factuality falsification: failures are geometric noise, not fabrication, and generated text never enters the corpus or the final answer context.
- **C2/C8**: general-IR must-cite, but does not cover legal IR; GuRE/KoBLEX still needed separately.

## Differentiation
We are pre-empted on the macro claim: the weak-helps/strong-hurts crossover, its robustness across generator LMs and retriever families, and the qualitative noise-dilution mechanism are all theirs (2024). We must not claim novelty for the dataset-level regime story. Our honest wedge: (1) a **quantitative per-query mechanism** — CE-affinity movement toward gold predicts per-query expansion benefit — where they offer 30 hand-annotated cases and descriptive rank-shift violins; (2) **label-free per-query/regime routing**, addressing the labeled-data limitation they state themselves; (3) a tested **falsification of the factuality explanation** they only gesture at; (4) **downstream answer outcomes** and the retrieval-to-answer conversion gap (their study stops at nDCG@10); (5) the legal-corpus, fixed-retriever, query-side-regime axis (their moderator is model baseline strength; ours is query/dataset vocabulary regime with the retriever held fixed). Do not overclaim: they do look at per-query rank movement descriptively, so our delta is "quantified and made predictive/deployable," not "first per-query look."

## Links
[[scope]], [[hyde]], [[query2doc]], [[vocabulary-gap]], [[weak-vs-strong-query-regime]], [[query-drift]], [[qpp]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[regime-routing]], [[generated-query-family]], [[legal-rag-benchmarks]], [[icml-ai4law-2026-rejection]], [[gure]], [[koblex-parser]], [[scope-paper-2026]]

## Raw source
- references/weller-drift.pdf (arXiv:2309.08541v2, EACL 2024 camera ready; ACL Anthology 2024.findings-eacl.134)
