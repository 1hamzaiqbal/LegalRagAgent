---
title: Learning to Replicate Expert Judgment in Financial Tasks (Thinking Machines / Bridgewater blog 2026)
type: source
tags: [expert-judgment, llm-judge, fine-tuning, domain-adaptation, evaluation, finance]
created: 2026-07-02
updated: 2026-07-02
status: draft
url: https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/
local: references/thinking-machines-expert-judgment.md
authors: Su et al. (Bridgewater AIA Labs + Thinking Machines)
year: 2026
venue: Thinking Machines Lab blog (not peer-reviewed)
code: none
---

## TL;DR

Bridgewater AIA Labs + Thinking Machines fine-tune Qwen3-235B (RL via Tinker: GRPO -> CISPO with asymmetric clipping + interleaved task batching + on-policy distillation with an adaptive best-checkpoint teacher) to replicate expert investor judgment on six document-filtering tasks (relevance, boilerplate truncation). Frontier LLMs plateau below the 80% trust threshold even with expert-written prompts (74.3-78.2%); the custom model hits 84.66% avg accuracy at $4.72/1k tasks (13.8x cheaper than GPT 5.5). Expert judgment is encoded as *labels*, not rubrics — the explicit thesis is that the judgments that matter are the ones experts cannot articulate in a prompt. This is a direction-inspiration blog post, not a citable benchmark paper.

## Key claims / numbers

- Six tasks drawn from investors' daily workflow: 3 binary/graded relevance classifications, 1 recurring-vs-mixed content labeling, 2 boilerplate-truncation-point tasks; metric = accuracy per investor labels (+F1 on classification). *our-relevance:* relevance judgment as the atomic unit of expert work maps directly onto C1's complaint that SCOPE optimizes retrieval metrics, not practitioners' analytical processes.
- Naive prompting: frontier models at 45.6-50.1% avg accuracy (Opus 4.6/4.8, Gemini 3.1 Pro, GPT 5.4/5.5). Expert-written prompts (including expert task *reframing*, e.g. 3-way relevant-interesting / relevant-uninteresting / irrelevant) lift to 74.3-78.2%; automatic prompt optimization adds nothing further. *our-relevance:* articulable instructions hit a hard ceiling; tacit judgment needs training signal — an argument that prompt-only "legal expertise" (C3's critique of SCOPE) is structurally limited.
- Judgment collection: vendor non-expert labels were noisy; a disagreement-routing scheme (train a preliminary model on the noisy labels, evaluate it on its own training data, route the examples it can't fit to expert investors for relabeling) cleaned the train set cheaply; final eval on a held-out test set. *our-relevance:* a concrete, cheap protocol for building lawyer-verified relevance qrels — directly usable against C8's missing corpus-level/qrel analysis.
- Training recipe with full leave-one-out ablations: final 84.66% / F1 92.99; removing interleaved batching -> 72.18%, removing CISPO+asymmetric clipping -> 74.56%, removing OPD -> 72.39%, frozen instead of adaptive teacher -> 81.55%. Component deltas: +12.1% (interleaving vs mixed), +10.1% (CISPO vs importance sampling), +3.1% (adaptive vs frozen teacher). *our-relevance:* this is exactly the ablation discipline reviewers said SCOPE lacks (C7 snap-vs-HyDE untested, C12 a0-discard guardrail never ablated).
- Result: 84.7% vs best frontier 78.2% (GPT 5.5) = 29.8% error reduction; $4.72 vs $19.96-$92.59 per 1k tasks; cost accounting is end-to-end per task. *our-relevance:* honest all-inclusive cost reporting, unlike SCOPE's token-efficiency claim that excluded first-stage generation tokens (C11).
- No dataset sizes, annotation counts, or CIs are disclosed; data is proprietary, "a subset cleared for public release." *our-relevance:* usable as a design pattern only — we cannot import its numbers as baselines, and it would itself fail C11-style rigor scrutiny.

## Bearing on the review

- **C1 (practitioner analytical processes):** this post is the clearest template for answering C1 non-cosmetically. The pattern: decompose practitioner work into atomic judgment tasks -> collect expert labels (with cheap disagreement-routing verification) -> train/evaluate a judge against those labels, because expert judgment resists prompt articulation. A revised SCOPE paper should at minimum (a) score retrieval against *graded lawyer relevance judgments* rather than a single gold passage id, and (b) frame a learned legal-judgment layer (lawyer-label-trained judge over RAG outputs) as the path from retrieval metrics to practitioner utility.
- **C7/C12 (unablated components):** the leave-one-out ablation table is the standard to copy — every recipe component gets a quantified delta. SCOPE must do this for snap-answer conditioning and for discarding a0.
- **C11 (rigor/cost):** report end-to-end cost per question including query-generation tokens, as done here per 1k tasks.
- **C9 (parametric knowledge drives answers):** indirectly supportive of the reviewers — if judgment/answer-conversion is the bottleneck (our 8x retrieval lift moving answers only 72.3 -> 72.9), the leverage sits in the answerer/judge, not the retriever. A judgment-trained answerer is a stronger response to C9 than another query-expansion variant.

## Transfer to legal

Direct analogues of the six tasks: (1) passage relevance to a fact pattern (is this statute/holding on point for this bar-exam scenario — the exact judgment our Hit@5 proxies with one gold id); (2) authority-signal classification (does this case signal the controlling rule, cf. central-bank rate-direction); (3) question-conditioned document usefulness (their task 3 is literally RAG qrel annotation); (4-6) boilerplate segmentation of statutes/contracts/opinions. Concrete transfers for us: (i) **lawyer-graded qrels** — replace single-gold-id Hit@5 with graded relevance (controlling / persuasive / irrelevant), using the disagreement-routing trick to spend scarce lawyer time only on contested passages; (ii) **learned legal answer judge** — grade RAG outputs beyond exact-match (issue spotting, rule statement, citation support), trained on lawyer labels rather than prompted rubrics — our own [[geometry-vs-factuality]] result (LLM-judged factuality AUC ~0.55-0.58) already shows prompted LLM judges are weak, which is this post's starting observation in finance; (iii) **judgment-trained answerer** to attack the [[answer-conversion-gap]] directly. Honest caveats: we have no standing lawyer-annotator pool; the post hides sample sizes so the label budget is unknown; document triage is plausibly easier than multi-step legal application (BarExamQA MC), so the 84.7% pattern may not transfer to reasoning-heavy tasks; and RL-tuning a 235B MoE is outside our current compute envelope (a small trained reranker/judge is the realistic entry point, cf. [[gure]]'s trained rewriter precedent).

## Differentiation

No overlap with SCOPE's contribution: this is training-time judgment replication for document filtering in finance; SCOPE is inference-time generated-query retrieval in law. We are not pre-empted by it — but it also gives us no citable legal-NLP cover (non-peer-reviewed, proprietary data, no Ns). Its real bearing is uncomfortable in a useful way: it argues the highest-leverage use of expert signal is training a judge/answerer, not engineering queries — which aligns with our reviewers' C1/C9 reading that SCOPE optimized the wrong layer. If we adopt its pattern, that is a new work package (expert-judgment evaluation layer), not a revision of the SCOPE method itself.

## Links

[[expert-judgment-replication]], [[icml-ai4law-2026-rejection]], [[scope]], [[answer-conversion-gap]], [[geometry-vs-factuality]], [[legal-rag-benchmarks]], [[vocabulary-gap]], [[qpp]]; sibling sources: [[magesh2024hallucinationfree]] (lawyer evaluation of legal RAG outputs), [[gure]] (trained component for legal retrieval), [[guha2023legalbench]] (legal task decomposition), [[afane2026laborbench]].

## Raw source

- references/thinking-machines-expert-judgment.md (full-text markdown archive of the blog post, fetched 2026-07-02; no PDF/technical report exists — chart numbers extracted from the page's embedded data attributes)
