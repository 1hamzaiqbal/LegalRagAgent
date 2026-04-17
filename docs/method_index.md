# Evaluation Method Index

This document describes every evaluation mode in `eval/eval_config.py:EVAL_MODES` and how each one is implemented in `eval/eval_harness.py`.

## Conventions

- "Formatted question" means the output of `_fmt(row, config)`, which wraps the raw question in dataset-specific formatting.
- "Snap" means an initial direct answer generated with `_system_prompt(config, "answer")`.
- "Retrieval" means the local helper `_retrieve_and_format()`: pooled query retrieval, evidence formatting, and cross-encoder reranking. It does not spend an LLM call.
- Unless otherwise noted, retrieval uses `k=5`, honors `_where_from_config(config)` for `source_filter`, and searches `_collection_for_config(config)`.
- Call counts below are mechanics, not performance metrics. Where the harness branches, totals are shown as ranges or formulas.
- `full_pipeline` is the exception: it delegates to `main.run()` and uses the skill prompts in `skills/*.md` rather than `_system_prompt()`.

## System Prompt Keys

These are the keys handled by `_system_prompt(config, role)` in `eval/eval_harness.py`.

- `answer`: Direct answering prompt for the dataset. It tells the model to answer the formatted question directly and emit the dataset-specific final format.
- `rag`: Answering prompt used when retrieved passages are shown. It tells the model to reason independently first, then use the passages to verify or refine the answer.
- `hyde`: HyDE generation prompt used when the query should be a hypothetical legal reference passage generated from the question alone.
- `snap_hyde`: Snap-informed HyDE generation prompt used when the query should be a hypothetical legal reference passage generated from the model's prior answer and reasoning.
- `devil_hyde`: Adversarial HyDE generation prompt used to write a passage that would challenge the snap answer.
- `top2_snap`: Direct-answer prompt that asks for a first-choice answer plus a plausible second-choice alternative.
- `top2_hyde`: HyDE generation prompt that writes a passage supporting the second-choice answer surfaced by `top2_snap`.

Dataset-specific behavior:

- `barexam`: multiple-choice wording with `Answer: (X)`.
- `housing`: housing-law Yes/No wording with `Answer: Yes` or `Answer: No`.
- `casehold`: holding-selection wording with `Answer: (X)`.
- `legal_rag` and `australian`: open-ended legal answer wording.
- If a dataset-specific prompt map does not define the requested key, `_system_prompt()` falls back to `answer`.

Inline prompts not covered by `_system_prompt()`:

- Some modes also use custom inline prompts or skill prompts for query rewriting, arbitration, gap analysis, report writing, decomposition, vectorless note generation, or aspect-query generation.

## 1. Baselines

This family also includes `rag_arbitration`, which is present in `EVAL_MODES` even though it was not listed in the original family breakdown.

### LLM Only (`llm_only`)

**One-line summary:** Direct LLM answer, no retrieval

**Pipeline:**

1. Answer the formatted question directly with `_system_prompt(config, "answer")`.
Total: `1` LLM call.

**System prompts used:** `answer`

**What the final decision-maker sees:** The formatted question only.

**Key design choice:** This is the pure parametric baseline with no retrieval, rewriting, or review stage.

**Example trace:** Detail log: [logs/eval_llm_only_cluster-vllm_20260408_1811_detail.jsonl](../logs/eval_llm_only_cluster-vllm_20260408_1811_detail.jsonl)

### RAG Simple (`rag_simple`)

**One-line summary:** Raw question → hybrid retrieval → synthesize

**Pipeline:**

1. Retrieve with the raw question text.
2. Answer once with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `1` LLM call.

**System prompts used:** `rag`

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** This is the simplest retrieval baseline: no query rewriting, no snap, and no arbitration.

**Example trace:** Detail log: [logs/eval_rag_simple_cluster-vllm_20260408_1813_detail.jsonl](../logs/eval_rag_simple_cluster-vllm_20260408_1813_detail.jsonl)

### RAG Rewrite (`rag_rewrite`)

**One-line summary:** Query rewrite → hybrid retrieval → synthesize

**Pipeline:**

1. Call `load_skill("query_rewriter")` to produce a primary query plus alternatives.
2. Retrieve with the rewritten multi-query set.
3. Answer once with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` LLM calls.

**System prompts used:** `rag`, plus the custom `query_rewriter` skill prompt.

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** It spends one extra call up front to improve retrieval recall instead of changing the final answer prompt.

**Example trace:** Detail log: [logs/eval_rag_rewrite_deepseek_20260322_20_detail.jsonl](../logs/eval_rag_rewrite_deepseek_20260322_20_detail.jsonl)

### Golden Passage (`golden_passage`)

**One-line summary:** LLM answer with gold passage injected as context

**Pipeline:**

1. If `gold_passage` is missing, fall back to `llm_only`.
2. Otherwise answer once with `_system_prompt(config, "rag")` over the gold passage plus the formatted question.
Total: `1` LLM call.

**System prompts used:** `rag`

**What the final decision-maker sees:** The gold passage plus the formatted question.

**Key design choice:** It isolates answer synthesis quality by bypassing retrieval entirely and handing the model the known-good passage.

**Example trace:** Detail log: [logs/eval_golden_passage_cluster-vllm_20260408_1615_detail.jsonl](../logs/eval_golden_passage_cluster-vllm_20260408_1615_detail.jsonl)

### Golden Arbitration (`golden_arbitration`)

**One-line summary:** LLM answers naive, then reviews golden passage (neutral framing)

**Pipeline:**

1. If `gold_passage` is missing, fall back to `llm_only`.
2. Produce a snap answer with `_system_prompt(config, "answer")`.
3. Re-answer with a custom neutral arbitration prompt over the previous answer, the gold passage, and the formatted question.
Total: `2` LLM calls.

**System prompts used:** `answer`, plus a custom neutral arbitration prompt.

**What the final decision-maker sees:** The previous answer, the gold passage, and the formatted question.

**Key design choice:** It tests revision behavior separately from retrieval by forcing a before/after review over oracle evidence.

**Example trace:** No detail log found in `logs/`.

### Golden Arb Conservative (`golden_arb_conservative`)

**One-line summary:** LLM answers naive, then reviews golden passage (biased toward keeping)

**Pipeline:**

1. If `gold_passage` is missing, fall back to `llm_only`.
2. Produce a snap answer with `_system_prompt(config, "answer")`.
3. Re-answer with a custom conservative arbitration prompt over the previous answer, the gold passage, and the formatted question.
Total: `2` LLM calls.

**System prompts used:** `answer`, plus a custom conservative arbitration prompt.

**What the final decision-maker sees:** The previous answer, the gold passage, and the formatted question.

**Key design choice:** The second call is explicitly biased toward keeping the first answer unless the evidence is clearly stronger.

**Example trace:** No detail log found in `logs/`.

### RAG Arbitration (`rag_arbitration`)

**One-line summary:** LLM answers naive, then reviews retrieved passages (conservative)

**Pipeline:**

1. Call `load_skill("query_rewriter")` to generate retrieval queries.
2. Produce a snap answer with `_system_prompt(config, "answer")`.
3. Retrieve with the rewritten queries.
4. Re-answer with a custom conservative arbitration prompt over the previous answer, the retrieved passages, and the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, plus the custom `query_rewriter` skill prompt and a custom conservative arbitration prompt.

**What the final decision-maker sees:** The previous answer, the retrieved passages, and the formatted question.

**Key design choice:** It keeps the rewrite baseline but swaps the final one-shot `rag` call for a snap-then-review arbitration step.

**Example trace:** No detail log found in `logs/`.

## 2. HyDE Family

### RAG HyDE (`rag_hyde`)

**One-line summary:** HyDE: LLM generates hypothetical answer, embeds it to retrieve

**Pipeline:**

1. Generate one hypothetical legal-reference passage from the question with `_system_prompt(config, "hyde")`.
2. Retrieve using that passage as the query.
3. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` LLM calls.

**System prompts used:** `hyde`, `rag`

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** Retrieval is steered by a synthetic answer passage rather than the raw question.

**Example trace:** Detail log: [logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl](../logs/eval_rag_hyde_cluster-vllm_20260415_1346_detail.jsonl)

### RAG Multi-HyDE (`rag_multi_hyde`)

**One-line summary:** Multi-HyDE: 3 hypothetical passages (rule/exception/application)

**Pipeline:**

1. Generate one custom multi-HyDE response that contains separate rule, exception, and application passages.
2. Split those passages and retrieve with the pooled set.
3. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` LLM calls.

**System prompts used:** `rag`, plus a custom inline multi-HyDE generation prompt.

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** It broadens recall by searching with multiple synthetic views of the same issue in a single generation step.

**Example trace:** No detail log found in `logs/`.

### RAG Snap-HyDE (`rag_snap_hyde`)

**One-line summary:** Snap-informed HyDE: answer first, then targeted retrieval

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a HyDE passage from that answer with `_system_prompt(config, "snap_hyde")`.
3. Retrieve with the snap-informed HyDE passage.
4. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question; the snap is not shown to the final call.

**Key design choice:** Retrieval is conditioned on the model's first-pass reasoning, but the final answer is still a fresh evidence-driven answer.

**Example trace:** Detail log: [logs/eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl](../logs/eval_rag_snap_hyde_cluster-vllm_20260413_1102_detail.jsonl)

### Snap-HyDE Aligned (`snap_hyde_aligned`)

**One-line summary:** Snap-HyDE aligned: HyDE for dense retrieval, raw question for cross-encoder reranking

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Retrieve densely with the HyDE passage but rerank against the raw question.
4. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** It decouples the embedding query from the reranking query to isolate embedding quality from reranker quality.

**Example trace:** No detail log found in `logs/`.

### RAG HyDE Arb (`rag_hyde_arb`)

**One-line summary:** HyDE retrieval + snap-then-review arbitration (conservative)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a question-only HyDE passage with `_system_prompt(config, "hyde")`.
3. Retrieve with that HyDE passage.
4. Re-answer with a custom conservative arbitration prompt over the previous answer, the retrieved passages, and the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, `hyde`, plus a custom conservative arbitration prompt.

**What the final decision-maker sees:** The previous answer, the retrieved passages, and the formatted question.

**Key design choice:** It combines HyDE retrieval with explicit revision pressure instead of a plain `rag` final answer.

**Example trace:** No detail log found in `logs/`.

### Snap-HyDE Aspect (`snap_hyde_aspect`)

**One-line summary:** Snap-HyDE + aspect queries: HyDE passage + rule/exception queries for diverse retrieval

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Generate two custom aspect queries (`rule` and `exception`) as JSON.
4. Retrieve with the pooled set: HyDE passage plus the aspect queries.
5. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `4` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom aspect-query generator prompt.

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question.

**Key design choice:** It augments one snap-informed synthetic passage with explicit rule/exception probes to diversify retrieval.

**Example trace:** No detail log found in `logs/`.

### RAG Devil HyDE (`rag_devil_hyde`)

**One-line summary:** Devil's advocate HyDE: retrieve for AND against snap answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a supporting passage with `_system_prompt(config, "snap_hyde")`.
3. Generate an opposing passage with `_system_prompt(config, "devil_hyde")`.
4. Retrieve with both passages pooled together.
5. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `4` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `devil_hyde`, `rag`

**What the final decision-maker sees:** Raw retrieved passages pooled from both the supportive and adversarial HyDE queries, plus the formatted question.

**Key design choice:** It deliberately searches for counter-evidence rather than only evidence that matches the snap answer.

**Example trace:** No detail log found in `logs/`.

### RAG Top2 HyDE (`rag_top2_hyde`)

**One-line summary:** Top-2 HyDE: retrieve for snap answer + second-choice answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "top2_snap")`, including a first and second choice.
2. Generate a primary-support passage with `_system_prompt(config, "snap_hyde")`.
3. Generate a second-choice support passage with `_system_prompt(config, "top2_hyde")`.
4. Retrieve with both passages pooled together.
5. Answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `4` LLM calls.

**System prompts used:** `top2_snap`, `snap_hyde`, `top2_hyde`, `rag`

**What the final decision-maker sees:** Raw retrieved passages pooled from both the first-choice and second-choice HyDE queries, plus the formatted question.

**Key design choice:** It tries to recover from a brittle first guess by retrieving for the most plausible alternative as well.

**Example trace:** No detail log found in `logs/`.

## 3. Gap-Informed

### Gap HyDE (`gap_hyde`)

**One-line summary:** Gap-informed HyDE: snap + gaps + evidence in final (full context)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run a custom gap-analysis prompt to extract up to 3 evidence gaps.
3. For each gap, generate a gap-focused HyDE query with `_system_prompt(config, "snap_hyde")` and retrieve supporting passages.
4. Answer once with `_system_prompt(config, "rag")` over the snap answer, structured gap sections, grouped evidence, and the formatted question.
Total: `2` calls if no gaps are found and the snap is reused; otherwise `3 + G` calls, where `G` is the number of investigated gaps.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** The snap answer, structured gap descriptions, grouped supporting passages, and the formatted question.

**Key design choice:** It forces retrieval to target explicit uncertainty points instead of searching the whole question at once.

**Example trace:** No detail log found in `logs/`.

### Gap HyDE EV (`gap_hyde_ev`)

**One-line summary:** Gap-informed HyDE: evidence only in final (no snap, no gap labels)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, generate a gap-focused HyDE query with `_system_prompt(config, "snap_hyde")` and retrieve supporting passages.
4. Answer once with `_system_prompt(config, "rag")` over flat retrieved passages plus the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** Flat retrieved passages plus the formatted question; the snap and gap labels are hidden.

**Key design choice:** It keeps gap-targeted retrieval but removes both anchoring and explicit gap structure from the final call.

**Example trace:** No detail log found in `logs/`.

### Gap HyDE NoSnap (`gap_hyde_nosnap`)

**One-line summary:** Gap-informed HyDE: gaps + evidence in final (no snap answer)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, generate a gap-focused HyDE query with `_system_prompt(config, "snap_hyde")` and retrieve supporting passages.
4. Answer once with `_system_prompt(config, "rag")` over structured gap sections, grouped evidence, and the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** Structured gap descriptions, grouped supporting passages, and the formatted question; the snap is hidden.

**Key design choice:** It preserves gap structure while explicitly testing whether the snap answer is causing anchoring.

**Example trace:** No detail log found in `logs/`.

### Gap HyDE Flat (`gap_hyde_flat`)

**One-line summary:** Gap-informed HyDE: snap + flat evidence in final (no gap structure)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, generate a gap-focused HyDE query with `_system_prompt(config, "snap_hyde")` and retrieve supporting passages.
4. Answer once with `_system_prompt(config, "rag")` over the snap answer, flat pooled passages, and the formatted question.
Total: `2` calls if no gaps are found and the snap is reused; otherwise `3 + G` calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** The snap answer, flat pooled passages, and the formatted question.

**Key design choice:** It keeps the gap-targeted retrieval step but removes the explicit per-gap framing from the final answer prompt.

**Example trace:** No detail log found in `logs/`.

### Gap RAG (`gap_rag`)

**One-line summary:** Gap-informed RAG: snap + gaps + evidence in final (full context)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run a custom gap-analysis prompt to extract up to 3 evidence gaps.
3. Retrieve directly from each gap sub-question without a HyDE generation step.
4. Answer once with `_system_prompt(config, "rag")` over the snap answer, structured gap sections, grouped evidence, and the formatted question.
Total: `2` calls if no gaps are found and the snap is reused; otherwise `3` calls.

**System prompts used:** `answer`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** The snap answer, structured gap descriptions, grouped supporting passages, and the formatted question.

**Key design choice:** It uses gap targeting without any synthetic query generation, so the only extra reasoning step is the gap analyzer itself.

**Example trace:** No detail log found in `logs/`.

### Gap RAG NoSnap (`gap_rag_nosnap`)

**One-line summary:** Gap RAG without snap in final — tests anchoring hypothesis

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. Retrieve directly from each gap sub-question without HyDE.
4. Answer once with `_system_prompt(config, "rag")` over structured gap sections, grouped evidence, and the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, `rag`, plus a custom gap-analysis prompt.

**What the final decision-maker sees:** Structured gap descriptions, grouped supporting passages, and the formatted question; the snap is hidden.

**Key design choice:** This is the clean anchoring ablation for the gap-RAG path because retrieval is still gap-driven but the final call never sees the snap.

**Example trace:** Detail log: [logs/eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl](../logs/eval_gap_rag_nosnap_cluster-vllm_20260416_0544_detail.jsonl)

## 4. Subagent

### Subagent HyDE (`subagent_hyde`)

**One-line summary:** Subagent HyDE: per-gap HyDE retrieval + LLM summarization → reports only (no snap)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, generate a gap-focused HyDE query with `_system_prompt(config, "snap_hyde")`, retrieve passages, and write a short report with a custom report prompt.
4. Answer once with `_system_prompt(config, "rag")` over the report bundle plus the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + 2G` calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus custom gap-analysis and report prompts.

**What the final decision-maker sees:** Research reports only plus the formatted question.

**Key design choice:** Each gap gets its own retrieval-plus-summary worker, and the main answerer only sees the condensed reports.

**Example trace:** Detail log: [logs/eval_subagent_hyde_cluster-vllm_20260414_2121_detail.jsonl](../logs/eval_subagent_hyde_cluster-vllm_20260414_2121_detail.jsonl)

### Subagent RAG (`subagent_rag`)

**One-line summary:** Subagent RAG: per-gap RAG + LLM summarization → reports only (no snap)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, retrieve directly from the gap sub-question and write a short report with a custom report prompt.
4. Answer once with `_system_prompt(config, "rag")` over the report bundle plus the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and report prompts.

**What the final decision-maker sees:** Research reports only plus the formatted question.

**Key design choice:** It turns each gap investigation into a report-writing subtask without showing raw evidence to the final call.

**Example trace:** Detail log: [logs/eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl](../logs/eval_subagent_rag_cluster-vllm_20260416_1720_detail.jsonl)

### Subagent Hybrid (`subagent_hybrid`)

**One-line summary:** Subagent hybrid: per-gap RAG + LLM knowledge → combined reports (no snap)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, retrieve directly from the gap sub-question and write a custom hybrid report that mixes retrieved evidence with the model's own legal knowledge.
4. Answer once with `_system_prompt(config, "rag")` over the report bundle plus the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and hybrid-report prompts.

**What the final decision-maker sees:** Research reports only plus the formatted question.

**Key design choice:** The per-gap worker is allowed to blend corpus evidence with parametric knowledge before handing a report to the final answerer.

**Example trace:** No detail log found in `logs/`.

### Subagent RAG Evidence (`subagent_rag_evidence`)

**One-line summary:** Subagent RAG + evidence: reports + raw passages (no snap)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, retrieve directly from the gap sub-question and write a short report.
4. Answer once with `_system_prompt(config, "rag")` over the report bundle, the raw supporting passages, and the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and report prompts.

**What the final decision-maker sees:** Research reports, raw supporting passages, and the formatted question.

**Key design choice:** It tests whether the subagent summaries are enough on their own or whether the final answerer still benefits from seeing the original evidence.

**Example trace:** No detail log found in `logs/`.

### Subagent RAG Snap (`subagent_rag_snap`)

**One-line summary:** Subagent RAG + snap: reports + snap answer in final (tests anchoring with reports)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, retrieve directly from the gap sub-question and write a short report.
4. Answer once with `_system_prompt(config, "rag")` over the snap answer, the report bundle, and the formatted question.
Total: `2` calls if no gaps are found and the snap is reused; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and report prompts.

**What the final decision-maker sees:** The snap answer, research reports, and the formatted question.

**Key design choice:** It isolates whether reports help when the final call is still anchored to the original snap answer.

**Example trace:** No detail log found in `logs/`.

### Subagent RAG Full (`subagent_rag_full`)

**One-line summary:** Subagent RAG maximum info: reports + snap + raw passages in final

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, retrieve directly from the gap sub-question and write a short report.
4. Answer once with `_system_prompt(config, "rag")` over the snap answer, the report bundle, the raw supporting passages, and the formatted question.
Total: `2` calls if no gaps are found and the snap is reused; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and report prompts.

**What the final decision-maker sees:** The snap answer, research reports, raw supporting passages, and the formatted question.

**Key design choice:** It is the maximum-information subagent variant, testing whether extra context helps more than it hurts.

**Example trace:** No detail log found in `logs/`.

## 5. Snap + Report Combos

### Snap-HyDE Report (`snap_hyde_report`)

**One-line summary:** Snap-HyDE + summarization: snap_hyde retrieval → summarize → report only (no snap, no raw)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Retrieve with that HyDE passage.
4. Write one condensed report over all retrieved passages with a custom report prompt.
5. Answer once with `_system_prompt(config, "rag")` over the report plus the formatted question.
Total: `4` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom report prompt.

**What the final decision-maker sees:** One synthesized report plus the formatted question.

**Key design choice:** It keeps the proven snap-HyDE retrieval path but replaces raw evidence with a single denoised report.

**Example trace:** No detail log found in `logs/`.

### Snap-HyDE Report Snap (`snap_hyde_report_snap`)

**One-line summary:** Snap-HyDE + summarization + snap: report + snap answer in final

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Retrieve with that HyDE passage.
4. Write one condensed report over all retrieved passages with a custom report prompt.
5. Answer once with `_system_prompt(config, "rag")` over the snap answer, the report, and the formatted question.
Total: `4` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom report prompt.

**What the final decision-maker sees:** The snap answer, one synthesized report, and the formatted question.

**Key design choice:** It is the anchored version of `snap_hyde_report`, letting the final call see both the snap and the denoised report.

**Example trace:** No detail log found in `logs/`.

### Snap RAG (`snap_rag`)

**One-line summary:** Snap + simple RAG: snap answer then retrieve with raw question, re-answer with both

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Retrieve with the raw question.
3. Answer once with `_system_prompt(config, "rag")` over the snap answer, retrieved passages, and the formatted question.
Total: `2` LLM calls.

**System prompts used:** `answer`, `rag`

**What the final decision-maker sees:** The snap answer, raw retrieved passages, and the formatted question.

**Key design choice:** It is the cheapest way to test whether exposing the snap helps a plain RAG answer.

**Example trace:** No detail log found in `logs/`.

### Snap RAG NoSnap (`snap_rag_nosnap`)

**One-line summary:** Snap + simple RAG: snap then retrieve, but final call only sees evidence (control)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Retrieve with the raw question.
3. Answer once with `_system_prompt(config, "rag")` over the retrieved passages and the formatted question only.
Total: `2` LLM calls.

**System prompts used:** `answer`, `rag`

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question; the snap is hidden.

**Key design choice:** It keeps the same retrieval path as `snap_rag` but removes the final anchoring input.

**Example trace:** No detail log found in `logs/`.

## 6. Parametric / Vectorless

### Vectorless Direct (`vectorless_direct`)

**One-line summary:** Historical 'vectorless' reasoning: snap → generate doctrinal note from parametric knowledge → answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a doctrinal note with the custom `_VECTORLESS_DIRECT` prompt.
3. Answer once with the custom `_VECTORLESS_FINAL` prompt over the generated note plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, plus custom vectorless generation and final prompts.

**What the final decision-maker sees:** A generated legal-reference note plus the formatted question.

**Key design choice:** It replaces corpus retrieval with a synthetic doctrinal note drawn from parametric memory.

**Example trace:** Detail log: [logs/eval_vectorless_direct_cluster-vllm_20260413_0140_detail.jsonl](../logs/eval_vectorless_direct_cluster-vllm_20260413_0140_detail.jsonl)

### Vectorless Role (`vectorless_role`)

**One-line summary:** Historical 'vectorless' reasoning: snap → role-conditioned parametric note (textbook/casebook/barprep via --tag) → answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a role-conditioned note with one of the custom `_VECTORLESS_ROLES` prompts chosen from `config.tag`.
3. Answer once with the custom `_VECTORLESS_FINAL` prompt over the generated note plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, plus custom role-conditioned vectorless generation and final prompts.

**What the final decision-maker sees:** A generated role-specific note plus the formatted question.

**Key design choice:** The only thing that changes is the style and framing of the generated doctrinal note.

**Example trace:** No detail log found in `logs/`.

### Vectorless Elements (`vectorless_elements`)

**One-line summary:** Historical 'vectorless' reasoning: snap → identify dispositive legal elements → answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate an element-by-element analysis with the custom `_VECTORLESS_ELEMENTS` prompt.
3. Answer once with the custom `_VECTORLESS_FINAL` prompt over that generated analysis plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, plus custom vectorless element-generation and final prompts.

**What the final decision-maker sees:** A generated element map plus the formatted question.

**Key design choice:** The intermediate note is structured around dispositive elements rather than a free-form doctrinal summary.

**Example trace:** No detail log found in `logs/`.

### Vectorless Choice Map (`vectorless_choice_map`)

**One-line summary:** Historical 'vectorless' reasoning: snap → map rule + distractor + decisive fact → answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a rule-versus-distractor map with the custom `_VECTORLESS_CHOICE_MAP` prompt.
3. Answer once with the custom `_VECTORLESS_FINAL` prompt over that generated map plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, plus custom vectorless choice-map generation and final prompts.

**What the final decision-maker sees:** A generated rule/distractor note plus the formatted question.

**Key design choice:** The intermediate note is optimized for distinguishing the best answer from the strongest distractor.

**Example trace:** No detail log found in `logs/`.

### Vectorless NoSnap (`vectorless_nosnap`)

**One-line summary:** Historical 'vectorless' reasoning without snap: question → generate knowledge → answer (2-call snap ablation)

**Pipeline:**

1. Generate a doctrinal note directly from the formatted question with the custom `_VECTORLESS_DIRECT` prompt.
2. Answer once with the custom `_VECTORLESS_FINAL` prompt over that generated note plus the formatted question.
Total: `2` LLM calls.

**System prompts used:** Custom vectorless generation and final prompts only.

**What the final decision-maker sees:** A generated legal-reference note plus the formatted question.

**Key design choice:** It is the snap-ablation control for the vectorless family.

**Example trace:** Detail log: [logs/eval_vectorless_nosnap_cluster-vllm_20260414_0518_detail.jsonl](../logs/eval_vectorless_nosnap_cluster-vllm_20260414_0518_detail.jsonl)

### Vectorless Hybrid (`vectorless_hybrid`)

**One-line summary:** Hybrid: generated parametric knowledge + vector RAG evidence pooled → answer (4 calls)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a doctrinal note with the custom `_VECTORLESS_DIRECT` prompt.
3. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")` and retrieve `k=3` passages.
4. Answer once with the custom `_VECTORLESS_FINAL` prompt over the generated note, retrieved passages, and the formatted question.
Total: `4` LLM calls.

**System prompts used:** `answer`, `snap_hyde`, plus custom vectorless generation and final prompts.

**What the final decision-maker sees:** A generated legal-reference note, raw retrieved passages, and the formatted question.

**Key design choice:** It is the only vectorless mode that explicitly combines parametric note generation with corpus evidence.

**Example trace:** No detail log found in `logs/`.

### Vectorless Keyword (`vectorless_keyword`)

**One-line summary:** Historical 'vectorless' keyword baseline: snap → generate search terms → corpus retrieval → answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate 3 to 5 legal search phrases with a custom keyword-generator prompt.
3. Retrieve separately for each keyword, deduplicate, and cross-encoder rerank against the raw question.
4. Answer once with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `3` LLM calls.

**System prompts used:** `answer`, `rag`, plus a custom keyword-generator prompt.

**What the final decision-maker sees:** Raw retrieved passages plus the formatted question; the snap is hidden.

**Key design choice:** It is a bridge baseline between vectorless note generation and ordinary retrieval: the model only generates search terms.

**Example trace:** No detail log found in `logs/`.

### Gap Vectorless (`gap_vectorless`)

**One-line summary:** Gap + historical 'vectorless' reasoning: per-gap generated knowledge reports, no corpus retrieval

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run custom gap analysis.
3. For each gap, generate a gap-specific doctrinal note with the custom `_VECTORLESS_DIRECT` prompt.
4. Answer once with `_system_prompt(config, "rag")` over the generated gap reports plus the formatted question.
Total: `3` calls on the no-gap path because it forces a fresh answer; otherwise `3 + G` calls.

**System prompts used:** `answer`, `rag`, plus custom gap-analysis and vectorless generation prompts.

**What the final decision-maker sees:** Generated per-gap reports plus the formatted question.

**Key design choice:** It keeps the gap-targeting machinery but replaces corpus retrieval with generated legal notes.

**Example trace:** No detail log found in `logs/`.

## 7. Entity Search

### Entity Search (`entity_search`)

**One-line summary:** Entity graph search: NLP inverted index → real corpus passages → cross-encoder rerank → answer (1 LLM call, zero embeddings)

**Pipeline:**

1. Load the prebuilt entity graph and inverted index.
2. Extract legal terms and phrases from the raw question, score candidate passage IDs by entity overlap, and rerank the resulting passages with the cross-encoder.
3. Answer once with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
4. If the graph is missing or no candidates survive, fall back to one `answer` call.
Total: `1` LLM call.

**System prompts used:** `rag` on the search path, `answer` on the fallback path.

**What the final decision-maker sees:** Raw entity-search passages plus the formatted question.

**Key design choice:** It is the zero-embedding retrieval baseline: symbolic entity lookup plus reranking instead of vector search.

**Example trace:** Detail log: [logs/eval_entity_search_cluster-vllm_20260415_0454_detail.jsonl](../logs/eval_entity_search_cluster-vllm_20260415_0454_detail.jsonl)

### Snap Entity Search (`snap_entity_search`)

**One-line summary:** Snap + entity search: snap first, then entity graph corpus search, answer fresh without snap (2 LLM calls)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run entity-graph retrieval from the raw question, rerank the candidates, and answer with `_system_prompt(config, "rag")`.
3. If the graph or candidates are unavailable, make a fresh `answer` call instead of leaking the snap.
Total: `2` LLM calls.

**System prompts used:** `answer`, `rag`

**What the final decision-maker sees:** Raw entity-search passages plus the formatted question; the snap is hidden from the final call.

**Key design choice:** It adds a snap call as a control without letting that snap steer the entity retrieval query.

**Example trace:** Detail log: [logs/eval_snap_entity_search_cluster-vllm_20260414_1918_detail.jsonl](../logs/eval_snap_entity_search_cluster-vllm_20260414_1918_detail.jsonl)

### Snap Entity Informed (`snap_entity_informed`)

**One-line summary:** Snap-informed entity search: extract entities from snap reasoning + question for better search terms (2 LLM calls)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Concatenate the raw question and snap reasoning, run entity-graph retrieval over that combined text, rerank the candidates, and answer with `_system_prompt(config, "rag")`.
3. If the graph or candidates are unavailable, make a fresh `answer` call instead.
Total: `2` LLM calls.

**System prompts used:** `answer`, `rag`

**What the final decision-maker sees:** Raw entity-search passages plus the formatted question; the snap is hidden from the final call.

**Key design choice:** It is the entity-search analogue of HyDE: the snap reasoning expands the lexical entity set used for retrieval.

**Example trace:** No detail log found in `logs/`.

## 8. Confidence / Gating

### Confidence Gated (`confidence_gated`)

**One-line summary:** Confidence-gated: 3 snap votes, unanimous=skip RAG, disagreement=Snap-HyDE

**Pipeline:**

1. Produce three independent snap answers with `_system_prompt(config, "answer")`.
2. If all three extracted answers agree, return the majority snap directly.
3. Otherwise generate one snap-informed HyDE passage from the majority answer with `_system_prompt(config, "snap_hyde")`, retrieve, and answer with `_system_prompt(config, "rag")`.
Total: `3` calls on the unanimous path; `5` calls on the Snap-HyDE path.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Either no final decision-maker at all because the majority snap is returned, or raw retrieved passages plus the formatted question on the routed path.

**Key design choice:** It uses answer agreement itself as the gate for whether retrieval is worth paying for.

**Example trace:** Detail log: [logs/eval_confidence_gated_groq-llama70b_20260325_1903_detail.jsonl](../logs/eval_confidence_gated_groq-llama70b_20260325_1903_detail.jsonl)

### CE Threshold (`ce_threshold`)

**One-line summary:** CE-thresholded Snap-HyDE: discard low-scoring evidence, fall back to snap answer

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Retrieve and inspect the best cross-encoder score.
4. If `max_ce_score < 4.0`, return the snap answer directly.
5. Otherwise answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` calls on the CE-fallback path; `3` calls on the RAG path.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Either no final decision-maker at all because the snap is returned, or raw retrieved passages plus the formatted question on the high-CE path.

**Key design choice:** It gates on evidence quality rather than answer confidence.

**Example trace:** Detail log: [logs/eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl](../logs/eval_ce_threshold_cluster-vllm_20260415_2022_detail.jsonl)

### Conf CE Threshold (`conf_ce_threshold`)

**One-line summary:** Confidence-gated + CE threshold: 3-vote gating, then CE threshold on RAG path

**Pipeline:**

1. Produce three independent snap answers with `_system_prompt(config, "answer")`.
2. If all three extracted answers agree, return the majority snap directly.
3. Otherwise generate one snap-informed HyDE passage from the majority answer with `_system_prompt(config, "snap_hyde")` and retrieve.
4. If `max_ce_score < 4.0`, return the majority snap directly.
5. Otherwise answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `3` calls on the unanimous path, `4` calls on the CE-fallback path, and `5` calls on the RAG path.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Either no final decision-maker because a snap is returned, or raw retrieved passages plus the formatted question on the high-CE path.

**Key design choice:** It stacks an answer-confidence gate and an evidence-quality gate before paying for a final evidence-grounded answer.

**Example trace:** No detail log found in `logs/`.

### CE Threshold K3 (`ce_threshold_k3`)

**One-line summary:** CE-thresholded Snap-HyDE with k=3: fewer passages, higher quality

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`.
3. Retrieve only `k=3` passages and inspect the best cross-encoder score.
4. If the score is below the threshold, return the snap answer directly.
5. Otherwise answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` calls on the CE-fallback path; `3` calls on the RAG path.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Either no final decision-maker because the snap is returned, or raw retrieved passages plus the formatted question on the high-CE path.

**Key design choice:** It is the same quality gate as `ce_threshold`, but with a tighter evidence set.

**Example trace:** No detail log found in `logs/`.

## 9. Multi-Step Reasoning

### Decompose (`decompose`)

**One-line summary:** Decompose-then-answer: split into sub-questions, answer each, synthesize (no RAG)

**Pipeline:**

1. Use a custom decomposition prompt to split the question into 1 to 3 sub-questions; `config.tag` selects `structured` or `natural` decomposition.
2. Answer each sub-question independently with `_system_prompt(config, "answer")`.
3. Synthesize the sub-answers into one final answer with `_system_prompt(config, "answer")`.
Total: `2 + N` LLM calls, where `N` is the number of sub-questions (`3` to `5` total).

**System prompts used:** `answer`, plus a custom decomposition prompt.

**What the final decision-maker sees:** Sub-question analyses plus the original formatted question.

**Key design choice:** It buys extra reasoning structure without bringing in any external evidence.

**Example trace:** Detail log: [logs/eval_decompose_groq-llama70b_20260325_2109_detail.jsonl](../logs/eval_decompose_groq-llama70b_20260325_2109_detail.jsonl)

### Decompose RAG (`decompose_rag`)

**One-line summary:** Decompose + Snap-HyDE: sub-questions with per-issue retrieval, then synthesize

**Pipeline:**

1. Use a custom decomposition prompt to split the question into 1 to 3 sub-questions; `config.tag` selects `structured` or `natural` decomposition.
2. For each sub-question, answer it with `_system_prompt(config, "answer")`.
3. For each sub-question, generate a snap-informed HyDE passage with `_system_prompt(config, "snap_hyde")`, retrieve `k=3` passages, and keep the evidence bundle.
4. Synthesize all sub-question analyses and supporting passages into one final answer with `_system_prompt(config, "rag")`.
Total: `2 + 2N` LLM calls, where `N` is the number of sub-questions (`4` to `8` total).

**System prompts used:** `answer`, `snap_hyde`, `rag`, plus a custom decomposition prompt.

**What the final decision-maker sees:** Each sub-question, its snap analysis, its supporting passages, and the original formatted question.

**Key design choice:** It is the issue-by-issue retrieval version of `decompose`, so evidence is collected separately per sub-issue before synthesis.

**Example trace:** Detail log: [logs/eval_decompose_rag_groq-llama70b_20260326_0253_detail.jsonl](../logs/eval_decompose_rag_groq-llama70b_20260326_0253_detail.jsonl)

### Self Verify (`self_verify`)

**One-line summary:** Self-verification: snap answer then review for errors (2 calls, no RAG)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run a custom reviewer prompt that critiques and optionally corrects that answer.
Total: `2` LLM calls.

**System prompts used:** `answer`, plus a custom review prompt.

**What the final decision-maker sees:** The previous answer and the original formatted question.

**Key design choice:** The second call is cooperative self-correction rather than evidence-grounded retrieval.

**Example trace:** Detail log: [logs/eval_self_verify_groq-llama70b_20260327_1546_detail.jsonl](../logs/eval_self_verify_groq-llama70b_20260327_1546_detail.jsonl)

### Double Snap (`double_snap`)

**One-line summary:** Double-snap: two answers, agree=use, disagree=CE-threshold RAG

**Pipeline:**

1. Produce two independent snap answers with `_system_prompt(config, "answer")`.
2. If the extracted answers agree, return the first snap directly.
3. Otherwise generate one snap-informed HyDE passage from the first snap with `_system_prompt(config, "snap_hyde")` and retrieve.
4. If the best cross-encoder score is below the threshold, return the first snap.
5. Otherwise answer with `_system_prompt(config, "rag")` over the retrieved passages plus the formatted question.
Total: `2` calls on the agreement path, `3` calls on the CE-fallback path, and `4` calls on the RAG path.

**System prompts used:** `answer`, `snap_hyde`, `rag`

**What the final decision-maker sees:** Either no final decision-maker because a snap is returned, or raw retrieved passages plus the formatted question on the RAG path.

**Key design choice:** It is the cheapest confidence signal in the codebase: just two snaps before optional retrieval.

**Example trace:** Detail log: [logs/eval_double_snap_groq-llama70b_20260327_1648_detail.jsonl](../logs/eval_double_snap_groq-llama70b_20260327_1648_detail.jsonl)

### Snap Debate (`snap_debate`)

**One-line summary:** Snap-debate: snap then adversarial critique (2 calls, no RAG)

**Pipeline:**

1. Produce a snap answer with `_system_prompt(config, "answer")`.
2. Run a custom adversarial critique prompt that tries to find legal flaws and replace the answer if needed.
Total: `2` LLM calls.

**System prompts used:** `answer`, plus a custom adversarial critique prompt.

**What the final decision-maker sees:** The previous answer and the original formatted question.

**Key design choice:** The second call is explicitly hostile to the first answer, unlike the more cooperative `self_verify` reviewer.

**Example trace:** No detail log found in `logs/`.

## 10. Full Pipeline

### Full Pipeline (`full_pipeline`)

**One-line summary:** Full agentic pipeline (planner → executor → synthesizer)

**Pipeline:**

1. Call `main.run()`, which first routes the question to one or more collections with the `router` skill prompt.
2. Generate a planning table with the `planner` skill prompt.
3. Execute each planned step in the parallel executor:
   - `direct_answer` steps make one `synthesize_and_cite` call.
   - `rag_search` steps make a snap call, a HyDE call, and optionally a `synthesize_and_cite` call if the CE threshold is passed.
   - `web_search` steps run search/scrape locally and make one `synthesize_and_cite` call.
4. Run the `synthesizer` skill prompt over completed step outputs plus the accumulated evidence index.
5. If the synthesizer marks the answer incomplete, add new RAG steps from `missing_topics` and repeat executor plus synthesizer; the harness allows up to 3 rounds.
Total: Variable. Base graph calls are router `1`, planner `1`, and synthesizer `1` per round; each direct or web step adds `1` call, each RAG step adds `2` to `3` calls, and extra replanning rounds add another synthesizer round.

**System prompts used:** None from `_system_prompt()`; this mode uses the `router`, `planner`, `synthesize_and_cite`, and `synthesizer` skill prompts in `main.py`.

**What the final decision-maker sees:** The synthesizer sees completed step results plus the full evidence index, and it can request another research round before emitting the final answer.

**Key design choice:** This is the only fully agentic mode: planning, execution, synthesis, and completeness-driven replanning are all part of the loop.

**Example trace:** No detail log found in `logs/`.
