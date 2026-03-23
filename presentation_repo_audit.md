# LegalRagAgent Repo Audit For Presentation

This document is a source-of-truth slide-prep note based on the code and artifacts currently in this repo.

Primary source-of-truth order:

1. `main.py`
2. `rag_utils.py`
3. `eval/*.py`
4. `utils/*.py`
5. `logs/` for recorded experimental outputs

Important caveat:

- The current docs have been aligned to the present runtime, but some logs and historical artifacts still reflect older versions of the system. Use the runtime code in `main.py` and `rag_utils.py` as the canonical implementation snapshot.

## 1. What This Repo Is Building

LegalRagAgent is an agentic legal research system over two corpora:

- `legal_passages`: Bar-exam study material, Wex-like doctrinal text, and caselaw.
- `housing_statutes`: Housing statutes across US jurisdictions.

The runtime goal is to answer legal research questions by:

- routing the question to the right corpus,
- decomposing it into smaller legal sub-questions,
- retrieving evidence,
- judging whether the evidence is good enough,
- escalating if retrieval fails,
- and synthesizing a final IRAC-style answer with evidence citations.

Core implementation files:

- `main.py`: LangGraph runtime, node logic, state, escalation, synthesis.
- `rag_utils.py`: retrieval stack.
- `llm_config.py`: provider abstraction.
- `web_scraper.py`: web result enrichment.
- `skills/*.md`: prompt contracts for each LLM role.

## 2. Actual Current Runtime Graph

The current implementation is a 5-node graph, not 4.

```mermaid
flowchart LR
    A([START]) --> B[router_node]
    B --> C[planner_node]
    C --> D[executor_node]
    D --> E[replanner_node]
    E -->|next or retry| D
    E -->|complete| F[synthesizer_node]
    F --> G([END])
```

Actual graph assembly is in `main.py`.

Key note:

- The current docs now match the 5-node runtime.
- Older artifacts in `logs/` and `case_studies/` may still reflect earlier architectures.

## 3. Runtime State And Data Flow

Shared LangGraph state:

- `agent_metadata`: provider, model, timestamps.
- `inputs`: currently mainly `question`.
- `run_config`: includes `max_steps`, default 7 in current runtime.
- `collections`: corpus choices selected by the router.
- `planning_table`: ordered list of `PlanningStep`s.
- `evidence_store`: accumulated evidence across steps.
- `final_answer`: final synthesized answer.
- `audit_log`: timestamped node trace entries.

Each `PlanningStep` stores:

- `step_id`
- `sub_question`
- `authority_target`
- `retrieval_hints`
- `action_type`
- `rewrite_attempt`
- `status`
- `result`
- `confidence`
- `evidence_ids`
- `retry_of`
- `judge_verdict`

## 4. Node-By-Node Description

### 4.1 Router

Purpose:

- choose which Chroma collection(s) to search.

Input:

- original user question.

Output:

- `collections`, usually one of:
  - `legal_passages`
  - `housing_statutes`

How it works:

- LLM prompt with collection descriptions.
- defaults to `legal_passages` on parse failure or uncertainty.

Why it matters:

- this is what makes the system multi-domain instead of a single legal corpus RAG.

### 4.2 Planner

Purpose:

- break the question into 2-5 focused legal sub-questions.

Input:

- original question.

Output:

- list of `PlanningStep`s with:
  - sub-question,
  - authority target,
  - retrieval hints,
  - action type.

Action types:

- `rag_search`
- `web_search`
- `direct_answer`

Why it matters:

- converts long legal prompts into doctrinal retrieval units.
- makes multi-hop questions tractable.

### 4.3 Executor

Purpose:

- execute the next pending step.

Execution paths:

1. `rag_search`
   - query rewrite into 1 primary + 2 alternatives
   - retrieve from routed collection(s)
   - deduplicate against prior evidence
   - rerank
   - synthesize cited sub-answer

2. `web_search`
   - DuckDuckGo search
   - scrape top URLs with `trafilatura`
   - synthesize from snippets + scraped text

3. `direct_answer`
   - answer directly from model knowledge
   - expected to hedge when doctrine may vary

Executor outputs:

- step result text
- new evidence entries
- confidence score for logging
- judge verdict

Important implementation detail:

- confidence is logged only; it does not drive control flow.

### 4.4 Judge / Verifier

This is implemented inside executor as a post-step evaluation stage.

Purpose:

- decide whether the step produced useful evidence.

Possible verdicts:

- `full`
- `partial`
- `false`

Behavior:

- `judge.md` is used for `rag_search` and `web_search`.
- `verifier.md` is used for `direct_answer`.

Why it matters:

- this is the main reliability mechanism that prevents the system from blindly trusting weak retrieval.

### 4.5 Replanner

Purpose:

- determine what to do next after a completed step.

Deterministic escalation path:

1. failed `rag_search` with no prior rewrite -> retry as rewritten `rag_search`
2. failed rewritten `rag_search` -> escalate to `web_search`
3. failed `web_search` -> fall back to `direct_answer`
4. failed `direct_answer` -> defer decision to LLM replanner

If judge says `partial`:

- keep the evidence and continue.

If judge says `full` or `partial`:

- call the replanner LLM to choose:
  - `next`
  - `complete`
  - `retry`

Why it matters:

- this is where the system becomes adaptive instead of a fixed chain.
- the agent can also stop early if enough evidence is already gathered.

### 4.6 Synthesizer

Purpose:

- aggregate all completed steps into a final IRAC answer.

Input:

- completed step summaries
- full evidence store

Output:

- final answer with `[Evidence N]` citations.

Why it matters:

- turns multiple local sub-answers into one globally coherent legal answer.

## 5. Retrieval Stack

The retrieval design is one of the main technical stories in this repo.

Components:

- Vector store: ChromaDB at `./chroma_db`
- Bi-encoder embeddings: `Alibaba-NLP/gte-large-en-v1.5`
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lexical retriever: BM25 via `rank-bm25`

Default `rag_search` retrieval flow:

1. rewrite sub-question into 3 queries
2. for each query, retrieve dense candidates
3. also retrieve BM25 candidates when corpus size allows
4. pool and deduplicate candidates by `idx`
5. rerank against the primary query with the cross-encoder
6. keep top 5 passages

Important design nuance:

- BM25 is disabled for collections larger than 1,000,000 documents.
- That means `housing_statutes` currently falls back to dense-only retrieval in the present code.
- Housing documents do carry `state` and `citation` metadata, but the current runtime does not yet use them for automatic retrieval filtering.

Why that matters for slides:

- the system is not using the exact same retrieval strategy for both corpora.
- bar exam corpus can use hybrid BM25+dense.
- housing corpus is effectively dense+rereank only under current code.

## 6. Data Pipeline

### 6.1 Datasets On Disk

Observed local dataset sizes:

- BarExam passages: 686,324 rows
- BarExam QA: 1,960 rows
- Housing statutes: 1,837,403 rows
- Housing QA: 6,853 rows

### 6.2 Chroma Collections Present Locally

Observed persisted collections:

- `legal_passages`: 686,324 embeddings
- `housing_statutes`: 1,837,403 embeddings
- `qa_memory`: 7 embeddings

Important note:

- `qa_memory` exists in Chroma but is not referenced by the current runtime registry or graph logic.
- Treat it as an orphan or leftover artifact unless future code starts using it.

### 6.3 Offline Prep Scripts

`utils/download_data.py`

- downloads and flattens BarExam QA data into CSV form.

`utils/download_housingqa.py`

- downloads housing statutes and question JSON, flattens QA labels into CSV.

`utils/load_corpus.py`

- LangChain-based loader, useful for smaller subsets or curated builds.

`utils/fast_embed.py`

- fast production embedding path.
- uses sentence-transformers directly.
- fp16 batching.
- chunked insert into Chroma.
- supports `--resume`.

## 7. Prompt Files And Their Roles

All 7 files in `skills/` are effectively part of the implementation, not just documentation.

`skills/planner.md`

- defines how questions are decomposed.

`skills/query_rewriter.md`

- defines the 3-query expansion behavior for `rag_search`.

`skills/synthesize_and_cite.md`

- defines per-step evidence-grounded synthesis.

`skills/judge.md`

- defines step-level sufficiency evaluation for retrieval-based steps.

`skills/verifier.md`

- verifies direct answers that did not use retrieval.

`skills/replanner.md`

- governs when to continue, retry, or stop.

`skills/synthesizer.md`

- defines the final IRAC answer format and citation policy.

Other markdown files:

- `README.md`: user-facing setup and current-system summary.
- `CLAUDE.md`: detailed internal repo note aligned to the current runtime.
- `ideas/parallel_agents.md`: future-work concept note for parallel retrieval specialists.

## 8. Evaluation Status

For the current presentation, only three evaluation runs should be treated as current-system, presentation-backed results.

### 8.1 Direct LLM Baseline

`eval/eval_baseline.py`

- asks the model the question directly, with no RAG and no graph.

Why it matters:

- this is the true "how much does the base model already know?" baseline.

### 8.2 Simple Retrieve-And-Answer Baseline

`eval/eval_bm25_baseline.py`

- runs one retrieval call plus one answer generation call.
- despite the filename, this now uses the current retriever (`retrieve_documents`), not BM25-only retrieval.

Why it matters:

- this is the main non-agent baseline for "retrieve once, answer once."

### 8.3 Golden Passage Upper Bound

`eval/eval_golden.py`

- gives the LLM the ground-truth passage directly.

Why it matters:

- estimates performance under perfect single-passage retrieval.

### 8.4 Older / Exploratory Eval Code Still Present In The Repo

The repo still contains older or exploratory eval scripts, including:

`eval/eval_retrieval_recall.py`
`eval/eval_reranker.py`
`eval/eval_retrieval.py`
`eval/web_search_suite.py`

These are useful code assets, but they should not be included in the current presentation unless they are rerun and explicitly revalidated.

## 9. Current Presentation-Backed Results

These are the results to use in the current slide deck.

### 9.1 Direct LLM Baseline

- `logs/eval_baseline_deepseek_20260322_13.txt`: `85/100 = 85%`

### 9.2 Simple Retrieve-And-Answer Baseline

- `logs/eval_bm25_baseline_deepseek_20260322_15.txt`: `70/100 = 70%`

### 9.3 Golden Passage Upper Bound

- `logs/eval_golden_deepseek_20260322_13.txt`: `77/100 = 77%`

## 10. What The Current Results Actually Suggest

This is the honest current presentation-backed performance story:

1. The base model is already strong on this benchmark.

- A direct answer baseline reaches `85%`, so BarExam multiple-choice performance is not purely retrieval-limited.

2. Simple one-shot retrieval currently underperforms the direct model baseline.

- The simple retrieve-and-answer baseline is `70%`, which is below the direct baseline.

3. Even perfect single-passage grounding still trails the direct baseline.

- The golden-passage upper bound is `77%`, which suggests that this benchmark rewards model prior knowledge heavily and that single-passage grounding alone is not enough to surpass the base model's performance.

4. The current presentation should avoid over-claiming end-to-end agent performance.

- The refreshed, presentation-backed numbers right now are the three baselines above. The full end-to-end eval story is still being refreshed and should be presented as in progress rather than final.

## 11. Design Considerations Worth Highlighting In Slides

These are the strongest design choices to explain to a PI.

### 11.1 Agentic Decomposition Instead Of Single-Shot RAG

Why:

- legal questions are often multi-issue, jurisdictional, and exception-heavy.

What to say:

- the planner converts a large legal question into targeted doctrinal lookups so retrieval can operate on cleaner sub-problems.

### 11.2 Judge-Gated Control Flow

Why:

- retrieval often returns superficially related text.

What to say:

- instead of trusting retrieval blindly, the system explicitly judges whether each step found useful evidence before moving on.

### 11.3 Deterministic Escalation Path

Why:

- many agent systems overuse LLM control flow and become unstable.

What to say:

- this design keeps failure handling predictable:
  - rewrite first,
  - then web fallback,
  - then direct answer as a last resort.

### 11.4 Hybrid Retrieval

Why:

- legal text needs both exact doctrinal terminology and semantic matching.

What to say:

- BM25 captures exact legal vocabulary.
- dense retrieval covers paraphrase and doctrinal similarity.
- cross-encoder reranking cleans up the candidate pool.

### 11.5 Corpus Routing

Why:

- bar-exam doctrine and housing statutes are very different sources.

What to say:

- routing avoids searching the wrong corpus and is the repo's current multi-domain mechanism.

### 11.6 Evidence-Indexed Final Answer

Why:

- legal presentations need explainability.

What to say:

- the system stores evidence entries across steps and forces the final synthesis to cite them.

## 12. Current Weak Spots / Risks

These should be said carefully but directly.

### 12.1 Historical Artifact Drift

Some logs and traces come from an older architecture with nodes that do not exist in current code, such as:

- injection check
- classifier
- memory writeback
- older replanning flow

Examples:

- `logs/latest_run.txt`
- `case_studies/*.json`
- some `logs/playtest/*` files

These are useful as historical context but should not be presented as the current system unless explicitly labeled as earlier iterations.

### 12.2 Housing Eval Is Not Fully Built Yet

What exists:

- housing data pipeline
- housing collection
- housing routing support

What is missing:

- dedicated housing yes/no evaluation analogous to BarExam QA.

### 12.3 Full End-To-End Eval Refresh Is Still In Progress

- The codebase contains end-to-end and retrieval-oriented eval scripts, but they are not the current presentation-backed result set.
- Before using them in slides, they should be rerun and reviewed against the current runtime.

### 12.4 Provider Comparisons Are Not Yet Standardized

Current logs mix:

- DeepSeek
- DeepSeek Reasoner
- OpenRouter Gemma

Implication:

- some comparisons are apples-to-oranges unless carefully grouped by provider/model/date.

## 13. Recommended Slide Structure

This is the presentation sequence I would use.

### Slide 1: Problem And Goal

- What we are building: an agentic legal research assistant over doctrinal and statutory corpora.
- Why ordinary single-shot QA is not enough for legal reasoning.

### Slide 2: System Snapshot

- 5-node graph.
- one-sentence role for each node.

### Slide 3: Data And Knowledge Sources

- BarExam QA corpus
- HousingQA statutes
- web fallback for current/out-of-corpus questions

### Slide 4: End-To-End Flow

- question in
- router
- planner
- executor
- judge
- replanner
- synthesizer
- final IRAC answer out

### Slide 5: What Happens Inside A `rag_search` Step

- rewrite
- retrieve
- dedup
- rerank
- synthesize
- judge

### Slide 6: Design Choices

- why decomposition
- why hybrid retrieval
- why judge gating
- why escalation path
- why evidence-indexed synthesis

### Slide 7: Example Case Walkthrough

Use one strong example:

- bar doctrinal question,
- or web-search current-law example,
- or housing routing example.

Show:

- planned steps
- retrieved evidence
- early completion if applicable

### Slide 8: Evaluation Inventory

- direct LLM baseline
- simple RAG baseline
- golden upper bound
- note that full end-to-end eval refresh is still in progress

### Slide 9: Current Results

Recommended framing:

- report stored numbers honestly,
- keep the table limited to the three current, validated runs.

Suggested table:

- direct baseline
- simple RAG
- golden upper bound

### Slide 10: Interpretation Of Results

- base model prior knowledge is strong on BarExam MC
- simple one-shot retrieval does not yet beat the direct model
- golden-passage performance shows room to improve how retrieval evidence is used
- housing eval is still missing

### Slide 11: Current Limitations And Next Steps

- unify eval pipeline across bar, housing, web
- align retrieval metrics with planner/rewrite path
- add housing yes/no benchmark
- run clean provider-controlled ablations
- analyze error categories by node

### Slide 12: Takeaway

- strong system architecture and infrastructure are in place
- eval evidence is partial but real
- next phase is refreshing the full eval pipeline and targeted ablations, not rebuilding the whole system

## 14. What I Would Be Comfortable Saying To A PI Right Now

Use something close to this:

"We have a functioning agentic legal RAG stack with collection routing, plan decomposition, hybrid retrieval, judge-gated replanning, and evidence-grounded IRAC synthesis. The data pipeline is in place across doctrinal and statutory corpora. For the current presentation, the cleanest results are three baseline-style evaluations: a direct LLM baseline at 85%, a simple retrieve-and-answer baseline at 70%, and a golden-passage upper bound at 77%. Those numbers show that the benchmark is strongly answerable from model priors and that there is still headroom in how retrieved evidence is being used. The next milestone is refreshing the full end-to-end eval pipeline, especially for housing and full agent performance, rather than redesigning the architecture from scratch."

## 15. Immediate Follow-Up Work That Would Improve The Slides

If we want a stronger presentation deck, the next highest-value additions are:

1. one cleaned-up system diagram exported from the current graph
2. one slide-ready worked example from a current run
3. one standardized result table using a single provider/model
4. one "known caveats" slide so we stay accurate
5. one roadmap slide focused on full eval completion

## 16. Novelty And Retrieval Improvement Ideas

These ideas come from reading the current code, playtest traces, and available eval artifacts.

### 16.1 Make Query Rewriting Aspect-Aware, Not Just Terminology-Aware

Observation:

- The current `query_rewriter` mostly generates synonym-style alternatives.
- In the traces, this often means the system retrieves several very similar passages rather than complementary evidence.

Idea:

- Generate retrieval variants by *role*:
  - governing rule / elements
  - exceptions / defenses / limitations
  - application / procedural implementation

Why this is interesting:

- It is a lightweight way to make the system more agentic without changing the graph.
- It also aligns directly with the idea already sketched in `ideas/parallel_agents.md`.

### 16.2 Add Metadata-Aware Retrieval Constraints For Housing

Observation from `logs/playtest/08_housing_dense_only.txt`:

- The router correctly chooses `housing_statutes`.
- Retrieval then often returns the wrong state even when the question is explicitly about California.
- Cross-encoder scores can still be high for the wrong-jurisdiction passages.

Idea:

- Extract jurisdiction from the question and hard-filter or strongly bias retrieval by `state`.
- At minimum, use metadata filtering for explicit state questions.
- Better: split the housing corpus into per-state collections or build a state-first router.

Why this is high value:

- It directly targets one of the clearest failure modes in the traces.
- It should improve both accuracy and interpretability.

### 16.3 Make Citation-Aware Queries First-Class

Observation:

- Some failed queries in housing mention specific authorities like `CCP 1161`, but the retriever still struggles to land on the right text in the statute corpus.

Idea:

- Add a citation-targeting rewrite mode that emits variants optimized for:
  - statute citations
  - code section numbers
  - named doctrines
  - act names

Possible implementation:

- planner marks a step as `citation_seeking`
- query rewriter generates one citation-heavy query and one natural-language query

Why this is interesting:

- Legal retrieval is unusually citation-sensitive.
- Making citation retrieval explicit would be a domain-specific novelty point.

### 16.4 Reintroduce Source-Aware Balancing In Bar Retrieval

Observation:

- The old reranker A/B script contains source-aware mixing logic between study material and caselaw.
- The current production retriever pools everything together and lets the cross-encoder sort it out.

Idea:

- For `legal_passages`, enforce a small mix such as:
  - 2-3 doctrine passages (`mbe`, `wex`)
  - 2-3 application passages (`caselaw`)

Why it could help:

- Multiple-choice legal QA often benefits from concise doctrinal statements more than long case excerpts.
- This may improve answerability without requiring model changes.

### 16.5 Add Jurisdiction-Aware Confidence

Observation:

- In housing traces, wrong-jurisdiction retrieval can still receive strong cross-encoder scores.
- The current confidence is only the max cross-encoder logit, so it is blind to jurisdiction mismatch.

Idea:

- Replace or augment confidence with features such as:
  - jurisdiction agreement across top-k
  - source diversity
  - query-variant agreement
  - citation hit rate
  - judge verdict history on similar steps

Why this matters:

- It would make the system’s self-assessment more legally meaningful.

### 16.6 Build A Retrieval Failure Taxonomy

Observation:

- The traces show repeatable failure modes:
  - wrong jurisdiction
  - adjacent doctrine instead of target doctrine
  - existence of doctrine retrieved, but not the proof standard or elements
  - procedural rule missing while substantive rule is present

Idea:

- Tag failed steps by failure mode and track them in eval output.
- Use those labels to guide prompt or retriever improvements.

Why this is novel/useful:

- It turns the judge from a binary gate into a source of supervision for system improvement.

### 16.7 Turn Judge Feedback Into Retrieval Supervision

Observation:

- The judge often produces high-quality reasons and rewrite suggestions.

Idea:

- Log judge failure reasons in a structured way.
- Use them to:
  - fine-tune or redesign rewrite prompts
  - train a learned query-selection policy
  - build an offline corpus of failure-rewrite pairs

Why this is interesting:

- It creates a self-improving loop using artifacts the system already generates.

### 16.8 Add A Two-Stage Housing Pipeline

Observation:

- Housing retrieval is currently doing one huge dense search over all statutes.

Idea:

- Stage 1: retrieve candidate states or code families
- Stage 2: retrieve within that narrowed subset

Variants:

- question -> state classifier -> state collection retrieval
- question -> state-filtered BM25 / dense retrieval -> rerank

Why this is strong:

- It is a concrete systems contribution, not just a prompt tweak.
- It addresses the largest corpus with the clearest retrieval pain.

### 16.9 Evaluate Retrieval At The Step Level, Not Just The Question Level

Observation:

- The existing retrieval recall scripts mostly operate on the full question.
- The actual system retrieves on rewritten sub-questions.

Idea:

- Create a step-level eval set:
  - planner step
  - rewritten query
  - expected evidence type / jurisdiction / source family

Why this matters:

- It would align evaluation with the actual agent design.
- It would make retrieval improvements much easier to attribute.

### 16.10 Novelty Framing For The Project

If we want the project to feel more interesting than "legal RAG with planning," the strongest framing is probably:

- **Adaptive legal research over heterogeneous authority sources**
  - doctrinal corpus
  - statute corpus
  - web fallback

combined with:

- **judge-gated escalation**
  - retrieval -> rewrite -> web -> direct answer

and optionally:

- **aspect-specialized multi-query or multi-agent retrieval**
  - rule / exception / application
  - jurisdiction-aware retrieval
  - citation-aware retrieval

That combination feels more distinctive than just "we added a planner."

### 16.11 Highest-Confidence Next Experiments

If the goal is to improve results quickly while also making the project feel more novel, I would prioritize:

1. Housing jurisdiction filtering first.

- This is the clearest failure mode in the traces.
- It should be straightforward because `state` metadata already exists in the corpus.

2. Aspect-aware query rewriting second.

- This directly targets the "right doctrine, wrong sub-aspect" failures in bar-style multihop traces.
- It is also a strong novelty angle because it makes the multi-query retrieval strategy more principled.

3. Citation-aware retrieval variants for statutes third.

- Legal retrieval is unusually citation-sensitive.
- This would make the housing pipeline more domain-specific and more interesting scientifically than generic semantic retrieval.

4. Step-level retrieval evaluation immediately after that.

- Right now, some retrieval metrics are misaligned with the actual planner-plus-rewrite runtime.
- A step-level eval would make future improvements easier to defend in slides and papers.
