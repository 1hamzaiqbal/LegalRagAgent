# Dataset and Metric Consolidation - 2026-04-30

Purpose: consolidate what the current benchmark set is actually testing, what
Speculative RAG contributes as an evaluation frame, and which wiring gaps block
the next research move.

## Primary Sources

- Speculative RAG, ICLR 2025: <https://openreview.net/forum?id=xgQfWbV6Ey>
  and paper PDF <https://proceedings.iclr.cc/paper_files/paper/2025/file/2ea06b52f613716e67458f5ab3fb7558-Paper-Conference.pdf>.
- Zheng et al., "A Reasoning-Focused Legal Retrieval Benchmark", CS&Law 2025:
  <https://arxiv.org/abs/2505.03970>.
- LegalBench SCALR task page:
  <https://hazyresearch.stanford.edu/legalbench/tasks/scalr.html>.
- CaseHOLD project page:
  <https://reglab.stanford.edu/data/casehold-benchmark/>.
- LegalBench paper: <https://arxiv.org/abs/2308.11462>.
- LegalBench-RAG: <https://arxiv.org/abs/2408.10343>.
- Legal RAG Bench, March 2026: <https://arxiv.org/abs/2603.01710>
  and HF blog <https://huggingface.co/blog/isaacus/legal-rag-bench>.
- Open Australian Legal QA:
  <https://huggingface.co/datasets/isaacus/open-australian-legal-qa>.

## What Speculative RAG Gives Us

GROUNDED: Speculative RAG is not mainly "more retrieval." It is a
draft-and-verify decomposition: retrieved documents are clustered, multiple
document subsets are sampled, a smaller drafter generates answer+rationale
drafts in parallel, and a larger verifier selects a draft.

GROUNDED: The paper evaluates answer quality and system cost together:
free-form answer containment for TriviaQA/MuSiQue/PopQA, accuracy for
closed-set PubHealth/ARC-C, latency, and rationale-vs-document verification
cost. It also ablates multi-perspective sampling, random/same-cluster sampling,
rationale generation, and verifier score components.

GROUNDED: Their verifier scores require language-model probabilities:
draft likelihood, self-containment, and self-reflection. Current API detail logs
do not contain token logprobs for these scores.

GROUNDED: The paper's appendix-style analysis also matters for the narrative:
multi-perspective sampling is not a universal multi-hop solution. It is weaker
when the hard part is discovering a bridge entity rather than selecting among
already-retrieved evidence perspectives. In legal QA, the analogous "bridge"
may be a doctrine, exception, jurisdiction, procedural posture, or statutory
cross-reference.

HYPOTHESIZED: For us, Speculative RAG is most valuable as an evaluation
template, not as an immediate implementation target. The useful transferable
questions are:

1. Does generated intermediate context compress useful evidence or just add
   unsupported commitments?
2. Do multiple evidence perspectives rescue different rows, or do they retrieve
   redundant distractors?
3. Can a verifier/routing step select the right intervention cheaply, without
   building a large agentic system first?

Implemented now:

- `eval/eval_metrics.py` computes offline answer quality, containment, EM/F1,
  latency, token use, evidence volume, generated-context volume, compression
  ratio, draft-count hooks, and speculative-score coverage.
- `scripts/build_speculative_metrics_report.py` builds a reusable Markdown
  report from either a manifest or explicit detail logs.
- `docs/speculative_metrics_report_2026-04-30.md` applies the report to the
  current 15-log bottleneck manifest.

Adjacent metric families worth adopting later:

| Metric family | Source | What it adds | Current repo status |
|---|---|---|---|
| RAGAS | <https://arxiv.org/abs/2309.15217> | Reference-free context focus, faithfulness, and answer quality. | Not wired; needs judge prompts or package integration. |
| ARES | <https://aclanthology.org/2024.naacl-long.20/> | Context relevance, answer faithfulness, answer relevance, plus PPI correction with a small human set. | Not wired; useful once we have a stable legal validation slice. |
| RAGChecker | <https://arxiv.org/abs/2408.08067> | Fine-grained retrieval/generation diagnostics with better human-correlation claims than coarse metrics. | Not wired; closest conceptual match to the bottleneck taxonomy. |
| Legal RAG Bench | <https://arxiv.org/abs/2603.01710> | Correctness, groundedness, retrieval accuracy, and factorial error decomposition. | Not loaded; highest-value legal benchmark addition. |

## Dataset Spirit

| Dataset | What it is meant to test | Canonical metric shape | Current interpretation risk |
|---|---|---|---|
| MuSiQue | Multi-hop QA requiring multiple evidence pieces. Speculative RAG uses it as a free-form QA benchmark. | EM/F1 and answer containment; retrieval gold-hit helps diagnose missing hops. | Strong for retrieval-depth and query-formulation mechanisms, but not legal-domain evidence by itself. |
| BarExamQA | Zheng legal RAG benchmark: bar-exam fact patterns with hand-annotated legal passages and MC answers. | Retrieval Recall@k plus downstream MC accuracy. | Downstream gain can be flat if the LLM already has the rule or cannot use the passage; do not equate top-k flatness with retrieval irrelevance globally. |
| HousingQA | Zheng legal RAG benchmark: practical housing statute questions with jurisdiction-specific statutory passages and yes/no answers. | Retrieval Recall@k plus downstream yes/no accuracy. | Critical missing diagnostic slice for us: it is the cleanest "obscure legal knowledge" test and should be less parametric-memory-saturated than BarExam. |
| CaseHOLD | Five-way holding identification from legal citation context. | MC accuracy; retrieval metrics only meaningful if the gold holding is mapped into the corpus. | Current `gold_retrieved=0/200` is instrumentation, not evidence that retrieval failed. Treat current logs as answer-level only. |
| LegalBench-SCALR | Five-way Supreme Court holding selection from question-presented text. | MC accuracy; in our packaged corpus, gold holding retrieval can be meaningful. | Top-5 helps over top-1, but top-10 adds gold hits without answer gain; this is candidate-set saturation, not a generic "more k helps" story. |
| MLEB-SCALR | Retrieval-only packaging of the SCALR family. | qrels metrics such as Recall@k, MRR, nDCG. | Does not fit `eval_harness.py` as QA; should use a retrieval scorer or `eval/run_embedding_comparison.py`. |
| LegalBench-RAG | Legal retrieval benchmark focused on minimal relevant snippets rather than broad document IDs/chunks. | Precise retrieval metrics over expert-annotated snippet targets. | Useful for retrieval-only calibration and citation granularity, not directly an answer-generation benchmark. |
| Legal-RAG-QA | Small open-ended legal QA over 190 passages and 138 questions. | Open-ended answer judge, containment where gold answers are extractive enough, relevant-passage recall. | Existing historical results suggest it may be too easy; use for harness sanity, not headline novelty. `utils/download_new_datasets.py` now writes `gold_idx` from `relevant_passages`, and `eval_harness.py` accepts that fallback for retrieval scoring. |
| Australian Legal QA | GPT-4-synthesized QA from Open Australian Legal Corpus snippets. | Open-ended answer judge/containment and source retrieval. | Synthetic QA may reward source-copy behavior; useful for jurisdiction/source logging and open-ended scoring plumbing. Current downloader maps each QA to a source-passage `gold_idx`. |
| Legal RAG Bench | 2026 end-to-end legal RAG benchmark with 100 complex criminal-law questions and 4,876 passages. | Correctness, groundedness, retrieval accuracy, and factorial error decomposition. | Very aligned with our bottleneck-taxonomy framing; needs a new loader and judge path. |

## What The Current Metrics Say

GROUNDED from `docs/speculative_metrics_report_2026-04-30.md`:

- MuSiQue top-1 collapse is visible in answer quality and retrieval: top-5
  `rag_simple` is 27.5% accuracy / 84.0% gold-hit; top-1 is 13.0% / 47.0%.
- MuSiQue `rag_snap_hyde_2call` spends extra generation but has a plausible
  compression profile: generated context is about 0.79x retrieved-evidence
  tokens and accuracy rises to 37.0%.
- LegalBench-SCALR top-5 and top-10 both land at 77.0% accuracy even though
  gold-hit rises from 54.0% to 63.0%. That is evidence-utilization or
  candidate-saturation, not raw retrieval shortage.
- BarExam top-1/top-5 is flat at N=200, while `two_call` spends much more
  generated context. Treat this as answer-option/legal-prior dominated until
  full-N paired confirmation says otherwise.
- CaseHOLD stays answer-flat and has broken gold-hit instrumentation. Fix
  retrieval mapping before saying anything about recall.

## Research Gap We Can Defend

GROUNDED: Prior adaptive RAG work routes by uncertainty, generation state, or
question complexity. Legal retrieval benchmarks increasingly separate retrieval
from downstream answer quality. Our logs show those two ideas need to be joined:
the same method helps, hurts, or does nothing depending on the active
bottleneck.

HYPOTHESIZED contribution:

> A bottleneck-aware RAG evaluator can predict which intervention is worth
> applying before launching a bigger agent: deeper retrieval, pseudo-document
> query formulation, answer anchoring, contradiction search, or no extra
> retrieval.

This is more interesting than "we built a Speculative RAG variant for legal QA."
Speculative RAG already owns the draft-verifier idea. Our opening is to ask
when draft diversity and verification are warranted, and when they are wasted
because the task is actually option-disambiguation or evidence-utilization
limited.

This also keeps the agentic claim honest relative to recent work:
Adaptive-RAG routes by query complexity; MAIN-RAG uses multi-agent consensus to
filter noisy documents; RAG-Star uses retrieval-augmented verification inside
deliberative search. Our variant should not be "more agents." It should be:
measure the bottleneck, share evidence state, then escalate only to the smallest
intervention that matches the measured failure.

## Wiring Gaps

1. Retrieval-only qrels scorer: needed for MLEB-SCALR and LegalBench-RAG
   style metrics. Add Recall@k, MRR@k, nDCG@k, and snippet-overlap hooks.
2. Legal RAG Bench loader: add corpus/QA downloader, retrieval collection, and
   correctness/groundedness/retrieval-accuracy judge schema.
3. Legal-RAG-QA local restoration: the downloader/harness path is fixed, but
   current local `datasets/legal_rag_qa/` files are absent and must be restored
   before reruns.
4. Speculative draft logging: if we build a mini SpecRAG arm, detail rows need
   `answer_drafts`, `draft_rationales`, `draft_doc_ids`, `selected_draft_idx`,
   and verifier score/proxy fields.
5. Verifier probability support: true Speculative RAG scores need logprobs. If
   unavailable, log a separately named proxy such as `llm_verifier_vote`; do not
   call it `rhoSelf-reflect`.
6. Generated-context typing: split `hyde_passage`, `snap_reasoning`,
   `verifier_rationale`, and `agent_report` so compression is interpretable.
7. CaseHOLD gold mapping: repair before more CaseHOLD retrieval claims.
8. HousingQA local restoration and diagnostic run: current local
   `datasets/housing_qa/questions.csv` is absent. Restore data and verify
   `housing_statutes`, then run k=1/k=5/k=10 plus `two_call`; this is the
   missing legal dataset that should most directly test Zheng's retrieval-helps
   regime.

## Next Experiments

Priority order:

1. HousingQA depth and pseudo-doc slice on cluster: `rag_simple` k=1/k=5/k=10
   plus `rag_snap_hyde_2call`, N=200 first. This tests whether the Zheng
   "retrieval helps obscure statutes" regime matches our bottleneck taxonomy.
2. Retrieval-only scorer for MLEB-SCALR or LegalBench-RAG-mini. This adds a
   pure retrieval axis without another expensive LLM run.
3. Mini SpecRAG probe on MuSiQue and SCALR only: cluster top-k docs, generate
   3 answer+rationale drafts, verify with one model call. Success criterion is
   not raw accuracy alone; it must show row-level selection explains why MuSiQue
   benefits and SCALR does not.
4. Legal RAG Bench loader if we want the most current legal-RAG benchmark
   alignment. It directly supports factorial error decomposition, which is close
   to our proposed bottleneck taxonomy.

Do not prioritize a broad multi-agent system yet. The better near-term path is
an evidence-budgeted diagnostic agent: cheap probes first, bottleneck label
second, targeted intervention third, all written into a shared evidence ledger.
