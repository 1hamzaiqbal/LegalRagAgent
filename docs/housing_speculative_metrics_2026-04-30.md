# HousingQA Speculative-RAG Metrics - 2026-04-30

Generated from detail JSONL logs. Metrics are offline and do not call an LLM.

## Headline Takeaways

HousingQA now gives the cleanest legal-domain retrieval-depth signal in the
current matrix:

- `rag_simple` improves from top-1 **50.5%** to top-10 **58.0%** on the same
  N=200 rows.
- The paired top-1 to top-10 lift is **+7.5pp** with McNemar
  `b=38, c=23, p=0.0722`; directional, not paper-grade significance yet.
- top-5 is only **53.5%**, so the useful context appears deeper than the first
  few retrieved statutes.
- `rag_snap_hyde_2call` at k=5 reaches **57.0%**, roughly matching top-10
  `rag_simple` while using half as many retrieved documents but twice the LLM
  calls.
- Gold-hit remains low even as accuracy improves: top-1 **1.0%**, top-5
  **3.5%**, top-10 **5.5%**, and two-call **9.5%**. Treat this as evidence that
  the current Housing gold/proxy mapping is not sufficient by itself for an
  answer-quality explanation.

Working interpretation: unlike BarExam, HousingQA is not flat under retrieval
depth. The next audit should separate pure depth from jurisdiction/state
metadata effects before implementing a heavier SpecRAG-lite arm.

## Paired Tests

| Baseline | Treatment | N | Acc base | Acc treat | Delta | b | c | McNemar p | Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top-1 `rag_simple` | top-5 `rag_simple` | 200 | 50.5% | 53.5% | +3.0pp | 23 | 17 | 0.4296 | [-3.0, +9.5] |
| top-1 `rag_simple` | top-10 `rag_simple` | 200 | 50.5% | 58.0% | +7.5pp | 38 | 23 | 0.0722 | [0.0, +15.0] |
| top-5 `rag_simple` | k=5 `rag_snap_hyde_2call` | 200 | 53.5% | 57.0% | +3.5pp | 25 | 18 | 0.3604 | [-3.0, +10.0] |
| top-10 `rag_simple` | k=5 `rag_snap_hyde_2call` | 200 | 58.0% | 57.0% | -1.0pp | 26 | 28 | 0.8919 | [-8.0, +6.5] |

## Speculative-RAG Metric Mapping

| Speculative RAG metric family | What this report computes now | Gap / caveat |
|---|---|---|
| Answer quality | closed-set accuracy, MuSiQue EM/F1, and free-form gold-answer containment when aliases are logged | Containment is only an automatic proxy; legal open-ended rows still need judge/rubric scoring. |
| Efficiency | average, p50, and p95 latency; LLM calls; input/output token use | Local timings mix API latency and harness overhead, so compare only like-for-like runs. |
| Rationale/context compression | generated pseudo-context tokens versus retrieved evidence tokens | This approximates Speculative RAG rationale-vs-document compression; our logs do not yet separate verifier rationale from HyDE/snap artifacts. |
| Drafting | draft count and speculative-score row coverage | Current modes do not log answer drafts or verifier probabilities, so rhoDraft/rhoSelf-contain/rhoSelf-reflect are not computable yet. |
| Retrieval diagnostics | gold-hit rate, retrieval row rate, empty retrieval, evidence docs/tokens | CaseHOLD gold-hit instrumentation is known untrustworthy in current logs. |

## Run Matrix

| Label | Dataset | Mode | N | Acc | EM | F1 | Contains gold | Gold hit | Evid docs/q | Evid tok/q | Gen ctx tok/q | Gen/Evid | Calls/q | Lat avg/p95 | In tok/q | Out tok/q | Drafts/q | Spec score rows |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| housing_top1 | housing | rag_simple | 200 | 50.5% | - | - | - | 1.0% | 1.0 | 201 | 0 | 0.00 | 1.00 | 10.62/40.21 | 388 | 378 | 0.0 | 0.0% |
| housing_top5 | housing | rag_simple | 200 | 53.5% | - | - | - | 3.5% | 5.0 | 1343 | 0 | 0.00 | 1.00 | 11.91/28.05 | 1920 | 540 | 0.0 | 0.0% |
| housing_top10 | housing | rag_simple | 200 | 58.0% | - | - | - | 5.5% | 10.0 | 2853 | 0 | 0.00 | 1.00 | 11.60/25.53 | 3979 | 604 | 0.0 | 0.0% |
| housing_2call | housing | rag_snap_hyde_2call | 200 | 57.0% | - | - | - | 9.5% | 5.0 | 1517 | 542 | 0.36 | 2.00 | 18.10/51.87 | 2453 | 707 | 0.0 | 0.0% |

## Log Provenance

| Label | Detail log | Hypothesis | Caveat |
|---|---|---|---|
| housing_top1 | `logs/eval_rag_simple_or-gemma4-26b_20260430_0415_detail.jsonl` | - | - |
| housing_top5 | `logs/eval_rag_simple_or-gemma4-26b_20260430_0502_detail.jsonl` | - | - |
| housing_top10 | `logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl` | - | - |
| housing_2call | `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260430_0644_detail.jsonl` | - | - |

## Immediate Wiring Gaps

- Add explicit `answer_drafts` and `draft_rationales` arrays if we implement a Speculative-RAG arm.
- Store verifier logprob-derived scores only when the backend exposes token logprobs; otherwise log a separate `llm_verifier_vote` field and keep it labeled as a proxy.
- Split generated-context logging into `query_pseudo_context`, `reasoning_trace`, and `verifier_rationale` so compression is not overloaded.
- Repair CaseHOLD gold-option retrieval mapping before interpreting gold-hit or recall numbers.
