# Benchmark and Method Birdseye - 2026-04-30

Purpose: consolidate what we can test now, what each benchmark is actually
probing, how the current method families map onto those probes, and which gaps
should be fixed before spending more API/cluster budget.

## Audit Snapshot

Current branch: `codex/evidence-ledger-router`.

Repo-grounded capability:

- `eval/eval_config.py` exposes 64 harness modes.
- The QA harness supports `barexam`, `housing`, `legal_rag`, `australian`,
  `casehold`, `musique`, and `legalbench_scalr`.
- `scripts/run_retrieval_qrels_eval.py` plus
  `scripts/score_retrieval_qrels.py` cover retrieval-only qrels datasets such
  as MLEB-SCALR.
- Local files are present for BarExamQA, HousingQA, CaseHOLD,
  LegalBench-SCALR, MLEB-SCALR, and MuSiQue.
- Local files are absent for Legal-RAG-QA and Australian Legal QA.
- Local Chroma currently has `mleb_scalr_holdings` and
  `legalbench_scalr_holdings`; local `legal_passages` is empty, so local
  BarExam RAG runs are blocked unless the collection is rebuilt.
- Cluster job `57949` verified usable `housing_statutes`, `legal_passages`,
  `musique_passages`, and `casehold_holdings` collections, but the cluster
  checkout should be pulled to latest before new scripts are launched.

## Benchmark Map

| Benchmark | Local rows/corpus | Current run surface | What it tests best | Current signal | Main caveat |
|---|---:|---|---|---|---|
| BarExamQA | 1,195 QA / 686,324 passages | QA harness; cluster RAG ready, local RAG blocked by empty `legal_passages` | Legal MC, bar-style fact patterns, answer anchoring, parametric legal priors | Top-1/top-5 flat at N=200; full-N `rag_snap_hyde` is the strongest signed-off legal MC lift | Not a clean retrieval-depth benchmark; flatness does not mean retrieval is globally useless |
| HousingQA | 6,853 QA / 1,837,403 statutes | QA harness; cluster RAG ready; local detail logs pulled | Statutory yes/no, jurisdiction/state sensitivity, retrieval depth | top-1 50.5%, top-10 58.0%, 2-call 57.0% at N=200 | Gold-hit and state-match are both low; need metadata/filter audit before method claims |
| MuSiQue | 2,417 QA / 48,315 passages | QA harness; special in-row BM25 retrieval, no Chroma needed | Multi-hop retrieval depth, query formulation, bridge/composition failures | top-1 collapses; `two_call`, `mhd`, and `iter_ptable` are complementary | Not legal-domain evidence; use as mechanism analog, not legal headline |
| LegalBench-SCALR | 571 QA / 1,733 holdings | QA harness; local Chroma ready | Supreme Court holding selection, small candidate-set retrieval | top-1 59.5%, top-5/top-10 77.0%; top-10 adds recall without net accuracy | Best read is candidate-set saturation, not "more k is better" |
| MLEB-SCALR | 120 queries / 523 holdings / 120 qrels | Retrieval-only runner; local Chroma ready | Pure retriever calibration without LLM answer confounds | gte-large Recall@1 34.2%, Recall@5 65.0%, Recall@10 72.5% | Not a QA benchmark; do not mix into answer accuracy tables |
| CaseHOLD | 3,600 test / 51,296 repaired holdings | QA harness; collection rebuild needed after gold mapping repair | Holding-option disambiguation and distractor sensitivity | top-1/top-5/2-call are answer-flat in old N=200 audits | Old `gold_retrieved=0` was instrumentation; rebuild Chroma before new retrieval claims |
| Legal-RAG-QA | Missing locally; downloader exists | QA harness once restored/embedded | Small open-ended criminal-law QA and relevant-passage recall sanity | Not currently runnable locally | Likely too small/easy for headline novelty |
| Australian Legal QA | Missing locally; downloader exists | QA harness once restored/embedded | Open-ended jurisdiction/source grounding over synthetic QA | Not currently runnable locally | Synthetic construction may overreward source-copy behavior |
| LegalBench-RAG | Not wired | Retrieval-only future loader | Legal snippet precision and citation granularity | No local run | Needs span/character-overlap scorer, not document-id qrels only |
| Legal RAG Bench | Not wired | Future end-to-end legal RAG loader | Correctness, groundedness, retrieval accuracy, factorial error decomposition | No local run | Highest-alignment addition, but requires new loader and judge schema |

## Method Family Map

| Family | Representative modes/tools | Best use | Avoid using it for |
|---|---|---|---|
| Direct/no retrieval | `llm_only`, `snap_only_in_final`, `vectorless_*`, `self_verify`, `snap_debate`, `friend_foe_attribution` | Parametric sufficiency, anchoring, attribution, and "do we need retrieval at all?" probes | Retrieval claims |
| Basic retrieval/depth | `rag_simple --retrieval-k`, `rag_rewrite`, `rag_multi_query`, `rag_state_filter`, `golden_passage` | Top-k policy, raw retrieval value, metadata-filtered retrieval, oracle/context-utilization checks | Agentic novelty claims |
| Pseudo-document/query formulation | `rag_hyde`, `rag_snap_hyde`, `rag_snap_hyde_2call`, `multi_hyde_diverse`, `iter_hyde` | Query-formulation bottlenecks and answer-conditioned retrieval | Datasets where top-k is flat or option choice is already saturated |
| Decomposition/agentic | `subagent_*`, `decompose_rag`, `planning_table`, `iterative_planning_table`, `advisor_planning_table` | Multi-hop/composition probes and evidence-state experiments | Blanket "agents help legal QA" claims; MuSiQue shows extra evidence can also harm |
| Routing/verification | `adaptive_snap_route`, `confidence_gated`, `ce_threshold`, `double_snap`, router scripts | Cost-aware escalation and cheap controller tests | Paper claims until leave-one-dataset-out improves |
| Entity/structured search | `entity_search`, `snap_entity_search`, `snap_entity_informed` | BarExam-only structured retrieval alternatives | Non-BarExam datasets; those paths fall back to `llm_only` with `entity_fallback=dataset_not_supported` |
| Retrieval-only calibration | `scripts/run_retrieval_qrels_eval.py`, `scripts/score_retrieval_qrels.py` | MLEB-SCALR, future LegalBench-RAG-style retrieval metrics | Generated-answer comparisons |

## Current Mechanism Picture

The benchmarks now separate into useful regimes:

| Regime | Evidence | Best next action |
|---|---|---|
| Retrieval-depth limited | MuSiQue top-1 collapse; Housing top-1 to top-10 directional lift | Keep k-sweeps and paired disagreement audits; do not rely on N=200 alone for paper-grade significance |
| Query-formulation limited | MuSiQue `two_call` and `multi_hyde_diverse` rescue partially overlapping rows | Test SpecRAG-lite only on MuSiQue/Housing after metadata filtering is checked |
| Candidate-set limited then saturated | SCALR top-1 fails, top-5/top-10 tie, top-10 adds gold-hit without accuracy | Route to a small candidate set, then stop |
| Answer-option/legal-prior dominated | BarExam top-k flatness; snap-style methods can still help | Frame as anchoring/option reasoning, not retrieval-depth |
| Option-disambiguation / instrumentation-limited | CaseHOLD answer flatness plus broken `gold_retrieved` mapping | Fix retrieval instrumentation before another LLM run |
| Pure retrieval calibration | MLEB-SCALR qrels baseline | Run embedding A/B locally before spending LLM budget |

## Housing Metadata Read

The new Housing result should not be simplified to "more legal retrieval helps."

The state-metadata audit shows:

- `rag_simple` top-10 improves accuracy over top-1 by +7.5pp, but average
  same-state retrieval fraction stays tiny: 2.5% at top-1 and 2.3% at top-10.
- `rag_snap_hyde_2call` at k=5 raises any-state-match to 34.0% and average
  state-match fraction to 14.1%, but it still only reaches 57.0%, slightly
  below top-10 `rag_simple`.
- Therefore the top-10 lift is probably not just jurisdiction repair. The
  two-call arm is better at state targeting, but state targeting alone does not
  guarantee answer correctness.

Implication: test explicit state-filtered retrieval before SpecRAG-lite. If a
state filter beats top-10 or cuts context without losing accuracy, the legal
contribution is metadata-aware evidence allocation. If it does not, then the
failure is deeper: statute chunking, legal support utilization, or yes/no bias.

## What We Can Test Now

No new infra:

1. MLEB-SCALR embedding A/B locally: re-embed 523 holdings with alternative
   embedders and score qrels. Current local A/B has `gte-large` ahead of
   `all-MiniLM-L6-v2`; larger downloads are deferred.
2. SCALR full/test-size reruns locally if API budget allows; local collection
   exists.
3. MuSiQue method probes locally because the harness uses per-row retrieval,
   not Chroma.
4. Housing offline audits over pulled logs: state metadata, yes/no bias,
   rescued/harmed rows, gold-hit/state-hit correlation.
5. Router table v2 using existing logs plus Housing features.

Cluster-ready after pulling latest branch:

1. Housing `rag_state_filter` k=5/k=10 via
   `scripts/hpc/slurm_housing_state_filter.sh`.
2. BarExam full-N confirmatory rows if API budget allows.
3. CaseHOLD reruns only after gold-option retrieval mapping is fixed.
4. Larger Housing N if the top-1 to top-10 trend needs significance.

Needs restoration or wiring:

1. Restore/download and embed Legal-RAG-QA.
2. Restore/download and embed Australian Legal QA.
3. Add LegalBench-RAG loader and span/character scorer.
4. Add Legal RAG Bench loader plus correctness/groundedness/retrieval schema.
5. Add SpecRAG-lite draft logging only after a diagnostic gate says it is
   needed.

## Priority Recommendation

Do not build a broad multi-agent system next. The more defensible path is:

1. Add a Housing state-filtered retrieval arm.
2. Build router table v2 with Housing and metadata features.
3. Run MLEB-SCALR embedding A/B as the cheap retriever-calibration side path.
4. Rebuild CaseHOLD Chroma from the repaired gold mapping.
5. Only then implement SpecRAG-lite on the cells where the audit says draft
   diversity should matter: MuSiQue and maybe Housing, not BarExam/CaseHOLD by
   default.

The emerging paper shape is:

> Legal RAG methods fail because the bottleneck moves. The contribution is not
> a bigger agent; it is a diagnostic harness and controller that identifies
> whether the row needs deeper retrieval, metadata-constrained retrieval,
> query reformulation, candidate-set expansion, or no retrieval at all.

## Verification Commands

```bash
git status --short --branch
uv run python utils/fast_embed.py status
python3 - <<'PY'
from eval.eval_config import EVAL_MODES
print(len(EVAL_MODES))
PY
uv run python scripts/audit_housing_metadata_depth.py \
  --log top1=logs/eval_rag_simple_or-gemma4-26b_20260430_0415_detail.jsonl \
  --log top5=logs/eval_rag_simple_or-gemma4-26b_20260430_0502_detail.jsonl \
  --log top10=logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl \
  --log two_call=logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260430_0644_detail.jsonl \
  --out docs/housing_metadata_depth_audit_2026-04-30.md
```
