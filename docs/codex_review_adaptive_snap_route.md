# Review: `adaptive_snap_route`

## Update 2026-04-29

Implementation status after `ffdb4a9`: P0 route parsing is fixed with separate `route_parse_ok` and `passage_parse_ok`; P1 routed-to fallback and audit-parity fields are present; P2 prompt framing was changed to a format/type + change-of-answer rule. The SUFFICIENT branch still scores `final_answer = snap_block` by design, relying on the existing dataset extractors, so keep the scoring-target caveat below as a conscious tradeoff rather than an unresolved blocker.

Eval status: adaptive remains exploratory, not paper-facing. V4 routed MuSiQue N=100 to `NEEDS_RETRIEVAL` on 100/100 rows and landed at 26%, while LegalBench-SCALR N=100 routed 95/100 rows to `SUFFICIENT` and landed at 76%. The original summary guard falsely tagged SCALR as `FAILED-EMPTY-RETRIEVAL` because SUFFICIENT rows intentionally have `retrieved_ids=[]`; this was patched after the review to count only retrieval-attempted rows for `adaptive_snap_route`.

1. **[P0] Fix route-parse auditing before paper runs.** `_split_route_snap_hyde` defaults `route = "NEEDS_RETRIEVAL"` (`eval/eval_harness.py:1231-1235`) but later sets `parse_ok = True` solely when the passage parses (`1260-1268`). A malformed route token plus valid passage silently becomes `NEEDS_RETRIEVAL` with `adaptive_parse_ok=True` and no `routed_to`.
2. **[P1] Make the SUFFICIENT `final_answer` answer-only/sanitized.** The early branch scores `out_base["final_answer"] = snap_block` (`1304-1307`). Keep raw text in `snap_answer`, but score a parsed final-answer block/span to avoid rationale or formatting leakage.
3. **[P1] Complete `routed_to` fallback handling.** Current markers only fire when `not parse_ok` (`1313-1315`, `1336-1338`), missing route-token default cases; the SUFFICIENT marker text says `default_to_sufficient`, but the parser never defaults to SUFFICIENT (`1233`).
4. **[P1] Add audit parity fields.** The reference mode logs `hyde_passage_raw`, `hyde_contains_answer_artifact`, `retrieval_queries`, `rerank_query`, evidence, ids, and gold hit (`1475-1492`). Adaptive omits some fields, especially on SUFFICIENT.
5. **[P2] Rebalance the route prompt.** The current wording explicitly says, “When in doubt, choose NEEDS_RETRIEVAL” (`1205-1207`), which is likely too conservative for an adaptive contribution.

## 1. Correctness Bugs

The SUFFICIENT branch does not break the harness’s required retrieval-schema contract: it explicitly writes `evidence_store=[]`, `retrieved_ids=[]`, and `gold_retrieved=False` (`1308-1312`), and the main loop also normalizes those keys for every record (`5292-5298`). It also records `llm_calls_actual=1` (`1307`).

The scoring target is weaker. `final_answer = snap_block` (`1304-1307`) differs from `run_rag_snap_hyde_1call`, which parses after `## Answer` before scoring (`1424-1435`), and from `run_rag_snap_hyde_2call`, whose `final_answer` is a fresh synthesis call (`1471-1476`). For MC/MuSiQue, `_extract_answer` may survive this (`198-213`), but paper logs should not score a raw mixed block.

The parser fallback is incomplete. `parse_ok` means “passage parsed,” not “route and passage parsed” (`1260-1268`). A missing/invalid route token is unobservable if a passage exists. Also, `re.search(...)` scans all `after_route` (`1251`), not just the line before `## Passage`, so passage text could theoretically supply the first route token.

## 2. Prompt-Engineering Quality

The prompt is anchored to the desired output shape, but biased toward retrieval. It defines NEEDS_RETRIEVAL as uncertainty or any “specific external fact ... factual hop” and adds “When in doubt” choose it (`1201-1207`). That is sensible for MuSiQue but can swamp option-disambiguation and reasoning-bottlenecked tasks.

Concrete edit: replace the default with a change-of-answer test: choose NEEDS_RETRIEVAL only if you can name a missing, retrievable fact/rule whose truth could change the answer; choose SUFFICIENT when the prompt/options contain the decisive facts and retrieval would mostly restate them or introduce distractors. Add a one-sentence `## Route Reason` if format budget allows.

## 3. Routing Pathology

**Hypothesis:** 100% NEEDS_RETRIEVAL on MuSiQue may be partly prompt-induced, not purely calibrated confidence. The wording flags “date, name ... factual hop” (`1205-1206`), which describes most MuSiQue questions. To make routing more discriminating, require the model to state the decisive missing fact before selecting NEEDS_RETRIEVAL; if it cannot name one, route SUFFICIENT.

## 4. Missing Audit Fields

For paired McNemar, the harness supplies `idx`, `predicted_answer`, `is_correct`, mode, provider, and dataset (`5245-5264`). For cost, it records measured `llm_calls` (`5255`) and adaptive adds `llm_calls_actual` (`1307`, `1328`).

Missing for paper-grade analysis: `route_decision_reason`, separate `route_parse_ok` and `passage_parse_ok`, `hyde_passage_raw`, `hyde_contains_answer_artifact`, `rerank_query`, and explicit `retrieval_queries=[]` on SUFFICIENT. These are not existing harness requirements, but they are needed for route/cost/leakage audits.

## 5. Faithfulness To Narrative

Not yet faithful to the current paper narrative. The docs frame `snap_hyde_2call` as “fixed-cost and question-scoped” (`docs/paper_narrative_2026-04-28.md:10-10`) and define the method as fixed two-call: call 1 snap+HyDE, retrieve, call 2 synthesize (`28-28`). The prescriptive claim is per-dataset bottleneck typology “measured by retrieval-depth sensitivity” (`24-24`), not per-question self-routing.

`run_adaptive_snap_route` is a plausible new extension, but the narrative must change from fixed-cost method demonstration to variable-cost adaptive policy, and the implementation needs the audit fields above before it can carry that claim.
