# HousingQA Metadata/Depth Audit - 2026-04-30

Generated offline from HousingQA detail logs and `datasets/housing_qa/statutes.csv`.

## Headline Read

- `rag_simple` top-10 improves answer accuracy over top-1 by +7.5pp, but average same-state retrieval fraction stays tiny: 2.5% at top-1 and 2.3% at top-10. That means the top-10 lift is not explained by a simple jurisdiction repair story.
- `rag_snap_hyde_2call` retrieves same-state statutes much more often (any-state 34.0%, average state-match fraction 14.1%), but it scores 57.0% versus top-10 `rag_simple` at 58.0%. State targeting helps retrieval diagnostics, but it is not sufficient for answer correctness.
- The next method should be explicit state-filtered retrieval or state-aware reranking, not SpecRAG-lite yet.

## Run-Level Metadata

| Run | N | Accuracy | Gold hit | Docs/q | Top-1 state match | Any state match | All state match | Avg state-match fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top1 | 200 | 50.5% | 1.0% | 1.0 | 2.5% | 2.5% | 2.5% | 2.5% |
| top5 | 200 | 53.5% | 3.5% | 5.0 | 3.5% | 12.5% | 0.0% | 2.9% |
| top10 | 200 | 58.0% | 5.5% | 10.0 | 3.0% | 17.0% | 0.0% | 2.3% |
| two_call | 200 | 57.0% | 9.5% | 5.0 | 15.0% | 34.0% | 2.0% | 14.1% |

## Paired Metadata Deltas

| Baseline | Treatment | N | Acc delta | Rescued | Harmed | Avg state-frac delta | Rescued state-frac delta | Harmed state-frac delta | Any-state F->T / T->F | Gold-hit F->T / T->F |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top1 | top5 | 200 | +3.0pp | 23 | 17 | +0.4pp | +0.9pp | -4.7pp | 20/0 | 5/0 |
| top1 | top10 | 200 | +7.5pp | 38 | 23 | -0.2pp | +1.6pp | -1.7pp | 29/0 | 9/0 |
| top5 | two_call | 200 | +3.5pp | 25 | 18 | +11.2pp | +12.0pp | +13.3pp | 52/9 | 14/2 |
| top10 | two_call | 200 | -1.0pp | 26 | 28 | +11.8pp | +10.8pp | +2.1pp | 49/15 | 13/5 |

## Interpretation

- This audit measures whether retrieved statutes come from the same state as the HousingQA question.
- It does not prove the retrieved statute is legally controlling; same-state retrieval is a metadata sanity signal, not a relevance label.
- If accuracy improves while same-state fraction does not, deeper retrieval is probably adding topical/legal context rather than fixing jurisdiction.
- If rescued rows show large same-state gains, metadata filtering should be tested before heavier draft/verifier methods.

## Provenance

- Statute metadata: `datasets/housing_qa/statutes.csv`
- `top1`: `logs/eval_rag_simple_or-gemma4-26b_20260430_0415_detail.jsonl`
- `top5`: `logs/eval_rag_simple_or-gemma4-26b_20260430_0502_detail.jsonl`
- `top10`: `logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl`
- `two_call`: `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260430_0644_detail.jsonl`
