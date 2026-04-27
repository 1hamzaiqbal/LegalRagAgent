# multi_hyde_diverse Mechanism Decomposition (preliminary, Tier 1 N=100)

## Status
Tier 1 directional finding. N=200 confirmation in flight. NOT a paper claim yet.

## The decomposition

The +12pp Llama 70b mhd lift can be approximately decomposed into two components:
1. Query diversity component
2. HyDE-style answer-bearing passage component

## Llama 3.3 70b dense at N=100
| Method | EM | Δ vs rag_simple | Component contributed |
|---|---:|---:|---|
| rag_simple | 21% | 0pp | Baseline single-query retrieval |
| rag_multi_query | 25% | +4pp | Query diversity component |
| multi_hyde_diverse | 33% | +12pp | Query diversity plus HyDE-style answer-bearing passage component; +8pp beyond rag_multi_query |

## Llama 4 Scout 17b MoE at N=100
| Method | EM | Δ vs rag_simple | Component contributed |
|---|---:|---:|---|
| rag_simple | 30% | 0pp | Baseline single-query retrieval |
| rag_multi_query | 25% | -5pp | Query diversity alone hurts at this capacity |
| multi_hyde_diverse | 29% | -1pp | HyDE-style answer-bearing passages recover most of the multi-query loss; approximately flat vs baseline within Tier 1 noise |

## Mechanism interpretation
- At sufficient capacity (Llama 3.3 70b), query diversity contributes approximately +4pp over rag_simple, and HyDE-style answer-bearing passages contribute an additional approximately +8pp on top, producing the observed +12pp mhd lift. At insufficient capacity (Llama 4 Scout 17b MoE), query diversity actively HURTS (-5pp), likely because the model is overwhelmed by retrieving more candidates without the synthesis ability to pick the right one. mhd's HyDE-style passages rescue this back to approximately flat by anchoring retrieval to plausible answer-bearing text.
- The implementation difference is exactly the intended decomposition: `run_rag_multi_query` generates two diverse question rewrites, forbids answer selection, pools them with the raw question for retrieval, and answers once (`eval/eval_harness.py:4420-4495`). `run_multi_hyde_diverse` generates three diverse HyDE-style hypothetical answer-passages that target different plausible answers/entities/aspects, pools those passages with the raw question for retrieval, and answers once (`eval/eval_harness.py:1060-1130`). That design choice, 3 diverse HyDE candidates with DIFFERENT entities pooled with the raw question, is what lets the larger model exploit both axes.

## Caveats
- Tier 1 only; Gemma 27B rag_simple jumped 22% at N=100 to 28.5% at N=200, so direction can flip.
- Scout's 25% rag_multi_query and 29% mhd may both be flat-vs-30% rag_simple within noise; await N=200.
- Gemma 27B currently has rag_simple 22% at N=100, rag_simple 28.5% at N=200, and mhd 30% at N=100; rag_multi_query for Gemma 27B is not yet logged.
- The additive decomposition is an interpretive approximation, not yet an identified causal mechanism. The data are also consistent with an interaction-dominated mechanism in which diversity helps mainly when the extra retrieval surfaces are answer-bearing passages rather than question rewrites.

## Follow-ups needed before any paper citation
- mhd × Llama 70b N=200 (currently N=100 only)
- rag_multi_query × Llama 70b N=200 (currently N=100 only)
- rag_multi_query × Scout N=200 (running)
- mhd × Gemma 27B N=200 (running)
- rag_multi_query × Gemma 27B N=100/200 (not yet tested)

## Why this is a strong story IF it survives N=200
- Cleanly decomposes the mhd lift into known mechanisms (diversity + HyDE).
- Explains capacity floor: low-capacity models can't use diversity; HyDE rescue is necessary.
- Makes mhd a robust default (always at least flat) rather than a fragile one.
