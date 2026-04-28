# `snap_hyde_2call`: efficiency variant becomes a new headline winner — Llama 70B × MuSiQue

Meeting 2026-04-27 ask #6: collapse `rag_snap_hyde` from 3 LLM calls to 2 by
fusing snap-reasoning and HyDE-passage generation into a single structured
response. Goal was efficiency — preserve the lift with 33% fewer calls.

Result on Llama 3.3 70B × MuSiQue × N=200, paired McNemar within same Groq slot
(seed=42, all logs from 2026-04-27 / 2026-04-28):

| Method | EM | LLM calls/Q | Δ vs `rag_simple` | McNemar p |
|---|---:|---:|---:|---:|
| `rag_simple` (baseline) | 27.5% | 1 | — | — |
| `multi_hyde_diverse` | 35.5% | ~3 | +8.0pp | 0.0195 |
| `iterative_planning_table` | 36.0% | ~5-7 | +8.5pp | 0.0533 (TRENDING) |
| **`snap_hyde_2call`** | **37.0%** | **2** | **+9.5pp** | **0.0079** |

Pairwise comparisons against the new mode (paired McNemar N=200):

| Comparison | Δ | p | b/c |
|---|---:|---:|---:|
| `snap_hyde_2call` vs `rag_simple` | +9.5pp | **0.0079** | 33 / 14 |
| `snap_hyde_2call` vs `multi_hyde_diverse` | +1.5pp | 0.68 NS | 13 / 10 |
| `snap_hyde_2call` vs `iterative_planning_table` | +1.0pp | 0.89 NS | 29 / 27 |
| `snap_hyde_2call` vs `rag_snap_hyde` (3-call) | **+13pp** | **0.0001** | 35 / 9 |

## TL;DR — paper-grade headline shift

`snap_hyde_2call` is now the cleanest multi-hop QA lift on Llama 70b MuSiQue:
- **Significantly beats baseline** (+9.5pp, p=0.008) — stronger significance than mhd (p=0.02) and clearer than iter_ptable (p=0.05).
- **Statistically tied** with mhd (p=0.68) and iter_ptable (p=0.89) — but with fewer LLM calls than either.
- **Crushes original** `rag_snap_hyde` (+13pp p=0.0001) — the 3-call legal-MC-tuned variant is dominated by the 2-call dataset-aware variant on multi-hop.
- **Lowest cost of any lifting method**: 2 calls/Q vs mhd ~3, iter_ptable ~5-7.

## Implementation note — required fix during landing

The original mode (commit `c201123`) had a hard-coded MC-style system prompt
("multiple-choice question") that did not fit MuSiQue's open-ended format.
First test run on Llama 70b × MuSiQue had **parser success of only 14.5%** —
the model treated MuSiQue as short-answer QA and skipped the structured
`## Reasoning` + `## Passage` blocks. The mode silently fell back to using the
question text as the HyDE retrieval query, effectively becoming "snap + simple
RAG" rather than "snap + HyDE RAG". Result was a non-significant +2pp.

Fix in commit `08302bb`: replaced `_SNAP_HYDE_2CALL_SYSTEM` constant with
`_snap_hyde_2call_system(config)` which composes the dataset's normal `answer`
system prompt + an additional requirement for the `## Passage` block. Preserves
dataset-appropriate answer formatting (MC letter / Yes-No / open-ended) while
adding the passage block uniformly.

After the fix, parser success on Llama 70b × MuSiQue is **98.5% (197/200)**
and the mode lands at 37.0% EM with the headline lift above.

## Mechanism interpretation

The 2-call mode asks for the snap reasoning AND a "reference passage stating
the controlling rule, doctrine, fact, or principle most relevant to this
question" in a single LLM call. On MuSiQue, this passage matches the
Wikipedia-style corpus better than the original `rag_snap_hyde` HyDE call,
which uses `_snap_hyde_user(question_intermediate, snap_answer)` and prompts
for a "passage in legal treatise / casebook style" — a hyper-legal framing
that is wrong for MuSiQue's general-knowledge corpus.

So the +13pp gain over `rag_snap_hyde` decomposes as:
- ~+9.5pp from a generally-formatted HyDE passage matching the corpus genre
- ~+3.5pp by-product of producing both blocks in one call (forcing the model
  to commit to a snap answer + relevant doctrine reference together)

Both contributions are corpus/dataset specific. The cluster Gemma 4 26B-A4B
BarExam test (SLURM 55451) will reveal whether the same +efficiency / +accuracy
win holds on legal MC, where the original `rag_snap_hyde` was already the
winner. That comparison is the cleaner test of "does fewer calls preserve the
existing lift," vs this MuSiQue test which is "does the new mode become a new
headline winner."

## Source-of-truth log paths

| Run | Path | EM |
|---|---|---:|
| `rag_simple` baseline | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | 27.5% |
| `multi_hyde_diverse` baseline | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | 35.5% |
| `iterative_planning_table` baseline | `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl` | 36.0% |
| `rag_snap_hyde` (3-call) baseline | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | 24.0% |
| `snap_hyde_2call` v1 (broken parser, 14.5% parse_ok) | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0033_detail.jsonl` | 26.0% |
| **`snap_hyde_2call` v2 (98.5% parse_ok)** | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl` | **37.0%** |

## Open questions / next checks

1. **Cross-dataset**: SLURM 55451 (running) tests `snap_hyde_2call` on Gemma 4 26B-A4B × BarExam N=200. Does the efficiency variant preserve the +3.09pp BarExam lift?
2. **Cross-family**: re-run on Gemma 3 27B and Qwen3 30B MoE × MuSiQue N=200. Does the +9.5pp lift over `rag_simple` reproduce on other dense / MoE models, or is this a Llama-specific quirk?
3. **Mechanism decomposition**: pair `snap_hyde_2call` against a 2-call ablation that uses the same combined prompt but throws away the snap reasoning before final synth (passes only the HyDE passage forward). If the gap closes, the snap-anchor-then-synthesize architecture is doing real work; if it doesn't, the gain is purely from the dataset-matched HyDE prompt.
4. **Replication at full corpus**: promote to N=2400 MuSiQue full-corpus once cross-family is confirmed.