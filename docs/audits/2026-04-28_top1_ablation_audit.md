# Llama 70B MuSiQue Top-1 Retrieval-Depth Ablation Audit

**2026-04-28**  
**Focus**: top-1 vs top-5 retrieval-depth ablation, Llama 70B Groq x MuSiQue, N=200 paired, seed=42

---

## Executive Summary

Verdict: **APPROVED-WITH-CAVEAT for all four top-1 rows**.

The retrieval-depth plumbing is clean: every top-1 detail row has exactly one `evidence_store` item and exactly one `retrieved_ids` item. Pairing is also clean: each top-1 log has 200 rows and an exact 200-row `idx` intersection with its corresponding top-5 baseline, in the same order.

The caveat is not a flag-plumbing issue. The top-1 condition creates substantially more abstention-style predictions (`Unknown`, `Not Provided`, `None`, etc.) than the top-5 baselines and produced two obvious runaway/truncated final generations. These are answer-quality caveats for the paper, not evidence of wrong dataset, silent fallback, empty retrieval, or malformed retrieval depth.

| Method | Top-1 log | N | Top-1 EM | Data quality | One-line reason |
|---|---|---:|---:|---|---|
| `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` | 200 | 13.0% | MINOR | Retrieval-k proof clean, but 23/200 abstention-like predictions and one runaway/truncated final output. |
| `rag_multi_query` | `logs/eval_rag_multi_query_groq-llama70b_20260428_0029_detail.jsonl` | 200 | 14.0% | MINOR | Retrieval-k proof clean, but 25/200 abstention-like predictions and one runaway/truncated final output. |
| `rag_snap_hyde` | `logs/eval_rag_snap_hyde_groq-llama70b_20260428_0025_detail.jsonl` | 200 | 14.0% | MINOR | Retrieval-k proof clean; no obvious final truncation, but 27/200 abstention-like predictions. |
| `multi_hyde_diverse` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260428_0019_detail.jsonl` | 200 | 19.0% | MINOR | Retrieval-k proof clean; no obvious final truncation, but 20/200 abstention-like predictions. |

---

## Per-Log Counters

Empty-prediction here means the stored `predicted_answer` is an abstention-like string (`Unknown`, `Not Provided`, `Not provided`, `Not specified`, `Not enough information`, or `None`). There were no missing/null `predicted_answer` fields and no blank strings.

| Method | Records | Errors | Empty-like predictions | Empty retrieval | Avg LLM calls/Q | Routed-to markers | Obvious truncated/runaway outputs | `dataset` / `mode` / `provider` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `rag_simple` | 200 | 0 | 23 | 0 | 1.0 | 0 | 1 | `musique` / `rag_simple` / `groq-llama70b` on 200/200 |
| `rag_multi_query` | 200 | 0 | 25 | 0 | 2.0 | 0 | 1 | `musique` / `rag_multi_query` / `groq-llama70b` on 200/200 |
| `rag_snap_hyde` | 200 | 0 | 27 | 0 | 3.0 | 0 | 0 | `musique` / `rag_snap_hyde` / `groq-llama70b` on 200/200 |
| `multi_hyde_diverse` | 200 | 0 | 20 | 0 | 2.0 | 0 | 0 | `musique` / `multi_hyde_diverse` / `groq-llama70b` on 200/200 |

Abstention-like value distributions:

| Method | Distribution |
|---|---|
| `rag_simple` | `Unknown` 16, `Not Provided` 2, `None` 2, `Not specified` 2, `Not provided` 1 |
| `rag_multi_query` | `Unknown` 18, `Not specified` 5, `Not enough information` 1, `Not Provided` 1 |
| `rag_snap_hyde` | `Unknown` 21, `None` 3, `Not Provided` 2, `Not enough information` 1 |
| `multi_hyde_diverse` | `Unknown` 13, `Not Provided` 2, `Not provided` 2, `Not specified` 2, `None` 1 |

Obvious runaway/truncated final generations:

| Method | `idx` | Row | Evidence length | Output tokens | Final length | Stored prediction issue |
|---|---|---:|---:|---:|---:|---|
| `rag_simple` | `3hop1__409517_547811_41132` | 31 | 1 | 4096 | 22708 chars | Repetitive final answer loops on "composer most famously associated"; `predicted_answer` is long reasoning text, counted wrong vs `correct_answer='22'`. |
| `rag_multi_query` | `4hop1__726391_153080_33952_34053` | 55 | 1 | 4152 | 22763 chars | Repetitive final answer loops on "Oh Yeah performer"; `predicted_answer` is long reasoning text, counted wrong vs `correct_answer='1905'`. |

No top-1 log contained a `routed_to` field or a string `routed_to` marker in the audited rows.

---

## Paired-N Cross-Check

Join key: `idx`. Pairing is exact for all four comparisons.

| Method | Top-5 baseline log | Top-1 log | Top-5 N | Top-1 N | Exact `idx` intersection | Same order | Missing from top-1 | Extra in top-1 |
|---|---|---|---:|---:|---:|---|---:|---:|
| `rag_simple` | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` | 200 | 200 | 200 | yes | 0 | 0 |
| `rag_multi_query` | `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl` | `logs/eval_rag_multi_query_groq-llama70b_20260428_0029_detail.jsonl` | 200 | 200 | 200 | yes | 0 | 0 |
| `rag_snap_hyde` | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | `logs/eval_rag_snap_hyde_groq-llama70b_20260428_0025_detail.jsonl` | 200 | 200 | 200 | yes | 0 | 0 |
| `multi_hyde_diverse` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260428_0019_detail.jsonl` | 200 | 200 | 200 | yes | 0 | 0 |

McNemar recomputation from the paired detail logs using `scripts/compute_mcnemar.py`:

| Method | Top-5 EM | Top-1 EM | Delta | Top-1-only / top-5-only discordants | McNemar p |
|---|---:|---:|---:|---:|---:|
| `rag_simple` | 27.5% | 13.0% | -14.5pp | 3 / 32 | 4.176981747e-07 |
| `rag_multi_query` | 29.0% | 14.0% | -15.0pp | 7 / 37 | 5.299581744e-06 |
| `rag_snap_hyde` | 24.0% | 14.0% | -10.0pp | 8 / 28 | 0.001193242962 |
| `multi_hyde_diverse` | 35.5% | 19.0% | -16.5pp | 6 / 39 | 5.417768989e-07 |

---

## `retrieval_k=1` Verification

Critical check: if any record had more than one evidence item or more than one retrieved id, `--retrieval-k 1` plumbing would have failed. No such row exists.

| Method | `len(evidence_store)` distribution | `len(retrieved_ids)` distribution | Rows with evidence/retrieved length > 1 | Proof |
|---|---|---|---:|---|
| `rag_simple` | `{1: 200}` | `{1: 200}` | 0 | 200/200 rows have exactly one evidence item and one retrieved id. |
| `rag_multi_query` | `{1: 200}` | `{1: 200}` | 0 | 200/200 rows have exactly one evidence item and one retrieved id. |
| `rag_snap_hyde` | `{1: 200}` | `{1: 200}` | 0 | 200/200 rows have exactly one evidence item and one retrieved id. |
| `multi_hyde_diverse` | `{1: 200}` | `{1: 200}` | 0 | 200/200 rows have exactly one evidence item and one retrieved id. |

Gold retrieval under top-1, from the stored `gold_retrieved` field:

| Method | `gold_retrieved=True` | `gold_retrieved=False` |
|---|---:|---:|
| `rag_simple` | 94/200 | 106/200 |
| `rag_multi_query` | 96/200 | 104/200 |
| `rag_snap_hyde` | 53/200 | 147/200 |
| `multi_hyde_diverse` | 115/200 | 85/200 |

---

## Input-Token Sanity Check

Top-1 input contexts are materially smaller than the top-5 baselines, as expected if only one retrieved passage is entering the final prompt.

| Method | Top-5 avg input tokens | Top-1 avg input tokens | Reduction |
|---|---:|---:|---:|
| `rag_simple` | 832.6 | 256.4 | 69.2% |
| `rag_multi_query` | 1028.6 | 432.8 | 57.9% |
| `rag_snap_hyde` | 1422.9 | 867.5 | 39.0% |
| `multi_hyde_diverse` | 1033.5 | 395.6 | 61.7% |

LLM call counts stayed method-consistent between top-5 and top-1:

| Method | Top-5 avg calls | Top-1 avg calls |
|---|---:|---:|
| `rag_simple` | 1.0 | 1.0 |
| `rag_multi_query` | 2.0 | 2.0 |
| `rag_snap_hyde` | 3.0 | 3.0 |
| `multi_hyde_diverse` | 2.0 | 2.0 |

---

## Five-Record Sample Audit Per Log

Sample policy: rows 1, 2, 100, 199, and 200 from each top-1 detail log. For each sample, I checked `final_answer`, `predicted_answer`, `correct_answer`, `evidence_store`, `retrieved_ids`, and `gold_retrieved`.

### `rag_simple`

| Row | `idx` | `gold_retrieved` | Evidence/retrieved lengths | `predicted_answer` | `correct_answer` | Note |
|---:|---|---|---|---|---|---|
| 1 | `2hop__121145_561444` | true | 1 / 1 | `Rabbi Menachem Mendel Schneersohn's predecessors` | `Dovber Schneuri` | Wrong answer from a single Chabad evidence passage; dataset and retrieval shape valid. |
| 2 | `2hop__86689_728109` | false | 1 / 1 | `Unknown` | `Oklahoma City Thunder` | Abstention with non-gold NBA Draft passage; no empty retrieval. |
| 100 | `2hop__748182_78303` | false | 1 / 1 | `Not provided in the passage` | `Cordell Walker` | Abstention-like extraction; evidence is South Texas school passage. |
| 199 | `4hop3__524186_219173_548463_72134` | false | 1 / 1 | `1947` | `1839` | Wrong inference from Kolkata evidence; no malformed extraction. |
| 200 | `3hop1__79039_131926_87157` | false | 1 / 1 | `eastward` | `rises in northern Minnesota and meanders slowly southwards` | Wrong directional answer from Lake Michigan/Huron evidence. |

### `rag_multi_query`

| Row | `idx` | `gold_retrieved` | Evidence/retrieved lengths | `predicted_answer` | `correct_answer` | Note |
|---:|---|---|---|---|---|---|
| 1 | `2hop__121145_561444` | false | 1 / 1 | `Hirsch` | `Dovber Schneuri` | Wrong answer from Modern Orthodoxy evidence; no wrong dataset. |
| 2 | `2hop__86689_728109` | false | 1 / 1 | `Not provided in the passage` | `Oklahoma City Thunder` | Valid abstention-like behavior with non-gold NBA Draft evidence. |
| 100 | `2hop__748182_78303` | false | 1 / 1 | `Not provided in the passage` | `Cordell Walker` | Valid extraction; insufficient top-1 evidence. |
| 199 | `4hop3__524186_219173_548463_72134` | false | 1 / 1 | `1911` | `1839` | Wrong inference from Kolkata evidence. |
| 200 | `3hop1__79039_131926_87157` | false | 1 / 1 | `southward` | `rises in northern Minnesota and meanders slowly southwards` | Direction-only partial answer counted wrong by EM; extraction not malformed. |

### `rag_snap_hyde`

| Row | `idx` | `gold_retrieved` | Evidence/retrieved lengths | `predicted_answer` | `correct_answer` | Note |
|---:|---|---|---|---|---|---|
| 1 | `2hop__121145_561444` | true | 1 / 1 | `Rabbi Menachem Mendel Schneersohn's predecessors` | `Dovber Schneuri` | Wrong answer despite gold-retrieved true; single passage lacks enough chain context. |
| 2 | `2hop__86689_728109` | false | 1 / 1 | `Not provided in the passage` | `Oklahoma City Thunder` | Abstention with non-gold NBA Draft evidence. |
| 100 | `2hop__748182_78303` | false | 1 / 1 | `Not provided in the passage` | `Cordell Walker` | Valid abstention-like extraction. |
| 199 | `4hop3__524186_219173_548463_72134` | false | 1 / 1 | `1947` | `1839` | Wrong inference from Kolkata evidence. |
| 200 | `3hop1__79039_131926_87157` | false | 1 / 1 | `either direction` | `rises in northern Minnesota and meanders slowly southwards` | Wrong/underspecified directional answer; no fallback marker. |

### `multi_hyde_diverse`

| Row | `idx` | `gold_retrieved` | Evidence/retrieved lengths | `predicted_answer` | `correct_answer` | Note |
|---:|---|---|---|---|---|---|
| 1 | `2hop__121145_561444` | true | 1 / 1 | `Rabbi Menachem Mendel Schneersohn's predecessors` | `Dovber Schneuri` | Wrong answer with Chabad passage; retrieval shape valid. |
| 2 | `2hop__86689_728109` | false | 1 / 1 | `Not provided in the passage` | `Oklahoma City Thunder` | Valid abstention-like behavior under missing gold evidence. |
| 100 | `2hop__748182_78303` | true | 1 / 1 | `Chuck Norris` | `Cordell Walker` | Gold-retrieved true but answer is actor not character; no extraction failure. |
| 199 | `4hop3__524186_219173_548463_72134` | false | 1 / 1 | `Unknown` | `1839` | Abstention with non-gold Ghaziabad evidence. |
| 200 | `3hop1__79039_131926_87157` | true | 1 / 1 | `South` | `rises in northern Minnesota and meanders slowly southwards` | Semantically close direction-only answer still counted wrong by EM; extraction valid. |

Sample conclusion: all sampled rows are MuSiQue rows with exactly one evidence item, no empty retrieval, and no silent fallback marker. Failures are mainly expected top-1 under-context failures: missing gold evidence, single-passage insufficiency for multi-hop chaining, or EM-strict partial answers.

---

## Paper Caveats

1. **Cite the top-1 ablation as clean for retrieval-depth plumbing, not as a clean answer-quality run.** All four logs prove `retrieval_k=1` was applied, but top-1 materially increases abstention-like outputs.
2. **Top-1 is an under-context stress test.** The single-passage setting often omits the gold chain; `gold_retrieved=True` ranges from 53/200 to 115/200 depending on method.
3. **Two singleton runaway/truncated final outputs exist.** They affect `rag_simple` row `3hop1__409517_547811_41132` and `rag_multi_query` row `4hop1__726391_153080_33952_34053`; both are counted wrong and do not change the paired-N or retrieval-k conclusions.
4. **No evidence of wrong dataset, empty retrieval, silent fallback, or retrieval-k plumbing failure.** All audited rows are `dataset='musique'`, and top-1 evidence/retrieved-id lengths are exactly one on 800/800 rows.

---

## Sign-Off Recommendation

| Method | Audit verdict | Sign-off recommendation |
|---|---|---|
| `rag_simple` top-1 vs top-5 | MINOR | APPROVED-WITH-CAVEAT |
| `rag_multi_query` top-1 vs top-5 | MINOR | APPROVED-WITH-CAVEAT |
| `rag_snap_hyde` top-1 vs top-5 | MINOR | APPROVED-WITH-CAVEAT |
| `multi_hyde_diverse` top-1 vs top-5 | MINOR | APPROVED-WITH-CAVEAT |

