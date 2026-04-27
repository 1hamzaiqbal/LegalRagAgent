# Compiled Results — paper-grade, audit-referenced

## Update 2026-04-27 ~12:30 CDT

Change reason: housekeeping sweep after `docs/signoff_log.md` promoted the final Llama 70b N=200 MuSiQue matrix. This update supersedes older lower-file notes that still treated `subagent_rag` as unaudited or `rag_multi_query` as pending: `subagent_rag` is now a signed significant negative (15.5%, -12.0pp, p=0.0007), and Llama `rag_multi_query` is 29.0% (+1.5pp, p=0.728 NS). Current HEAD: `44427ad`.

Last updated: 2026-04-27 ~12:30 CDT
Branch: hpc-setup, HEAD: 44427ad
Source-of-truth for cited numbers: docs/audit_log.md (post-fix authoritative for BarExam Tier 3),
docs/mcnemar_2026-04-27.md (paired tests), logs/experiments.jsonl (raw run summaries).

Notation: `exp row` gives the `logs/experiments.jsonl` run-id timestamp prefix when present. `post-fix re-scored` means the value is verified from the detail log plus `docs/audit_log.md`, but no matching `experiments.jsonl` row exists for that post-fix detail log.

## Section 1 — Tier 3 / Full corpus (cite-able for paper)

### 1.1 BarExam cross-size rag_snap_hyde lift

| Model | rag_simple | rag_snap_hyde | Lift | n | Detail logs | exp row | Audit doc | Commit | Audit verdict |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Gemma 4 26B-A4B | 933/1195 = 78.08% | 970/1195 = 81.17% | +3.09pp | 1195 | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md`; `docs/signoff_log.md` | audit/result commit `8bbf0e7`; extractor audit `ed15eb7` | MINOR — `rag_simple` has 2 sampled null predictions with empty retrieval; exact-gold retrieval is low but evidence is otherwise present |
| Gemma 4 E4B | 699/1195 = 58.49% | 743/1195 = 62.18% | +3.69pp | 1195 | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md`; `docs/signoff_log.md` | audit/result commit `8bbf0e7`; extractor audit `ed15eb7` | MINOR — sampled records are parse-clean; full scan found 1 raw null `rag_snap_hyde` prediction and low exact-gold retrieval |

#### Audit — Gemma 4 26B-A4B rag_snap_hyde lift (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags in sampled `rag_simple`/`rag_snap_hyde` records |
| Empty/garbage pred | 2/15: `mbe_0` and `mbe_1` have `predicted_answer=null` in `eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text, not as hidden snap-stage echo |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled IDs, with non-empty evidence except the two `rag_simple` null-start records |
| Empty retrieval | 13.3% empty: 2/15 sampled IDs empty in `rag_simple`; `rag_snap_hyde` sample had non-empty retrieval |
| Format anomalies | 0/15: expected `predicted_answer`, `correct_answer`, and `idx`/`record_id` fields present; schema uses `correct_answer` as the gold field |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — isolated startup null predictions/empty retrieval in the baseline detail log; cite with that caveat, not a reason to drop the row.

#### Audit — Gemma 4 E4B rag_snap_hyde lift (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags in sampled `rag_simple`/`rag_snap_hyde` records |
| Empty/garbage pred | 0/15 sampled; full scan found 1 raw null parsed prediction at `mbe_95` in `eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled IDs, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required IDs empty |
| Format anomalies | 0/15: expected `predicted_answer`, `correct_answer`, and `idx`/`record_id` fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — sample is clean, but the raw detail log has one non-sampled null parsed prediction and the usual BarExam exact-gold retrieval caveat.

Discrepancy to preserve: `logs/experiments.jsonl` has older full-corpus rows for analogous BarExam runs (for example E4B `20260421_0812`/`20260421_1402` and 26B `20260421_1615`/`20260421_2234`) whose percentages differ. Do not silently substitute those rows for the post-fix values above; cite the rows above as `post-fix re-scored`.

### 1.2 Other Tier 3 BarExam method coverage on Gemma 4 26B-A4B

| Mode | EM | n | Detail log | exp row | Audit doc | Commit | Audit verdict |
|---|---:|---:|---|---|---|---|---|
| `golden_passage` | 940/1195 = 78.66% | 1195 | `logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | CLEAN |
| `llm_only` | 953/1195 = 79.75% | 1195 | `logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | CLEAN |
| `rag_hyde` | 943/1195 = 78.91% | 1195 | `logs/eval_rag_hyde_cluster-vllm_20260425_2240_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — exact-gold retrieval is low in BarExam sample/full scan, but evidence is non-empty |
| `rag_simple` | 933/1195 = 78.08% | 1195 | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — 2/15 sampled null predictions also have empty retrieval |
| `rag_snap_hyde` | 970/1195 = 81.17% | 1195 | `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — low exact-gold retrieval, no empty retrieval or parser leak in sample |
| `snap_only_in_final` | 963/1195 = 80.59% | 1195 | `logs/eval_snap_only_in_final_cluster-vllm_20260426_0154_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | CLEAN |
| `subagent_hybrid` | 887/1195 = 74.23% | 1195 | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0254_detail.jsonl` | absent; post-fix re-scored; raw detail stored 886/1195 = 74.14% | `docs/audit_log.md` | audit/result `8bbf0e7`; extractor `ed15eb7` | MINOR — full scan found low-rate null/truncated parsed outputs and 8 empty retrieval records |
| `subagent_rag` | 934/1195 = 78.16% | 1195 | `logs/eval_subagent_rag_cluster-vllm_20260425_2234_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — full scan found 8 empty retrieval records; sampled records had evidence but low exact-gold retrieval |

#### Audit — `golden_passage` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo; this mode has no snap stage |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; exact gold is injected by design |
| Empty retrieval | skipped: `golden_passage` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — `llm_only` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo; this mode has no snap stage |
| routed_to fallbacks | 0/15 triggered: no `routed_to` |
| Empty retrieval | skipped: `llm_only` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — `rag_hyde` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — exact-gold retrieval is low on BarExam, but sampled evidence is non-empty and there is no parser/sanitizer issue.

#### Audit — `rag_simple` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 2/15: `mbe_0` and `mbe_1` have `predicted_answer=null` |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records |
| Empty retrieval | 13.3% empty: 2/15 sampled retrieval-required records empty, matching the null predictions |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — two sampled startup records are null/empty-retrieval failures already counted as misses; cite with caveat.

#### Audit — `rag_snap_hyde` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low exact-gold retrieval is a BarExam retrieval caveat; sample is otherwise clean.

#### Audit — `snap_only_in_final` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): snap visibility is intentional for this ablation; no hidden leakage beyond the mode design |
| routed_to fallbacks | 0/15 triggered: no `routed_to` |
| Empty retrieval | skipped: `snap_only_in_final` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — `subagent_hybrid` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 sampled; full scan found 1 abrupt final tail at `mbe_1009` |
| Empty/garbage pred | 0/15 sampled; full scan found 2 null parsed predictions (`mbe_1009`, `mbe_1146`) |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 15/15 sampled records, with evidence present |
| Empty retrieval | 0.0% in sample; full scan found 8/1195 empty retrieval records |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate raw parser/retrieval artifacts remain, consistent with the existing post-fix re-score caveat.

#### Audit — `subagent_rag` Gemma 4 26B-A4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15 sampled; full scan found no null parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 15/15 sampled records, with evidence present |
| Empty retrieval | 0.0% in sample; full scan found 8/1195 empty retrieval records |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — sampled outputs are clean, but full-log empty retrieval and low exact-gold retrieval should stay caveated.

### 1.3 Other Tier 3 BarExam method coverage on Gemma 4 E4B

| Mode | EM | n | Detail log | exp row | Audit doc | Commit | Audit verdict |
|---|---:|---:|---|---|---|---|---|
| `rag_simple` | 699/1195 = 58.49% | 1195 | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — low exact-gold retrieval in BarExam, no sampled parser issue |
| `rag_hyde` | 724/1195 = 60.59% | 1195 | `logs/eval_rag_hyde_cluster-vllm_20260426_0714_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — low exact-gold retrieval in BarExam, no sampled parser issue |
| `rag_snap_hyde` | 743/1195 = 62.18% | 1195 | `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — full scan found 1 raw null parsed prediction; sample otherwise clean |
| `snap_hyde_report` | 726/1195 = 60.75% | 1195 | `logs/eval_snap_hyde_report_cluster-vllm_20260426_1713_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — low exact-gold retrieval in BarExam, no sampled parser issue |
| `snap_only_in_final` | 691/1195 = 57.82% | 1195 | `logs/eval_snap_only_in_final_cluster-vllm_20260426_1512_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | CLEAN |
| `subagent_hybrid` | 703/1195 = 58.83% | 1195 | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0545_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — sampled records had evidence but low exact-gold retrieval |
| `subagent_hyde` | 719/1195 = 60.17% | 1195 | `logs/eval_subagent_hyde_cluster-vllm_20260426_1739_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — sampled records had evidence but low exact-gold retrieval |
| `subagent_rag` | 728/1195 = 60.92% | 1195 | `logs/eval_subagent_rag_cluster-vllm_20260426_0545_detail.jsonl` | absent; post-fix re-scored | `docs/audit_log.md` | audit/result `8bbf0e7` | MINOR — sampled records had evidence but low exact-gold retrieval |

#### Audit — `rag_simple` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — exact-gold retrieval is low on BarExam, but sampled evidence is non-empty and parser fields are clean.

#### Audit — `rag_hyde` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): option letters appear only as normal multiple-choice analysis/final answer text |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — exact-gold retrieval is low on BarExam; sample is otherwise clean.

#### Audit — `rag_snap_hyde` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15 sampled; full scan found 1 raw null parsed prediction at `mbe_95` |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — one non-sampled raw parser miss remains; the sampled record set is clean aside from low exact-gold retrieval.

#### Audit — `snap_hyde_report` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — exact-gold retrieval is low on BarExam; sample is otherwise clean.

#### Audit — `snap_only_in_final` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): snap visibility is intentional for this ablation; no hidden leakage beyond the mode design |
| routed_to fallbacks | 0/15 triggered: no `routed_to` |
| Empty retrieval | skipped: `snap_only_in_final` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — `subagent_hybrid` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 15/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — sampled outputs are clean, but exact-gold retrieval remains low.

#### Audit — `subagent_hyde` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 15/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — sampled outputs are clean, but exact-gold retrieval remains low.

#### Audit — `subagent_rag` Gemma 4 E4B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 15/15 sampled records, with evidence present |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — sampled outputs are clean, but exact-gold retrieval remains low.

## Section 2 — Tier 2 / N=200 paired-McNemar (cite-able for paper)

### 2.1 Llama 70b MuSiQue method matrix (TODAY)

| Mode | EM | n | Δ vs rag_simple | McNemar p | Detail log | Commit | Audit verdict |
|---|---:|---:|---:|---:|---|---|---|
| `rag_simple` | 55/200 = 27.5% | 200 | baseline | — | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp row `20260427_0952`; doc `docs/mcnemar_2026-04-27.md` | row `31e69db`; result doc `6b58ddb` | MINOR — 1/20 sampled `predicted_answer='None'`; full scan 6/200 |
| `multi_hyde_diverse` | 71/200 = 35.5% | 200 | +8.0pp | 0.0195 SIG | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`; exp row `20260427_1010`; doc `docs/mcnemar_2026-04-27.md` | row `31e69db`; result doc `3ab2f51`/`6b58ddb` | MINOR — 1/20 sampled `predicted_answer='None'`; full scan 3/200 plus 1 routed fallback |
| `rag_multi_query` | 58/200 = 29.0% | 200 | +1.5pp | 0.728 NS | `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl`; exp row `20260427_1112`; doc `docs/mcnemar_2026-04-27.md` | row `75e8038`; result doc `75e8038`/`44427ad` | CLEAN enough for mechanism: diversity alone is non-significant |
| `rag_snap_hyde` | 48/200 = 24.0% | 200 | -3.5pp | 0.36 NS | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl`; exp row `20260427_1019`; doc `docs/mcnemar_2026-04-27.md` | row `3ab2f51`; result doc `21e687a` | MINOR — 1/20 sampled `predicted_answer='None'`; full scan 2/200 |
| `iter_hyde` | 49/200 = 24.5% | 200 | -3.0pp | 0.47 NS | `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl`; exp row `20260427_1036`; doc `docs/mcnemar_2026-04-27.md` | row `21e687a`; result doc `6b58ddb` | CLEAN |
| `subagent_rag` | 31/200 = 15.5% | 200 | -12.0pp | 0.0007 SIG negative | `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl`; exp row `20260427_1044`; doc `docs/mcnemar_2026-04-27.md` | row `6b58ddb`; result doc `75e8038`/`44427ad` | APPROVED as negative; see signoff log Section B |

#### Audit — `rag_simple` Llama 70b MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: `2hop__748182_78303` parsed as `None`; full scan found 6/200 |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 3/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `Answer: None` parsing, already counted as misses; retrieval shape is otherwise healthy.

#### Audit — `multi_hyde_diverse` Llama 70b MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: `2hop__374767_152023` parsed as `None`; full scan found 3/200 |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 sampled; full scan found 1/200 `single_hyde_fallback_only_2_passages` at `2hop__28287_89399` |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — paper-headline row has low-rate `None` predictions and one explicit fallback in the full log, but no sample truncation, think leakage, or empty retrieval.

#### Audit — `rag_snap_hyde` Llama 70b MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: `2hop__748182_78303` parsed as `None`; full scan found 2/200 |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 7/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` predictions; no empty retrieval or hidden fallback in sample/full scan.

#### Audit — `iter_hyde` Llama 70b MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/20: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 2/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

### 2.2 Gemma 3 27B MuSiQue (cross-family check, NULL'd)

| Mode | EM | n | Δ vs rag_simple | McNemar p | Detail log | Commit | Audit verdict |
|---|---:|---:|---:|---:|---|---|---|
| `rag_simple` | 57/200 = 28.5% | 200 | baseline | — | `logs/eval_rag_simple_or-gemma27b_20260427_0309_detail.jsonl`; exp row `20260427_0309`; doc `docs/mcnemar_2026-04-27.md` | row `c8bcd05`; result doc `83fb2fc` | CLEAN |
| `multi_hyde_diverse` | 62/200 = 31.0% | 200 | +2.5pp | 0.5901 NS | `logs/eval_multi_hyde_diverse_or-gemma27b_20260427_0404_detail.jsonl`; exp row `20260427_0404`; doc `docs/mcnemar_2026-04-27.md` | row `a3aee05`; result doc `83fb2fc` | MINOR — full scan found 1 truncated/abrupt parsed answer outside sample |

#### Audit — `rag_simple` Gemma 3 27B MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/20: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 2/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — `multi_hyde_diverse` Gemma 3 27B MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 sampled; full scan found 1 abrupt/truncated parsed answer at `4hop1__726391_153080_33952_33939` |
| Empty/garbage pred | 0/20: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 3/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — isolated full-log truncation outside the sample; no empty retrieval or fallback issue.

### 2.3 Llama 4 Scout MuSiQue baseline confirmations (sub-Tier 2 — kept for context but per user, dropping Scout going forward)

| Mode | EM | n | Interpretation | Detail log | Commit | Audit verdict |
|---|---:|---:|---|---|---|---|
| `rag_simple` | 30/100 = 30.0% | 100 | Tier 1 context; paired with N=100 mhd | `logs/eval_rag_simple_groq-scout_20260427_0246_detail.jsonl`; exp row `20260427_0246`; doc `docs/signoff_log.md` | row `46fe19b` | MINOR — full scan found 2/100 `None` predictions outside sample |
| `multi_hyde_diverse` | 29/100 = 29.0% | 100 | Tier 1 flat vs baseline | `logs/eval_multi_hyde_diverse_groq-scout_20260427_0249_detail.jsonl`; exp row `20260427_0249`; doc `docs/signoff_log.md` | row `46fe19b` | MINOR — 1/20 sampled `None`; full scan 3/100 |
| `rag_simple` | 60/200 = 30.0% | 200 | Tier 2 baseline context | `logs/eval_rag_simple_groq-scout_20260427_0459_detail.jsonl`; exp row `20260427_0459`; doc `docs/signoff_log.md` | row `6b7a922` | MINOR — full scan found 3/200 `None` predictions outside sample |
| `rag_multi_query` | 61/200 = 30.5% | 200 | Multi-query N=100 dip was noise | `logs/eval_rag_multi_query_groq-scout_20260427_0332_detail.jsonl`; exp row `20260427_0332`; doc `docs/signoff_log.md` | row `a3aee05` | MINOR — 1/20 sampled `None`; full scan 3/200 |

#### Audit — Llama 4 Scout `rag_simple` MuSiQue N=100 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/20 sampled; full scan found 2/100 `None` predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 3/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__84937_21969`, `3hop1__823336_228453_86925`, `2hop__748182_78303` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` outputs outside the sample; no retrieval/fallback issue in sample.

#### Audit — Llama 4 Scout `multi_hyde_diverse` MuSiQue N=100 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: `2hop__364489_861485` parsed as `None`; full scan found 3/100 |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 4/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__84937_21969`, `3hop1__823336_228453_86925`, `2hop__748182_78303` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` outputs; retrieval sample is non-empty and no fallback marker appears.

#### Audit — Llama 4 Scout `rag_simple` MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/20 sampled; full scan found 3/200 `None` predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 2/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` outputs outside the sample; no retrieval/fallback issue in sample.

#### Audit — Llama 4 Scout `rag_multi_query` MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: `2hop__435184_84856` parsed as `None`; full scan found 3/200 |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 3/20 sampled records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` outputs; retrieval sample is non-empty and no fallback marker appears.

## Section 3 — Mechanism decomposition (preliminary, multi-source verified)

| Model | rag_simple N=200 | rag_multi_query N=200 | mhd N=200 | Diversity comp | HyDE comp | Direct refs | Audit verdict |
|---|---:|---:|---:|---:|---:|---|---|
| Llama 70b | 27.5% (`20260427_0952`) | 29.0% (`20260427_1112`) | 35.5% (`20260427_1010`) | +1.5pp observed; p=0.728 NS | +6.5pp residual from answer-passage HyDE | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl`; `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`; docs `docs/mcnemar_2026-04-27.md`; commits `31e69db`, `75e8038`, `3ab2f51`, `44427ad` | Mechanism now Tier 2: diversity alone NS, MHD SIG |
| Gemma 3 27B | 28.5% (`20260427_0309`) | 28.5% (`20260427_0536`) | 31.0% (`20260427_0404`) | 0.0pp observed | +2.5pp observed, but McNemar p=0.5901 NULL | `logs/eval_rag_simple_or-gemma27b_20260427_0309_detail.jsonl`; `logs/eval_rag_multi_query_or-gemma27b_20260427_0536_detail.jsonl`; `logs/eval_multi_hyde_diverse_or-gemma27b_20260427_0404_detail.jsonl`; doc `docs/mcnemar_2026-04-27.md`; commits `c8bcd05`, `0d51b36`, `a3aee05`, `83fb2fc` | MINOR — full scan found one truncated mhd answer outside sample |
| Llama 4 Scout | 30.0% (`20260427_0459`) | 30.5% (`20260427_0332`) | pending at N=200; N=100 was 29.0% (`20260427_0249`) | +0.5pp observed at N=200 | pending | `logs/eval_rag_simple_groq-scout_20260427_0459_detail.jsonl`; `logs/eval_rag_multi_query_groq-scout_20260427_0332_detail.jsonl`; `logs/eval_multi_hyde_diverse_groq-scout_20260427_0249_detail.jsonl`; doc `docs/signoff_log.md`; commits `6b7a922`, `a3aee05`, `46fe19b` | MINOR — sampled `None` outputs; N=200 mhd audit deferred until run exists |
| Qwen3 30B MoE | N=100 only: 24.0% (`20260427_0334`) | pending | N=100 only: 28.0% (`20260427_0448`) | pending | N=100 +4pp total; not decomposed | `logs/eval_rag_simple_or-qwen3-30b-moe_20260427_0334_detail.jsonl`; `logs/eval_multi_hyde_diverse_or-qwen3-30b-moe_20260427_0448_detail.jsonl`; doc `docs/signoff_log.md`; commit `a3aee05` | MINOR — 2/20 sampled empty/null predictions and 1 sampled empty retrieval |

#### Audit — Llama 70b mechanism row (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 sampled; full scan found 1 abrupt `rag_multi_query` N=100 answer at `4hop1__88342_49853_128008_46748` |
| Empty/garbage pred | 2/20 sampled: `multi_hyde_diverse` `2hop__511296_2684` and `rag_simple` `2hop__748182_78303` parsed as `None`; full scan 11/500 artifacts |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 sampled; full scan found 1 `multi_hyde_diverse` fallback (`single_hyde_fallback_only_2_passages`) |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — mechanism row is usable as preliminary mechanism evidence, but the pending Llama N=200 `rag_multi_query` run remains unaudited until it completes.

#### Audit — Gemma 3 27B mechanism row (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 sampled; full scan found 1 abrupt/truncated `multi_hyde_diverse` answer at `4hop1__726391_153080_33952_33939` |
| Empty/garbage pred | 0/20: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 6/20 sampled IDs across cited logs |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — one full-log truncation outside the sample; no empty retrieval or fallback issue.

#### Audit — Llama 4 Scout mechanism row (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 2/20 sampled: `multi_hyde_diverse` `2hop__364489_861485` and `rag_multi_query` `2hop__435184_84856` parsed as `None`; full scan 9/500 artifacts |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 7/20 sampled IDs across cited logs |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — low-rate `None` outputs; N=200 mhd remains deferred because no completed detail log is cited.

#### Audit — Qwen3 30B MoE mechanism row (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 2/20 sampled: `4hop3__39836_29339_508306_70744` null and `3hop2__92991_89854_76291` empty string in `multi_hyde_diverse`; full scan 4/200 artifacts |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 6/20 sampled IDs across cited logs |
| Empty retrieval | 5.0% empty: 1/20 sampled retrieval-required records empty (`4hop3__39836_29339_508306_70744`) |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__84937_21969`, `3hop1__823336_228453_86925`, `2hop__748182_78303` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — Qwen N=100 mechanism evidence has low-rate empty/null outputs and one sampled empty retrieval; keep as directional only.

Treat this section as mechanism evidence, not a final causal decomposition.

## Section 4 — Friend/foe attribution probe

Brief: 4/30 Gemma + 6/30 Llama outcome changes; reference `docs/friend_foe_bias_analysis_2026-04-27.md`.

| Model | Accuracy | Kept snap self/foe/control | Outcome changes | Detail log | exp row | Commit | Audit verdict |
|---|---:|---:|---:|---|---|---|---|
| Gemma 3 27B | 3/30 = 10.0% | 27/30 / 27/30 / 27/30 | 4/30 = 13.3% | `logs/eval_friend_foe_attribution_or-gemma27b_20260427_0249_detail.jsonl`; doc `docs/friend_foe_bias_analysis_2026-04-27.md` | `20260427_0249` | row `46fe19b`; analysis doc `6b7a922` | CLEAN |
| Llama 70b | 4/30 = 13.3% | 25/30 / 25/30 / 22/30 | 6/30 = 20.0% | `logs/eval_friend_foe_attribution_groq-llama70b_20260427_0305_detail.jsonl`; doc `docs/friend_foe_bias_analysis_2026-04-27.md` | `20260427_0305` | row `393e12f`; analysis doc `6b7a922` | CLEAN |

#### Audit — Friend/foe Gemma 3 27B (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/15 triggered: no `routed_to` |
| Empty retrieval | skipped: `friend_foe_attribution` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__704217_82341`, `4hop1__88342_49853_128008_46748`, `3hop1__654651_55349_651302` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

#### Audit — Friend/foe Llama 70b (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15: no empty, `None`, placeholder, or whitespace-only parsed predictions |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/15 triggered: no `routed_to` |
| Empty retrieval | skipped: `friend_foe_attribution` is in `_NO_CHROMA_MODES` |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__704217_82341`, `4hop1__88342_49853_128008_46748`, `3hop1__654651_55349_651302` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CLEAN

## Section 5 — Negative findings (citeable as 'method does not transfer')

| Finding | Verified value | Citation status | Direct refs | Audit verdict |
|---|---|---|---|---|
| `iter_hyde` hurts/underperforms small-model MuSiQue settings | Gemma 27B: `iter_hyde` 2/30 = 6.7%; Scout: 5/30 = 16.7%; Qwen3 30B MoE: 2/30 = 6.7% | Direction only; not a definitive result. The requested -13 to -17pp statement uses mixed N=100 baselines for Scout/Qwen and Gemma; same-N Gemma N=30 baseline gives -20.0pp. | Detail logs `logs/eval_iter_hyde_or-gemma27b_20260427_0034_detail.jsonl`, `logs/eval_iter_hyde_groq-scout_20260427_0320_detail.jsonl`, `logs/eval_iter_hyde_or-qwen3-30b-moe_20260427_0347_detail.jsonl`; baseline logs `logs/eval_rag_simple_or-gemma27b_20260426_2355_detail.jsonl`, `logs/eval_rag_simple_groq-scout_20260427_0246_detail.jsonl`, `logs/eval_rag_simple_or-qwen3-30b-moe_20260427_0334_detail.jsonl`; exp rows `20260427_0034`, `20260427_0320`, `20260427_0347`; docs `docs/audit_log.md`, `docs/signoff_log.md`; commits `4d06a34`, `c8bcd05`, `a3aee05` | MINOR — 1/15 sampled routed fallback in Scout `iter_hyde`; full scan also has low-rate baseline `None` outputs |
| `iter_hyde` does not help Llama 70b at Tier 2 | 49/200 = 24.5% vs `rag_simple` 55/200 = 27.5%; -3.0pp; p=0.47 NS | Citeable as neutral/not significant, not as active harm | Detail logs `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl`, `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp rows `20260427_1036`, `20260427_0952`; doc `docs/mcnemar_2026-04-27.md`; result commit `6b58ddb` | MINOR — 1/20 sampled `None` in baseline `rag_simple`; `iter_hyde` sample is clean |
| `rag_snap_hyde` does not carry from BarExam to MuSiQue | MuSiQue Llama 70b N=200: 48/200 = 24.0% vs 55/200 = 27.5%; -3.5pp; p=0.36 NS | Citeable as method-specificity evidence | Detail logs `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl`, `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; exp rows `20260427_1019`, `20260427_0952`; doc `docs/mcnemar_2026-04-27.md`; result commit `21e687a` | MINOR — sampled shared record has `None` in both cited logs; full scan 8/400 such artifacts |
| BarExam `rag_snap_hyde` remains positive where it belongs | Gemma 4 26B-A4B +3.09pp; Gemma 4 E4B +3.69pp | Citeable Tier 3, but post-fix re-scored from audit/detail logs, not experiments rows | Detail logs from Section 1.1; docs `docs/audit_log.md`, `docs/signoff_log.md`; audit/result commit `8bbf0e7` | MINOR — repeats Section 1.1 caveats: 26B baseline startup nulls/empty retrieval and one E4B raw null parse outside sample |

#### Audit — `iter_hyde` small-model negative finding (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/15 sampled; full scan found 4 `None`/empty artifacts in cited baseline logs |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 1/15 triggered: Scout `iter_hyde` record `2hop__704217_82341` has `routed_to=iter_hyde_early_exit_empty_decider` |
| Empty retrieval | 0.0% empty: 0/15 sampled retrieval-required records empty |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `2hop__704217_82341`, `4hop1__88342_49853_128008_46748`, `3hop1__654651_55349_651302` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — the negative finding stays directional; one sampled Scout fallback means do not overstate as clean active harm.

#### Audit — `iter_hyde` Llama 70b Tier 2 negative finding (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20: baseline `rag_simple` `2hop__748182_78303` parsed as `None`; full scan 6/400 artifacts across cited logs |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 4/20 sampled IDs across cited logs |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — cite as neutral/not significant, not active harm; the `iter_hyde` detail log itself sampled cleanly.

#### Audit — `rag_snap_hyde` MuSiQue non-transfer finding (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 1/20 sampled ID (`2hop__748182_78303`) has `None` in both `rag_snap_hyde` and `rag_simple`; full scan 8/400 artifacts |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 7/20 sampled IDs across cited logs |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — suitable as method-specificity evidence with the low-rate `None` caveat.

#### Audit — BarExam `rag_snap_hyde` positive finding (15 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/15 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 2/15 sampled: 26B `rag_simple` `mbe_0` and `mbe_1` null; full scan also found E4B `rag_snap_hyde` `mbe_95` null |
| `<think>` leakage | 0/15: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | 0/15 (BarExam): no hidden snap-stage echo found in final answers |
| routed_to fallbacks | 0/15 triggered: no `routed_to`; `gold_retrieved=false` on 14/15 sampled IDs across cited logs |
| Empty retrieval | 13.3% empty: 2/15 sampled IDs empty in the 26B `rag_simple` baseline log |
| Format anomalies | 0/15: expected fields present |
| Sampled records | `mbe_0`, `mbe_1`, `mbe_2`, ..., `mbe_1149`, `mbe_1172`, `mbe_1186` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: MINOR — positive BarExam result is still citeable only with the existing post-fix/raw-detail caveats.

## Section 6 — Methodology hardening shipped this week

- Pre-flight gate, circuit breaker, summary guard, think-tag strip: `171c2c4`; documented in `docs/audit_log.md`.
- Empty-retrieval guard / FAILED-EMPTY-RETRIEVAL protection: `5f8b723`; caught `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL` in `logs/experiments.jsonl`.
- Extractor fallback and `_run_gap` routed_to marker: `ed15eb7`; `docs/audit_log.md` records the +1 subagent_hybrid recovery and silent-fallback audit.
- mhd / iter_hyde silent-empty fallback bug fixes: `393e12f`; implementation risks are retained in `docs/rigour_signoff.md`.
- Tier system / N<200 citation discipline: `800c454`; `docs/rigour_signoff.md`.
- McNemar paired-test infrastructure and result docs: `83fb2fc`, `3ab2f51`, `21e687a`, `6b58ddb`; `docs/mcnemar_2026-04-27.md`.

## Section 7 — Currently in flight

| Run | Status | Detail log | Source | Audit verdict |
|---|---|---|---|---|
| `gemma4_full` mhd-pair × Gemma 4 26B-A4B × N=2400 MuSiQue | In flight; `/tmp/mhd_pair_gemma4_full.log` tail showed `[324/2400]` (~13.5%) on the current local read, close to the requested ~12% snapshot | pending | `/tmp/mhd_pair_gemma4_full.log`; `docs/mcnemar_2026-04-27.md` lists Gemma 4 26B-A4B full MuSiQue in flight | audit deferred until run completes |
| `qwen_full` mhd-pair × Qwen3 30B MoE × N=2400 MuSiQue | In flight; `/tmp/mhd_pair_qwen_full.log` tail showed `[791/2400]` (~33.0%), close to the requested ~30% snapshot | pending | `/tmp/mhd_pair_qwen_full.log`; `docs/mcnemar_2026-04-27.md` lists Qwen3 30B MoE full MuSiQue in flight | audit deferred until run completes |
| SLURM BarExam mhd+iter_hyde × Gemma 4 26B-A4B N=200 | Pending/unverified locally. User supplied SLURM `55107`; docs conflict with older `55040` / `55094`, and live `ps` was blocked by sandbox. | pending | `docs/mcnemar_2026-04-27.md`; `docs/signoff_log.md`; user-provided current snapshot | audit deferred until run completes |
| `subagent_rag` × Llama 70b N=200 | Completed and signed as a significant negative: 31/200 = 15.5%, -12.0pp vs `rag_simple`, McNemar p=0.0007. | `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl` | `logs/experiments.jsonl` row `20260427_1044`; `docs/mcnemar_2026-04-27.md`; `docs/signoff_log.md` | APPROVED as negative evidence; not a candidate improvement |

#### Audit — `subagent_rag` Llama 70b MuSiQue N=200 (20 records sampled)
| Confound | Status |
|---|---|
| Truncation | 0/20 records flagged: no abrupt cutoffs or unmatched `<think>` tags |
| Empty/garbage pred | 0/20 sampled; full scan found 1/200 `predicted_answer='None'` at `3hop1__857_846_7846` |
| `<think>` leakage | 0/20: parsed predictions contain no `<think>` or `</think>` substrings |
| Snap letter leakage | N/A: MuSiQue is not BarExam |
| routed_to fallbacks | 0/20 triggered: no `routed_to`; `gold_retrieved=false` on 20/20 sampled records and 200/200 full records |
| Empty retrieval | 0.0% empty: 0/20 sampled retrieval-required records empty |
| Format anomalies | 0/20: expected fields present |
| Sampled records | `2hop__121145_561444`, `2hop__86689_728109`, `3hop1__462960_160545_62931`, ..., `4hop3__547073_88460_30152_20999`, `4hop3__524186_219173_548463_72134`, `3hop1__79039_131926_87157` |
| Audited at | 2026-04-27 ~11:18 CDT by codex |

Verdict: CAVEAT — retrieval is non-empty but exact-gold tracking is 0/200, so do not use this row for retrieval-recall mechanism claims. The paired EM result itself is signed as negative evidence in `docs/signoff_log.md`.

## Section 8 — What NOT to cite (failed runs / contaminated rows)

- All N=30 runs as `result`; use only as smoke/direction, even when the direction is consistent.
- All N=100 runs as definitive; at most Tier 1 directional unless reinforced by N=200+.
- The `advisor_planning_table` BarExam N=50 FAILED-EMPTY-RETRIEVAL row in `logs/experiments.jsonl`: `20260426_2242_advisor_planning_table_groq-llama70b_api-barexam-advisor-llama-n50_FAILED-EMPTY-RETRIEVAL`; detail log `logs/eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl`; doc `docs/audit_log.md`; commit `45f1e03`; empty retrieval 50/50.
- Pre-fix BarExam numbers before formatter/retrieval-query fixes `f95f316` and `3d5ff05`; use post-fix `docs/audit_log.md` values.
- Any positive framing of `subagent_rag` on Llama 70b MuSiQue; it is now signed negative evidence (-12.0pp, p=0.0007).

## Reproducibility appendix

- Current HEAD: `44427ad` (`docs: signoff_log Section F+G — codex retroactively audited 15 historical N>=200 runs`).
- Recent result commits from `git log --oneline -30`: `6b58ddb`, `21e687a`, `3ab2f51`, `83fb2fc`, `6b7a922`, `a3aee05`, `800c454`, `77dd9da`, `393e12f`, `5f8b723`, `8bbf0e7`.
- Historical hardening commits required for BarExam interpretation: `f95f316` (prompt column in BarExam formatting), `3d5ff05` (prompt column in retrieval/rerank query paths), `ed15eb7` (extractor fallback + routed_to marker), `171c2c4` (pre-flight/circuit/summary/think-tag guard).
- Data-state caveat: `logs/experiments.jsonl` is dirty in the current worktree. The latest MuSiQue rows are directly verifiable locally, but the JSONL edits themselves have not all been committed. The post-fix Tier 3 BarExam detail logs are under ignored `logs/` paths and absent from `logs/experiments.jsonl`; the committed audit trail for those values is `docs/audit_log.md`.
