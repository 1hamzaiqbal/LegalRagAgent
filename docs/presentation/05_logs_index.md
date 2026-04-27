# Logs Index

This file is an index of local detail logs for presentation-cited results. Every concrete `logs/...jsonl` path below was checked with `ls -la` and counted with `wc -l`. Stats and sign-off labels come from `docs/signoff_log.md`; `docs/compiled_results.md` supplies exact filenames where the signoff log gives a glob.

## BarExam Tier 3: full corpus, N=1195

### Gemma 4 26B-A4B

| Mode | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| `golden_passage` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_golden_passage_cluster-vllm_20260426_0224_detail.jsonl` | 1195 | APPROVED |
| `llm_only` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_llm_only_cluster-vllm_20260426_0027_detail.jsonl` | 1195 | APPROVED |
| `rag_hyde` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_rag_hyde_cluster-vllm_20260425_2240_detail.jsonl` | 1195 | APPROVED |
| `rag_simple` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` | 1195 | APPROVED-WITH-CAVEAT |
| `rag_snap_hyde` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` | 1195 | APPROVED |
| `snap_only_in_final` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_snap_only_in_final_cluster-vllm_20260426_0154_detail.jsonl` | 1195 | APPROVED |
| `subagent_hybrid` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0254_detail.jsonl` | 1195 | APPROVED-WITH-CAVEAT |
| `subagent_rag` | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_subagent_rag_cluster-vllm_20260425_2234_detail.jsonl` | 1195 | APPROVED |

### Gemma 4 E4B

| Mode | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| `rag_simple` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl` | 1195 | APPROVED-WITH-CAVEAT |
| `rag_hyde` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_rag_hyde_cluster-vllm_20260426_0714_detail.jsonl` | 1195 | APPROVED |
| `rag_snap_hyde` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` | 1195 | APPROVED |
| `snap_hyde_report` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_snap_hyde_report_cluster-vllm_20260426_1713_detail.jsonl` | 1195 | APPROVED |
| `snap_only_in_final` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_snap_only_in_final_cluster-vllm_20260426_1512_detail.jsonl` | 1195 | APPROVED |
| `subagent_hybrid` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_subagent_hybrid_cluster-vllm_20260426_0545_detail.jsonl` | 1195 | APPROVED |
| `subagent_hyde` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_subagent_hyde_cluster-vllm_20260426_1739_detail.jsonl` | 1195 | APPROVED |
| `subagent_rag` | Gemma 4 E4B / `cluster-vllm` | `logs/eval_subagent_rag_cluster-vllm_20260426_0545_detail.jsonl` | 1195 | APPROVED |

## MuSiQue Tier 2: Llama 70B N=200, paper headline

| Mode | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| `rag_simple` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | 200 | APPROVED baseline |
| `iterative_planning_table` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl` | 200 | APPROVED, TRENDING-SIG |
| `multi_hyde_diverse` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` | 200 | APPROVED, paper headline |
| `rag_multi_query` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_rag_multi_query_groq-llama70b_20260427_1112_detail.jsonl` | 200 | APPROVED mechanism decomposition |
| `rag_snap_hyde` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | 200 | APPROVED cross-domain negative |
| `iter_hyde` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_iter_hyde_groq-llama70b_20260427_1036_detail.jsonl` | 200 | APPROVED multi-round neutral |
| `advisor_planning_table` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_advisor_planning_table_groq-llama70b_20260427_1216_detail.jsonl` | 200 | APPROVED NS negative |
| `subagent_rag` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_subagent_rag_groq-llama70b_20260427_1044_detail.jsonl` | 200 | APPROVED, significant negative |

## MuSiQue Tier 2: other models

| Mode | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| `rag_simple` | Gemma 3 27B / `or-gemma27b` | `logs/eval_rag_simple_or-gemma27b_20260427_0309_detail.jsonl` | 200 | APPROVED |
| `multi_hyde_diverse` | Gemma 3 27B / `or-gemma27b` | `logs/eval_multi_hyde_diverse_or-gemma27b_20260427_0404_detail.jsonl` | 200 | APPROVED, NULL |
| `rag_multi_query` | Gemma 3 27B / `or-gemma27b` | `logs/eval_rag_multi_query_or-gemma27b_20260427_0536_detail.jsonl` | 200 | APPROVED mechanism context |
| `rag_simple` | Llama 4 Scout / `groq-scout` | `logs/eval_rag_simple_groq-scout_20260427_0459_detail.jsonl` | 200 | APPROVED |
| `rag_multi_query` | Llama 4 Scout / `groq-scout` | `logs/eval_rag_multi_query_groq-scout_20260427_0332_detail.jsonl` | 200 | APPROVED-WITH-CAVEAT |
| `rag_simple` | Qwen3 30B MoE / `or-qwen3-30b-moe` | `logs/eval_rag_simple_or-qwen3-30b-moe_20260427_0334_detail.jsonl` | 100 | APPROVED-WITH-CAVEAT, direction-only |
| `multi_hyde_diverse` | Qwen3 30B MoE / `or-qwen3-30b-moe` | `logs/eval_multi_hyde_diverse_or-qwen3-30b-moe_20260427_0448_detail.jsonl` | 100 | APPROVED-WITH-CAVEAT, direction-only |

## Cross-domain checks

| Check | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| BarExam winner on MuSiQue: `rag_snap_hyde` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_rag_snap_hyde_groq-llama70b_20260427_1019_detail.jsonl` | 200 | APPROVED negative transfer |
| MuSiQue comparator for above: `rag_simple` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl` | 200 | APPROVED baseline |
| MuSiQue winner on BarExam paired first 200: `multi_hyde_diverse` | Gemma 4 26B-A4B / `or-gemma4-26b` | `logs/eval_multi_hyde_diverse_or-gemma4-26b_20260427_1211_detail.jsonl` | 200 | APPROVED cross-domain rejection |
| BarExam comparator for above: `rag_simple` first 200 | Gemma 4 26B-A4B / `cluster-vllm` | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl` | 1195 | APPROVED-WITH-CAVEAT |

## Friend/foe attribution

| Mode | Model | Detail log path | Rows | Sign-off |
|---|---|---|---:|---|
| `friend_foe_attribution` | Gemma 3 27B / `or-gemma27b` | `logs/eval_friend_foe_attribution_or-gemma27b_20260427_0249_detail.jsonl` | 30 | APPROVED-WITH-CAVEAT, N=30 directional |
| `friend_foe_attribution` | Llama 3.3 70B / `groq-llama70b` | `logs/eval_friend_foe_attribution_groq-llama70b_20260427_0305_detail.jsonl` | 30 | APPROVED-WITH-CAVEAT, N=30 directional |

## In-flight: cluster/API, will land

| Run | Local detail log status | Sign-off |
|---|---|---|
| SLURM 55107 BarExam `multi_hyde_diverse` + `iter_hyde` x Gemma 4 26B-A4B N=200 | No local source detail log in this workspace yet; signoff says source log not present locally. | Expected APPROVED only after landing plus source log |
| `qwen_full` mhd-pair x Qwen3 30B MoE x N=2400 MuSiQue | No local source detail log in this workspace yet; signoff says source log not present locally. | Tier 2.5 partial only until full run plus audit lands |

## Killed, partial, or do not cite

| Run | Detail log path | Rows | Citation guidance |
|---|---|---:|---|
| `gemma4_full` mhd-pair x Gemma 4 26B-A4B x N=2400 MuSiQue | No local detail log found in signoff/compiled-results; operator snapshot only. | n/a | Cite only as OR-Gemma partial N=431 serving-caveated lower-bound if needed; do not cite as Tier 3. |
| `iterative_planning_table` x Gemma 27B N=200 | No local detail log path is given in signoff; killed at q29. | n/a | DO NOT CITE. |
| `advisor_planning_table` BarExam local Mac empty retrieval | `logs/eval_advisor_planning_table_groq-llama70b_20260426_2242_detail.jsonl` | 50 | DO NOT CITE; `FAILED-EMPTY-RETRIEVAL`. |
| Small-model `iter_hyde` direction-only logs | `logs/eval_iter_hyde_or-gemma27b_20260427_0034_detail.jsonl`; `logs/eval_iter_hyde_groq-scout_20260427_0320_detail.jsonl`; `logs/eval_iter_hyde_or-qwen3-30b-moe_20260427_0347_detail.jsonl` | 30 each | Direction-only, not definitive. |

## Quick how-to

Extract the first record:

```bash
head -1 logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl | jq '{idx, label, question, predicted_answer, correct_answer, is_correct}'
```

Extract by `record_id` when that field exists:

```bash
grep '"record_id": "mbe_0"' logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl | jq .
```

Most cited logs use `idx` and `label`, so this is usually the practical lookup:

```bash
grep '"idx": "mbe_0"' logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl | jq '{idx, label, predicted_answer, correct_answer, is_correct, error}'
```

Python lookup that handles `record_id`, `idx`, or `label`:

```bash
python3 - <<'PY'
import json

path = "logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl"
target = "2hop__121145_561444"

with open(path) as f:
    for line in f:
        row = json.loads(line)
        ids = {row.get("record_id"), row.get("idx"), row.get("label")}
        if target in ids:
            print(json.dumps(row, indent=2, ensure_ascii=False))
            break
PY
```
