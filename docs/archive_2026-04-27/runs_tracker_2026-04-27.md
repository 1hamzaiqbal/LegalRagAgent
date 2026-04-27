# ARCHIVED 2026-04-27 — superseded by docs/signoff_log.md
# Runs in flight (snapshot 2026-04-27 ~02:50 CDT)

## Cluster (SLURM)
| JobID | Mode(s) | Model | Dataset | N | ETA | Tag |
|---|---|---|---|---|---|---|
| (ssh unavailable) | (ssh `wustl 'squeue -u hiqbal'` failed) | (unknown) | (unknown) | (unknown) | (unknown) | (ssh unavailable) |

## Local (background processes)
| PID | Mode | Provider | Dataset | N | ETA | Log file |
|---|---|---|---|---|---|---|
| (ps unavailable) | `rag_simple` | `or-qwen3-30b-moe` | `musique` | 100 | ~25-35 min from `[16/100]` observed at 02:49 CDT | `/tmp/mhd_pair_qwen.log` |

## Targeted Question Each Answers
- `/tmp/mhd_pair_qwen.log`: What is the Qwen3-30B-MoE MuSiQue `rag_simple` baseline for the planned MHD pair at N=100?

## When all complete: what story do they tell?
- If positive, the story is that Qwen3-30B-MoE has a usable MuSiQue `rag_simple` baseline for a later MHD-pair comparison, giving another model family to test the retrieval-diversity claim against.
- If negative/null, the story is that the Qwen3-30B-MoE MuSiQue baseline is weak or unstable enough that any later MHD comparison needs careful framing around provider behavior and answer verbosity rather than a clean method effect.
