# Gemma 4 Size-Comparison Matrix (post leak-fix)

Side-by-side full-corpus runs across all 4 Gemma 4 sizes using the identical
4-mode matrix. Purpose: measure whether scaling past 8B gives meaningful
reasoning/retrieval gains, and get clean post-leak-fix baselines per size.

## Mode matrix (held constant across all sizes)

| # | Mode | Purpose |
|---|---|---|
| 1 | `rag_simple` | plain RAG baseline |
| 2 | `rag_hyde` | HyDE retrieval (question → hypothetical passage) |
| 3 | `rag_snap_hyde` | snap reasoning → HyDE retrieval → answer (snap hidden) |
| 4 | `snap_only_in_final` | **ablation cell**: snap visible to final, no retrieval |

All modes run at **N=1195 (full BarExam)**, seed=42, post-hardening commits.

## Jobs submitted

| Model | Size | N | Job ID | Target | SLURM script | Est. wall |
|---|---|---|---|---|---|---|
| `gemma-4-E2B-it` | 4B | 1195 × 4 | **50867** | a40-2206 (running) | `slurm_gemma4_rerun.sh` | ~6-8h |
| `gemma-4-E4B-it` | 8B | 1195 × 3 | **50858** | a40-2206 (running) | `slurm_gemma4_rerun.sh` | ~10h |
| `gemma-4-E4B-it` | 8B | 1195 × 1 (rag_snap_hyde) | **50859** | a40-2206 (running) | `slurm_gemma4_rerun.sh` | ~8h |
| `gemma-4-26B-A4B-it` | 25B (MoE) | 1195 × 4 | **50868** | a100-sxm4 (pending) | `slurm_gemma4_rerun_80gb.sh` | ~12-18h |
| `gemma-4-31B-it` | 31B | 1195 × 4 | **50865** | h100-2405 (pending) | `slurm_gemma4_rerun.sh` | ~15-20h |

Also running alongside:

| Job | What | Size | N |
|---|---|---|---|
| 50835 | E4B mini-eval (leak-fix validation) | 8B | 200 × 4 |
| 50864 | 31B N=200 matrix | 31B | 200 × 4 |

## Dispatcher model

SLURM queue is the dispatcher. All jobs submitted at once; SLURM schedules them
onto nodes as resources free. When a 50xxx job completes, the next pending job
auto-promotes without manual intervention.

Monitor with `squeue -u hiqbal`. Re-sbatch to queue additional runs
(different seed for repeatability, different model, etc.).

## Follow-on queue (plan)

Once the above 5 land, queue next:
- **Repeatability**: E4B + 31B with seed=99 on the same 4 modes (variance estimate)
- **Subagent ablation** (Priority 2 from the earlier plan): `subagent_rag`,
  `subagent_hyde`, `subagent_hybrid`, `snap_hyde_report` at N=1195 on E4B
- If H100 ever sits idle: 31B × subagent modes

## Expected comparison table (what to fill in once runs complete)

```
Model size      rag_simple   rag_hyde   rag_snap_hyde   snap_only_in_final
E2B (4B)        ?            ?          ?               ?
E4B (8B)        ?            ?          ?               ?
26B-A4B (25B)   ?            ?          ?               ?
31B (dense)     ?            ?          ?               ?

Historical E4B pre-fix:  54.2%   57.9%   58.6%   (no snap_only_in_final)
```

## Notes

- Triton cache now goes to `/tmp/hiqbal-triton/$SLURM_JOB_ID` per commit d0709bd.
  Each job re-compiles kernels (adds ~2-5 min to startup), but eliminates NFS
  atomic-rename failures that killed jobs 50860/50862/50863.
- Home quota cleaned from 30GB → 7GB on 2026-04-21; UV_CACHE_DIR now redirects
  future uv cache growth to engrfs.
- Nodelist restrictions: exclude `r28-1801` (RTX 2080, too small), `a100-2207`
  (bad vLLM init). `a100s-2306` was confirmed clean post-reconfig.

## Completion log

_(populate as jobs land)_
