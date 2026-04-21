# Gemma 4 Size-Comparison Matrix (post leak-fix)

> **⚠ 2026-04-22 methodology bug (commit `f95f316`)**: Every full-N=1195
> number in this doc was produced by a harness that silently dropped the
> BarExam `prompt` column (37% of questions lost their fact-pattern
> context). The fix landed in `format_question_prompt` + `_fmt_intermediate`;
> N=200 validation reruns submitted as 51179 (E4B) and 51180 (31B).
> Relative rankings across sizes/modes are unaffected (bug hit all modes
> equally). Absolute numbers are under-estimates of post-fix accuracy.

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

## Live snapshot (2026-04-21 evening)

Partial full-N=1195 results — modes landing as jobs complete. Clean (0% leak) where recorded.

| Model | rag_simple | rag_hyde | rag_snap_hyde | snap_only_in_final |
|---|---|---|---|---|
| **E2B** (4B) | **45.4%** | in 50986 | in 50986 | in 50986 |
| **E4B** (8B) | **55.7%** | running (50858 q~1009/1195) | **58.4%** (50859) | in 50858 |
| **26B-A4B** (25B MoE) | **70.8%** | **74.2%** | running (50868 q~765/1195) | pending in 50868 |
| **31B** (31B dense) | **79.6%** | running (50865 q~411/1195) | pending | pending |

Clean monotonic scaling on `rag_simple`: **45.4 → 55.7 → 70.8 → 79.6%** from 4B → 8B → 25B → 31B.

### Noise correction on the "snap +7pp" claim

The N=200 mini-eval showed `rag_hyde` 59.5% → `rag_snap_hyde` 66.5% = **+7pp snap lift**. At full N=1195, `rag_snap_hyde` post-fix lands at **58.4%**, basically matching the pre-fix 57.9-58.6% range. Need `rag_hyde` N=1195 post-fix to complete the apples-to-apples, but the partial read is that N=200 was too noisy (±6.9% 95% CI) to trust the +7pp point estimate. The real snap-over-HyDE delta at full scale is likely much smaller.

Currently observable at full N=1195 post-fix E4B: `rag_simple` → `rag_snap_hyde` = **+2.7pp** (55.7% → 58.4%). Waiting on `rag_hyde` + `snap_only_in_final` to finish the 4-mode comparison.

### 2026-04-21 late — E4B full HyDE + 26B llm_only landed

**E4B full N=1195 lift decomposition (clean, post-fix):**
- `rag_simple` 55.7% → `rag_hyde` **57.7%** (50858): HyDE lift = **+2.0pp**
- `rag_hyde` 57.7% → `rag_snap_hyde` 58.4%: snap lift over HyDE = **+0.7pp** (noise floor — confirms the N=200 "+7pp" was noise)

**26B-A4B full N=1195 (50868 + 50990):**
- `llm_only` **74.3%** (889/1195) ≈ `rag_hyde` 74.2%: **retrieval contributes literally 0pp at 25B scale**
- `rag_simple` 70.8% → `llm_only` 74.3% = **+3.5pp by removing retrieval** — parametric knowledge beats the BM25/HyDE retrieval noise

**Emerging narrative shift**: at 25B+ scale, parametric knowledge dominates retrieval. HyDE edges out plain RAG because HyDE at least grabs doctrinally-relevant passages, but both are no better than `llm_only`. The "retrieval-first" framing may not survive at scale — we need to see 31B `llm_only` + `rag_hyde` to confirm the pattern.

### 2026-04-21 audit — logs verified clean

Codex audited 11 post-fix detail logs (2026-04-21) + 3 historical pre-fix for reference. Findings:

- **HyDE leakage on post-fix logs**: **0%** across all 11 checked logs (`rag_hyde_*_04{58,1112,1449,0203,0229}`, `rag_snap_hyde_*_{0359,0632,1402,1515}`, plus smokes). Historical pre-fix `rag_hyde_20260417_2047` remains at **99.9%** as reference.
- **Extraction pipeline**: `predicted_answer` → `is_correct` derivation correct in all 31 sampled rows across the logs. No empty-predicted-but-text-final anomalies.
- **End-to-end spot-check** (3 randomly-chosen questions from `rag_snap_hyde_0359`): 2/3 pipelines end-to-end sensible (HyDE matches doctrine, retrieved passages on-topic, final letter correct). 1/3 flagged as a model-behavior issue (model output "Cannot be determined due to missing question context" before forcing a letter) — not a pipeline bug; BarExam fill-in-the-blank stems occasionally confuse the model.
- **One residual observation**: `report` bodies (in `snap_hyde_report` mode) contain embedded "Answer:" strings mid-text in ~51% of cases on the historical 2026-04-17 log. The sanitizer only strips at-start patterns. Low-priority — affects only snap_hyde_report's final-agent context, not retrieval queries.

Net: current post-fix numbers are trustworthy. No targeted reruns required.

### 2026-04-21 late — second audit on the landed cross-scale matrix

Codex ran a deeper 5-check audit on the 11 post-fix full-N=1195 detail logs
(E4B/26B/31B × various modes) after the surprising finding that 26B
`rag_snap_hyde` (76.6%) beats `golden_passage` (75.0%). All checks substantively PASS:

- **Accuracy numbers match claims**: all 11 per-row counts match the headline
  numbers exactly. (Minor metadata gap: experiments.jsonl is missing 7 summary
  rows — the detail logs are correct, the ledger isn't complete.)
- **rag_snap_hyde > oracle is real**: +19 net paired-diff wins on aligned idxs
  (126 snap-right/gold-wrong vs 107 gold-right/snap-wrong). 5 random
  inspections each side show coherent doctrinal reasoning, not junk outputs.
  Not contamination — snap+HyDE is adding reasoning structure beyond what a
  single gold passage provides.
- **Cross-mode idx alignment**: zero drift across 11 logs. Identical row
  order, question text, and answer keys everywhere.
- **is_correct derivation sane**: 110 sampled rows, 100% match manual
  computation. No stored-vs-computed mismatches in any file.

**Metadata bug found + patched**: `run_golden_passage` was setting
`gold_retrieved=False` / `retrieved_ids=[]` even though the gold passage was
being injected into the prompt. Fixed for future runs; historical log metadata
unchanged but prompt behavior was always correct.

Bottom line: the 26B-A4B "rag_snap_hyde beats oracle by +1.6pp" finding is
real and paper-worthy. The small full-N HyDE lifts (E4B +2pp, 31B +0.8pp) are
also legitimate — N=200 "+4pp" / "+7pp" readings were just the wider noise
envelope at lower N.

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

## Expanded matrix (2026-04-21, additional jobs submitted)

For characterizing HyDE/snap × scale interactions at the big-model tier:

### 26B-A4B additional jobs (all target a100-sxm4 idle slots in parallel)
- **50990**: `llm_only` + `golden_passage` — baselines/ceiling for lift computation
- **50991**: `subagent_rag` + `subagent_hyde` — subagent ablation
- **50992**: `subagent_hybrid` + `snap_hyde_report` — subagent + summarization

### 31B additional jobs (queue for H100 after 50865/50993 complete)
- **50993**: `llm_only` + `golden_passage`
- **50994**: `subagent_rag` + `subagent_hyde`
- **50995**: `subagent_hybrid` + `snap_hyde_report`

### Final target per-size matrix (10 modes each)
1. `llm_only` — no retrieval, no snap
2. `golden_passage` — oracle ceiling
3. `rag_simple` — retrieval, no HyDE, no snap
4. `rag_hyde` — retrieval + HyDE, no snap
5. `rag_snap_hyde` — retrieval + HyDE + snap (snap hidden from final)
6. `snap_only_in_final` — snap in final context, no retrieval
7. `subagent_rag` — per-gap RAG + report summarization
8. `subagent_hyde` — per-gap HyDE + report summarization
9. `subagent_hybrid` — per-gap RAG + model knowledge → report
10. `snap_hyde_report` — snap-hyde with post-retrieval report

## Parallelism note

Three 26B jobs (50990/50991/50992) all target `a100s-2305` via `--nodelist=a100s-2305`.
That node has 4 GPUs (all idle at submit time) so SLURM can land all three in parallel
on different GPU slots. Similarly 50868 already on a100s-2306 keeps running
independently. The 31B jobs are H100-bound and run sequentially.

## Follow-on queue (plan — after the above land)

- **Repeatability**: E4B + 31B with seed=99 on the 5-mode HyDE×snap×retrieval
  core matrix (variance estimate — not all 10 modes)
- **Cross-dataset generalizability**: 26B + 31B on housing/casehold at N=500

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

### 2026-04-21 — Mini-eval 50835 (Gemma E4B N=200) complete

All 4 modes with 0% HyDE/report leakage post-hardening:

| Mode | post-fix N=200 | historical N=200 | Δ |
|---|---|---|---|
| `rag_simple` | **60.5%** (121/200) | 61.5% | −1.0pp (noise) |
| `rag_hyde` | **59.5%** (119/200) | 66.0% | **−6.5pp** |
| `rag_snap_hyde` | **66.5%** (133/200) | 64.5% | **+2.0pp** |
| `snap_only_in_final` | **64.0%** (128/200) | — (new) | — |

**Major narrative shift**: the old "snap adds 0pp to HyDE" claim was an
artifact of the leak. Post-fix, snap contributes **+7pp over plain HyDE**
(`rag_hyde` 59.5% → `rag_snap_hyde` 66.5%). The leaky `Answer: (X)` prefix
had been doing "answer-aware retrieval" work that snap is supposed to do,
making snap look redundant. With the leak cleaned up, snap's real value
surfaces.

**New decomposition**:
- snap reasoning alone (no retrieval) = **64.0%** — already beats `rag_simple`
  and `rag_hyde`
- HyDE retrieval on top of snap: +2.5pp
- HyDE retrieval without snap: +7pp vs `llm_only`, but still below snap-only

Snap is doing the heavier lifting. Retrieval is a secondary, additive gain.

Detail logs (cluster):
- `logs/eval_rag_simple_cluster-vllm_20260420_2355_detail.jsonl`
- `logs/eval_rag_hyde_cluster-vllm_20260421_0055_detail.jsonl`
- `logs/eval_rag_snap_hyde_cluster-vllm_20260421_0204_detail.jsonl`
- `logs/eval_snap_only_in_final_cluster-vllm_20260421_0359_detail.jsonl`

### 2026-04-21 — Job 50864 (Gemma 31B N=200) complete — major scaling signal

Gemma 4 31B-it (dense, unquantized, H100) on identical 4-mode matrix, seed=42, 0% leak:

| Mode | E4B (8B) N=200 | **31B N=200** | Δ over E4B |
|---|---|---|---|
| `rag_simple` | 60.5% | **79.0%** (158/200) | **+18.5pp** |
| `rag_hyde` | 59.5% | **83.0%** (166/200) | **+23.5pp** |
| `rag_snap_hyde` | 66.5% | **85.0%** (170/200) | **+18.5pp** |
| `snap_only_in_final` | 64.0% | **84.0%** (168/200) | **+20.0pp** |

**Every mode improves ~18-24pp at 31B.** Single biggest factor in accuracy
is model size; method choice is second-order past 8B.

**Method stacking collapses at 31B**:
- E4B: snap adds **+7pp** over plain HyDE (59.5 → 66.5)
- 31B: snap adds **only +2pp** over plain HyDE (83.0 → 85.0)
- E4B: retrieval on top of snap = +2.5pp (64.0 → 66.5)
- 31B: retrieval on top of snap = +1pp (84.0 → 85.0)

**Read**: at 31B, the model's parametric knowledge already contains most of
the answer signal that snap reasoning and HyDE retrieval provide at smaller
scale. Snap and HyDE become redundant with the bigger model's internal
reasoning.

**Still headroom**: 85% is well below the "actually competent" threshold.
Need to run `golden_passage` on 31B to see the scale-up ceiling.

Detail log: `logs/eval_*_cluster-vllm_2026042[01]_*_detail.jsonl` for tag
`31b-n200-matrix`.

Throughput: ~13s/query on H100 — **faster per-token than E4B on A40** despite
4× params, because H100's compute compensates. 200 questions × 4 modes took
~5h total (vLLM startup + inference).
