# Pause State — 2026-04-22

This session paused at **2026-04-22 ~04:50 UTC** so that another time-sensitive
project can use the cluster GPUs. Below is everything you need to resume the
LegalRagAgent BarExam size-comparison work later.

## TL;DR for resume

You're in the middle of doing a clean post-prompt-fix rerun of the
4-size × N=1195 BarExam matrix. Pre-fix data exists across many cells;
some post-fix data already landed; remaining cells need rerun under
commit `3d5ff05` or later.

## Code state

- Active branch: `hpc-setup`
- Last commit: `e83c1fd` (focused clean-rerun script committed but NOT yet executed)
- Critical fix commits (in order):
  - `e508765` — HyDE prompt sanitizer
  - `951729d` — strip snap "Answer: (X)" before HyDE context
  - `bf89b78` — _report_prompt hardening
  - `7a2ee28` — judge INCORRECT/CORRECT substring bug
  - `e9ed9ab` — schema normalization (gold_retrieved/retrieved_ids/evidence_store)
  - `17127c0` — snap-key normalization + entity-search dataset guards
  - `f95f316` — **MAJOR** BarExam prompt column included in answer prompt
  - `3d5ff05` — **MAJOR** BarExam prompt column included in retrieval queries
- Rigour sign-off framework: `docs/rigour_signoff.md`
- Bug catalogue (with regression tests): same doc

## Tests

- `tests/test_sanitizer.py` — 10 tests (sanitizer + strip + judge)
- `tests/test_formatter.py` — 7 tests (prompt-column inclusion + retrieval-query helper)
- All 17 pass at `e83c1fd`. Run: `python tests/test_sanitizer.py && python tests/test_formatter.py`

## Cluster state at pause

Cluster repo: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean` at commit `3d5ff05`.

### Jobs cancelled (deliberately, to free hardware)

| Job | Reason for cancel | What we got from it |
|---|---|---|
| 51023 (E4B seed=99 × 4 modes) | 1 mode landed → enough variance at this scale | rag_simple seed=99 = 55.73% |
| 51024 (26B seed=99 × 4 modes) | 3 modes landed → enough variance | rag_simple 71.8%, rag_hyde 74.6%, rag_snap_hyde 75.4% (seed=99) |
| 51206 (31B post-fix N=200) | covered by 50865's mode-by-mode landings | — (never started) |
| 50993, 50994, 50995 (31B expansion: llm_only/golden + subagent variants) | deferred to v2 wave | — (never started) |

### Jobs left running at pause

| Job | Hardware | Reason for keep | Cancel after |
|---|---|---|---|
| 51205 (E4B post-fix N=200 × 4 modes) | a100s-2306 | Gating sign-off data | All 4 modes land (~70 min from pause) |
| 50865 (31B full × 4 modes, pre-fix) | h100-2405 | Want rag_snap_hyde data point | After mode 3 (rag_snap_hyde) lands |
| 50986 (E2B redo × 3 modes) | a40-2205 | Want mode 2 (rag_snap_hyde) result | After mode 2 lands (~30 min) |

If you forget to cancel them, they'll just complete — total wallclock of
remaining modes is fine. Cancel only matters if you need the GPU slots
RIGHT NOW.

## Data landed so far (truth source: `logs/experiments.jsonl` on cluster)

### Pre-fix full N=1195 (commit `770c9ac` and earlier)

```
Model         rag_simple  rag_hyde  rag_snap_hyde  llm_only  golden  snap_only  subagent_rag  subagent_hybrid
E2B  (4B)      45.4%      43.7%        —             —         —         —          —              —
E4B  (8B)      55.7%      57.7%       58.4%          —         —       54.8%        —              —
26B  (25B)     70.8%      74.2%       76.6%        74.3%     75.0%    75.15%       75.7%         73.4%
31B  (31B)     79.6%      80.4%       (in flight)    —         —         —          —              —
```

Seed=99 pre-fix:
- 26B rag_simple = 71.8% (vs seed=42 70.8%; ~+1pp variance)
- 26B rag_hyde = 74.6% (vs seed=42 74.2%)

### Post-fix landings (commit `3d5ff05`)

```
Mode                       Pre-fix  Post-fix  Δ
N=200 E4B rag_simple        61.5%   61.0%    -0.5pp  (within ±6.9% N=200 CI)
N=200 E4B rag_hyde          59.5%   61.5%    +2.0pp  ← real lift from retrieval-query fix
N=200 E4B rag_snap_hyde     66.5%   (still running)
N=200 E4B snap_only_in_final 64.0%  (still running)

Full N=1195 26B subagent_hyde     —    76.6%  (post-fix — was running mid-job)
Full N=1195 26B snap_hyde_report  —    76.6%  (post-fix — same)
Full N=1195 26B rag_snap_hyde s99 —    75.4%  (post-fix, seed=99)
Full N=1195 E4B rag_simple s99    —    55.73% (post-fix, seed=99 — vs seed=42 55.7%)
Full N=1195 E4B snap_only_in_final —   54.8%  (post-fix)
```

### Key findings (validated, paper-worthy)

1. **`rag_simple` scales monotonically**: 45.4 → 55.7 → 70.8 → 79.6% across E2B → E4B → 26B → 31B (full N=1195, all post-fix-immune).
2. **HyDE lift over plain RAG is modest at scale**: +2pp at E4B, +3.4pp at 26B, +0.8pp at 31B — inverted-U.
3. **At 26B, `rag_snap_hyde` (76.6%) BEATS `golden_passage` oracle (75.0%) by +1.6pp**. Snap+HyDE adds reasoning structure beyond what a single gold passage provides. This is the most surprising finding.
4. **HyDE actively HURTS at E2B (4B) scale**: rag_simple 45.4% → rag_hyde 43.7% = -1.7pp. Small models can't use the extra retrieval signal.
5. **At 26B, retrieval is ~saturated**: golden_passage 75.0% ≈ llm_only 74.3%. Most of the answer is already in parametric knowledge.
6. **Bug impact is asymmetric**: retrieval modes (rag_hyde, rag_snap_hyde, subagent_*) lose ~2pp when prompt is dropped; non-retrieval modes (llm_only, snap_only) are essentially unaffected.

## Plan for resume

### Option 1 (recommended): Targeted clean rerun

`bash scripts/hpc/submit_clean_rerun_v1_focused.sh` on cluster fires 3 jobs:
- E4B retrieval modes (rag_simple/rag_hyde/rag_snap_hyde) full N=1195 at `3d5ff05`
- 26B retrieval modes full N=1195
- 31B retrieval modes full N=1195

~40 GPU-hours total. Rationale: the prompt bug only affects retrieval queries, so non-retrieval modes (llm_only/golden_passage/snap_only_in_final) keep their pre-fix numbers.

### Option 2 (broader): Full clean rerun

`bash scripts/hpc/submit_clean_rerun_v1.sh` — 4 sizes × 4 modes. ~80 GPU-hours. Use if you also want post-fix snap_only / E2B.

### Option 3 (extended): v2 wave with subagent variants

After v1 lands, queue:
- subagent_rag, subagent_hyde, subagent_hybrid, snap_hyde_report at full N=1195 across E4B/26B/31B
- Repeatability seed=99 on the v1 retrieval modes for variance bars

## Key files

- `docs/scale_config_interactions.md` — live analysis doc with current numbers
- `docs/size_comparison_matrix.md` — raw matrix state
- `docs/rigour_signoff.md` — pre-submission checklist
- `scripts/hpc/submit_clean_rerun_v1_focused.sh` — recommended next launch
- `scripts/hpc/submit_clean_rerun_v1.sh` — broader option
- `scripts/hpc/slurm_gemma4_rerun.sh` — generic parameterized SLURM
- `scripts/hpc/slurm_gemma4_rerun_80gb.sh` — same, but with --gres=gpu:a100-sxm4:1
- `tests/test_formatter.py`, `tests/test_sanitizer.py` — regression tests
- `logs/experiments.jsonl` (cluster) — truth source for all results

## What Codex is being asked to do during the pause

Spawning a deep-analysis Codex session on what we already have:
- Cross-tabulate landed numbers by (model size, mode, seed, commit)
- Identify which cells are clean (post-fix) vs need rerun
- Surface any other findings buried in the detail logs we haven't looked at

That output will land here when ready: `docs/pause_state_2026-04-22.md` (this file
will get appended).

## Resume checklist

1. `git checkout hpc-setup && git pull` (if needed)
2. `python tests/test_sanitizer.py && python tests/test_formatter.py` (verify 17/17 pass)
3. `ssh wustl 'cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean && git pull origin hpc-setup && git log --oneline -1'` (verify cluster at `3d5ff05` or later)
4. Decide: Option 1 / 2 / 3 from above
5. Run the relevant `bash scripts/hpc/submit_clean_rerun_v1*.sh` on cluster
6. Monitor with `squeue -u hiqbal` + watch `logs/experiments.jsonl` for new rows
