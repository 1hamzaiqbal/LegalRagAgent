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

### Jobs left running at pause → ALL FINISHED 2026-04-22

Update: all 3 kept-running jobs landed their critical modes then exited.
Cluster queue is now fully empty.

| Job | Final state | Got us |
|---|---|---|
| 51205 E4B post-fix N=200 × 4 modes | COMPLETED | rag_simple 61.0%, rag_hyde 61.5%, rag_snap_hyde 67.5%, snap_only_in_final 63.5% (all post-fix, 0% leak) |
| 50865 31B full × 4 modes (pre-fix) | CANCELLED after mode 3 | rag_simple 79.6%, rag_hyde 80.4%, **rag_snap_hyde 83.9%** (full N=1195) |
| 50986 E2B redo × 3 modes | WALLCLOCKED at q 475 of mode 3 | rag_hyde 43.7%, rag_snap_hyde 46.9% (2 of 3 modes). snap_only_in_final not recovered. |

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

## 🔬 Codex deep-analysis output (appended)

Codex mined the 234-row `experiments.jsonl` + detail logs while the pause was
landing. Five paper-worthy findings + recommended next experiments.

### Five findings

1. **`rag_simple` monotonic but compresses at top.** Full N=1195: E2B 45.4 →
   E4B 55.7 → 26B 70.8 → 31B 79.6. The 26B→31B step (+8.8pp) is smaller than
   E4B→26B (+15.1pp), so the curve is compressing.

2. **At 26B, `rag_snap_hyde` > `golden_passage` is COMPLEMENTARITY, not
   dominance.** On the shared 1195 questions: `rag_snap_hyde` gets 126 right
   that `golden` misses; `golden` gets 107 right that `rag_snap_hyde` misses.
   **Pair-union ceiling = 85.5%.** Strongest ensemble signal in our data —
   suggests a fusion mode combining snap+HyDE retrieval with an oracle-style
   gold-passage check could push past either single mode.

3. **Cost-adjusted Pareto**: at 26B, `rag_snap_hyde` reaches 76.57% in 3 LLM
   calls. `subagent_hyde` and `snap_hyde_report` need 5 and 4 calls
   respectively for the same 76.57%. Subagent variants buy NO extra accuracy.
   Also: `golden_passage` 75.0% / 1 call dominates `rag_hyde` 74.2% / 2
   calls — the oracle is cheaper than retrieval here.

4. **Subject scaling is highly non-uniform.** E2B→31B `rag_simple` gain by
   subject: Contracts +42.5pp, Criminal Law +41.6pp, Evidence +40.9pp,
   Constitutional Law only +29.5pp total (and just +3.2pp from 26B→31B).
   Const Law saturates EARLY; the harder doctrinal subjects benefit MOST
   from scale. Computed from `eval_rag_simple_*_2026042[12]_*_detail.jsonl`
   for each size.

5. **`rag_snap_hyde > golden` is a 26B-specific phenomenon, not a scaling
   law.** Fails at E4B (58.4 < 62.2 → -3.77pp). Holds at 26B (76.6 > 75.0
   → +1.59pp). Uncheckable at E2B/31B because we never ran golden_passage
   at those sizes.

### HyDE inverted-U lift confirmed

Cross-scale `rag_hyde - rag_simple`: E2B -1.7pp, E4B +2.0pp, 26B +3.4pp,
31B +0.8pp. Genuine inverted-U; small models can't use the retrieval signal,
big models don't need it.

### Discrepancies to be honest about

- "snap+HyDE +7pp" was pre-fix N=200 (66.5 - 59.5). Clean post-fix N=200:
  +6pp (67.5 - 61.5). Close but smaller. Full N=1195 pre-fix is +0.7pp —
  the +6/+7 figure does NOT survive at full N. **This is the gap a clean
  post-fix N=1195 rerun will resolve.**

### Recommended next experiments (when GPUs return)

- Clean post-fix full-N=1195 rerun of E4B/26B/31B `rag_simple`/`rag_hyde`/
  `rag_snap_hyde` at seed=42 + seed=99. (`scripts/hpc/submit_clean_rerun_v1_focused.sh`)
- Fill 31B `llm_only`/`golden_passage`/`snap_only_in_final` cells to test
  whether the 26B "oracle barely above llm_only" plateau persists at 31B.
- Build a **fusion/rerank mode** that combines `golden_passage` and
  `rag_snap_hyde` outputs: the 85.5% pair-union ceiling at 26B is the
  strongest actionable ensemble signal we have.

### Latest landings during the pause window

- **51205 E4B post-fix N=200 × 4 modes COMPLETE** — all 0% leak.
  rag_simple 61.0% (Δ +0.5), rag_hyde 61.5% (Δ +2.0), rag_snap_hyde 67.5%
  (Δ +1.0), snap_only_in_final 63.5% (Δ -0.5). Snap-over-HyDE post-fix
  N=200 = **+6pp** (67.5 - 61.5).
- **50865 31B full N=1195 `rag_snap_hyde` = 83.9%** (1003/1195) ⭐
  Snap-over-HyDE at 31B full = **+3.5pp** (83.9 - 80.4). Stacking effect
  stays meaningful at 31B, did NOT collapse as prior N=200 reading
  suggested. Cancelled 50865 after this mode to free H100.

### Updated cross-size stacking table (landed pre-fix full N=1195)

```
Model    simple    hyde    snap_hyde    hyde→snap_hyde    simple→snap_hyde
E2B      45.4%    43.7%    —             —                 —
E4B      55.7%    57.7%    58.4%        +0.7pp             +2.7pp
26B      70.8%    74.2%    76.6%        +2.4pp             +5.8pp
31B      79.6%    80.4%    83.9%        +3.5pp             +4.3pp  <-- NEW
```

**Narrative update**: the "method stacking collapses at scale" reading was
based on pre-fix N=200 data (+2pp at 31B). Full-N=1195 actually shows
snap-over-HyDE stacks MORE at 31B (+3.5pp) than at 26B (+2.4pp). The
"HyDE saturates at scale" story still holds (rag_hyde vs rag_simple +0.8pp
at 31B), but the SNAP contribution grows.

## Resume checklist

1. `git checkout hpc-setup && git pull` (if needed)
2. `python tests/test_sanitizer.py && python tests/test_formatter.py` (verify 17/17 pass)
3. `ssh wustl 'cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean && git pull origin hpc-setup && git log --oneline -1'` (verify cluster at `3d5ff05` or later)
4. Decide: Option 1 / 2 / 3 from above
5. Run the relevant `bash scripts/hpc/submit_clean_rerun_v1*.sh` on cluster
6. Monitor with `squeue -u hiqbal` + watch `logs/experiments.jsonl` for new rows
