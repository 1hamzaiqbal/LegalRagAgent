#!/bin/bash
# E4B + 26B wide-coverage submission — clean post-fix N=1195 across all 10 core
# modes on both sizes. Replaces the size-spread (E2B/E4B/26B/31B) plan with a
# coverage-spread (E4B+26B × all modes) plan ahead of the 2026-04-27 meeting.
#
# Skipped from this wave (already landed post-fix at 3d5ff05+):
#   - 26B subagent_hyde     = 76.6%
#   - 26B snap_hyde_report  = 76.6%
#   - 26B rag_snap_hyde s99 = 75.4%
#   - 31B rag_snap_hyde     = 83.9%   (any 31B work deferred)
#
# 7 jobs total, 4 E4B (a40 default, 1 GPU each) + 3 26B (a100-sxm4 80GB).
# All seed=42. Multi-seed work deferred to a follow-up wave on Sunday.

set -e
REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean
cd "$REPO"

if ! git log --oneline -20 | grep -q "3d5ff05"; then
  echo "ERROR: cluster repo missing 3d5ff05 (retrieval prompt fix)." >&2
  exit 1
fi

if [[ ! -f scripts/analyze_detail_flags.py ]]; then
  echo "ERROR: scripts/analyze_detail_flags.py missing on cluster — pull repo first." >&2
  exit 1
fi

echo "Cluster repo at: $(git log --oneline -1)"
echo

# ============================================================================
# E4B (8B-effective, A40, 4 jobs) — all 10 core modes split for parallelism
# ============================================================================
# Wall-time budgets are generous: E4B on A40 is ~3x slower than 26B on sxm4.

# E4B-1: cheap modes seq, ~14h
sbatch -t 18:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8080,MODES="rag_simple rag_hyde llm_only golden_passage",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-e4b-cov1 \
  scripts/hpc/slurm_gemma4_rerun.sh

# E4B-2: snap-bearing retrieval modes, ~20h
sbatch -t 24:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8081,MODES="rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-e4b-cov2 \
  scripts/hpc/slurm_gemma4_rerun.sh

# E4B-3: subagent retrieval pair, ~22h
sbatch -t 28:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8082,MODES="subagent_rag subagent_hyde",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-e4b-cov3 \
  scripts/hpc/slurm_gemma4_rerun.sh

# E4B-4: subagent hybrid + report, ~22h
sbatch -t 28:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8083,MODES="subagent_hybrid snap_hyde_report",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-e4b-cov4 \
  scripts/hpc/slurm_gemma4_rerun.sh

# ============================================================================
# 26B-A4B (MoE, A100-sxm4 80GB, 3 jobs) — fills remaining 7 modes
# ============================================================================

# 26B-1: cheap modes + retrieval baseline, ~8h
sbatch -t 12:00:00 \
  --export=ALL,MODEL=google/gemma-4-26B-A4B-it,PORT=8090,MODES="rag_simple rag_hyde llm_only golden_passage",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.90,TAG_SUFFIX=clean-26b-cov1 \
  scripts/hpc/slurm_gemma4_rerun_80gb.sh

# 26B-2: snap-bearing retrieval, ~7h
sbatch -t 12:00:00 \
  --export=ALL,MODEL=google/gemma-4-26B-A4B-it,PORT=8091,MODES="rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.90,TAG_SUFFIX=clean-26b-cov2 \
  scripts/hpc/slurm_gemma4_rerun_80gb.sh

# 26B-3: missing subagent variants (subagent_hyde + snap_hyde_report already landed), ~9h
sbatch -t 14:00:00 \
  --export=ALL,MODEL=google/gemma-4-26B-A4B-it,PORT=8092,MODES="subagent_rag subagent_hybrid",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.90,TAG_SUFFIX=clean-26b-cov3 \
  scripts/hpc/slurm_gemma4_rerun_80gb.sh

echo
echo "=== queue ==="
squeue -u hiqbal --format="%i %j %T %M %R" | head -20
echo
echo "Submitted E4B+26B coverage wave: 7 jobs, ~17 mode-cells, ~50 GPU-hours."
echo "Yields complete post-fix coverage across both sizes for all 10 core modes."
