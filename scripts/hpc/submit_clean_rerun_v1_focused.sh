#!/bin/bash
# Clean-rerun v1 FOCUSED — only the retrieval modes affected by the prompt bug.
#
# Codex audit confirmed: 84/84 prompt-bearing rows show different retrieved
# passages pre/post-fix. So retrieval modes (rag_simple, rag_hyde,
# rag_snap_hyde, subagent_*) all need rerun. Modes that don't retrieve
# (llm_only, golden_passage, snap_only_in_final) are bug-immune and
# their pre-fix numbers can be kept as-is.
#
# Cells to rerun: 3 sizes × 3 retrieval modes = 9 cells.
#   E4B:  rag_simple, rag_hyde, rag_snap_hyde
#   26B:  rag_simple, rag_hyde, rag_snap_hyde
#   31B:  rag_simple, rag_hyde, rag_snap_hyde
#
# Skipped:
#   E2B (low publication value, accuracy floor anyway)
#   llm_only / golden_passage / snap_only_in_final (no retrieval, no bug)
#   subagent_* (v2 wave — focused first)
#
# Total estimate: ~40 GPU-hours across 3 jobs.

set -e
REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean
cd "$REPO"

if ! git log --oneline -20 | grep -q "3d5ff05"; then
  echo "ERROR: cluster repo missing 3d5ff05 (retrieval prompt fix)." >&2
  exit 1
fi

echo "Cluster repo at: $(git log --oneline -1)"
echo

# E4B (8B, A40-fits) — 3 retrieval modes seq, ~12h
sbatch -t 14:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8080,MODES="rag_simple rag_hyde rag_snap_hyde",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-rerun-v1-e4b-retrieval \
  scripts/hpc/slurm_gemma4_rerun.sh

# 26B-A4B (a100-sxm4 80GB) — 3 retrieval modes seq, ~14h
sbatch -t 18:00:00 \
  --export=ALL,MODEL=google/gemma-4-26B-A4B-it,PORT=8081,MODES="rag_simple rag_hyde rag_snap_hyde",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.90,TAG_SUFFIX=clean-rerun-v1-26b-retrieval \
  scripts/hpc/slurm_gemma4_rerun_80gb.sh

# 31B (h100, 80GB tight) — 3 retrieval modes seq, ~16h
sbatch --nodelist=h100-2405 -t 20:00:00 \
  --export=ALL,MODEL=google/gemma-4-31B-it,PORT=8082,MODES="rag_simple rag_hyde rag_snap_hyde",N_QUESTIONS=full,MAX_MODEL_LEN=4096,GPU_MEM_UTIL=0.92,TAG_SUFFIX=clean-rerun-v1-31b-retrieval \
  scripts/hpc/slurm_gemma4_rerun.sh

echo
echo "=== queue ==="
squeue -u hiqbal --format="%i %j %T %M %R" | head -10
echo
echo "Submitted clean-rerun-v1 FOCUSED: 3 jobs, 9 mode-cells, ~40 GPU-hours."
echo "Pre-fix numbers retained for: llm_only, golden_passage, snap_only_in_final"
echo "(those modes don't retrieve, so the prompt bug had no impact)."
