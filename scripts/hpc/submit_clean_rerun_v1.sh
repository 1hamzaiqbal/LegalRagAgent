#!/bin/bash
# Clean-rerun v1 submission — fires after sign-off in docs/rigour_signoff.md.
#
# DO NOT EXECUTE THIS BLINDLY. The rigour-architect (you, in the next session)
# must confirm:
#   1. Cluster repo is at commit 3d5ff05 or later (`git log --oneline -1`)
#   2. tests/test_formatter.py + tests/test_sanitizer.py pass
#   3. N=200 validation runs 51205 (E4B) and 51206 (31B) landed without
#      crashes and with 0% HyDE leak
#   4. The accuracy delta from pre-fix to post-fix at N=200 looks reasonable
#      (expected: post-fix > pre-fix on most modes, since model now sees
#      37% more fact patterns)
#
# Then tag the commit:  ssh wustl 'cd /engrfs/.../LegalRagAgent-clean && git tag clean-rerun-v1'
#
# And run this script (from cluster):
#   bash scripts/hpc/submit_clean_rerun_v1.sh

set -e

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean
cd "$REPO"

# Sanity: at least one of the recent fix commits must be in the log.
if ! git log --oneline -20 | grep -q "3d5ff05"; then
  echo "ERROR: cluster repo missing 3d5ff05 (retrieval prompt fix). Pull first." >&2
  exit 1
fi

echo "Cluster repo at: $(git log --oneline -1)"
echo

# === Full N=1195 cross-scale matrix at clean-rerun-v1 ===
# 4 models × 4 core modes (rag_simple, rag_hyde, rag_snap_hyde, snap_only_in_final)
# Skip extended modes (subagent variants, golden_passage, llm_only) for now —
# add to a v2 wave once the core matrix lands clean.

# E2B (4B, A40-fits) — 4 modes seq, ~12h
sbatch -t 14:00:00 \
  --export=ALL,MODEL=google/gemma-4-E2B-it,PORT=8070,MODES="rag_simple rag_hyde rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-rerun-v1-e2b \
  scripts/hpc/slurm_gemma4_rerun.sh

# E4B (8B, A40-fits) — 4 modes seq
sbatch -t 18:00:00 \
  --export=ALL,MODEL=google/gemma-4-E4B-it,PORT=8071,MODES="rag_simple rag_hyde rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.8,TAG_SUFFIX=clean-rerun-v1-e4b \
  scripts/hpc/slurm_gemma4_rerun.sh

# 26B-A4B (a100-sxm4 80GB) — 4 modes seq
sbatch -t 18:00:00 \
  --export=ALL,MODEL=google/gemma-4-26B-A4B-it,PORT=8072,MODES="rag_simple rag_hyde rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=8192,GPU_MEM_UTIL=0.90,TAG_SUFFIX=clean-rerun-v1-26b \
  scripts/hpc/slurm_gemma4_rerun_80gb.sh

# 31B (h100, 80GB tight) — 4 modes seq
sbatch --nodelist=h100-2405 -t 24:00:00 \
  --export=ALL,MODEL=google/gemma-4-31B-it,PORT=8073,MODES="rag_simple rag_hyde rag_snap_hyde snap_only_in_final",N_QUESTIONS=full,MAX_MODEL_LEN=4096,GPU_MEM_UTIL=0.92,TAG_SUFFIX=clean-rerun-v1-31b \
  scripts/hpc/slurm_gemma4_rerun.sh

echo
echo "=== queue ==="
squeue -u hiqbal --format="%i %j %T %M %R" | head -20
echo
echo "Submitted clean-rerun-v1 wave: 4 jobs (E2B, E4B, 26B, 31B × 4 core modes)."
echo "Expected total wall: ~24h (parallel where GPUs allow)."
