#!/bin/bash
#SBATCH --job-name=opsd_pc_env
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_env_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
bash "$REPO/scripts/hpc/setup_opd_positive_control_env.sh"
