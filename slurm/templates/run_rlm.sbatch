#!/bin/bash -l
#SBATCH --job-name=marquis-rlm
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

export MARQUIS_MODULES="${MARQUIS_MODULES-cuda/toolkit/12.8.1-1}"
source "${MARQUIS_SLURM_ENV:-slurm/templates/cluster_env.sh}"

export MAGMAR_VIDEO_ROOT="${MAGMAR_VIDEO_ROOT:?Set MAGMAR_VIDEO_ROOT}"
export MAGMAR_QUERIES_JSONL="${MAGMAR_QUERIES_JSONL:?Set MAGMAR_QUERIES_JSONL}"
export MARQUIS_CLAIMS_PATH="${MARQUIS_CLAIMS_PATH:?Set MARQUIS_CLAIMS_PATH}"
export MARQUIS_QUERY_ID="${MARQUIS_QUERY_ID:-1}"
export MARQUIS_RLM_OUTPUT_DIR="${MARQUIS_RLM_OUTPUT_DIR:-outputs/rlm}"

srun scripts/run_rlm.sh
