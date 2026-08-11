#!/bin/bash -l
#SBATCH --job-name=marquis-generate
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

export MARQUIS_CLAIMS_PATH="${MARQUIS_CLAIMS_PATH:?Set MARQUIS_CLAIMS_PATH}"
export MARQUIS_REPORT_OUTPUT_DIR="${MARQUIS_REPORT_OUTPUT_DIR:-outputs/reports}"

srun scripts/run_generation.sh
