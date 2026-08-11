#!/bin/bash -l
#SBATCH --job-name=marquis-eval
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

source "${MARQUIS_SLURM_ENV:-slurm/templates/cluster_env.sh}"

export MARQUIS_QRELS="${MARQUIS_QRELS:?Set MARQUIS_QRELS}"
export MARQUIS_RETRIEVAL_DIR="${MARQUIS_RETRIEVAL_DIR:-outputs/retrieval}"
export MARQUIS_EVAL_CSV="${MARQUIS_EVAL_CSV:-outputs/evaluation/retrieval_results.csv}"

srun scripts/run_evaluation.sh
