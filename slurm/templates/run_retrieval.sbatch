#!/bin/bash -l
#SBATCH --job-name=marquis-retrieval
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

source "${MARQUIS_SLURM_ENV:-slurm/templates/cluster_env.sh}"

export MAGMAR_QUERIES_JSONL="${MAGMAR_QUERIES_JSONL:?Set MAGMAR_QUERIES_JSONL}"
export MARQUIS_EXPANDED_QUERIES="${MARQUIS_EXPANDED_QUERIES:-outputs/retrieval/expanded_queries.json}"
export MARQUIS_SUBQUERY_RESULTS="${MARQUIS_SUBQUERY_RESULTS:?Set MARQUIS_SUBQUERY_RESULTS}"

srun scripts/run_retrieval.sh
