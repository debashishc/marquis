#!/bin/bash -l
#SBATCH --job-name=marquis-extract
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

export MARQUIS_MODULES="${MARQUIS_MODULES-cuda/toolkit/12.8.1-1}"
source "${MARQUIS_SLURM_ENV:-slurm/templates/cluster_env.sh}"

export MAGMAR_VIDEO_ROOT="${MAGMAR_VIDEO_ROOT:?Set MAGMAR_VIDEO_ROOT}"
export MAGMAR_QUERIES_JSONL="${MAGMAR_QUERIES_JSONL:?Set MAGMAR_QUERIES_JSONL}"
export MARQUIS_TOPIC_MAPPING="${MARQUIS_TOPIC_MAPPING:-examples/data/topic_video_mapping_sample.json}"

srun scripts/run_extraction.sh
