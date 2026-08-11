#!/usr/bin/env bash
# Shared setup for SLURM jobs. Source from the repo root.

: "${USER:?USER must be set}"

# Keep caches, managed Python installs, and package metadata off the login home
# filesystem. Set MARQUIS_CACHE_BASE to a fast scratch directory if available.
cache_base="${MARQUIS_CACHE_BASE:-$HOME/.cache/marquis}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_base/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$cache_base/uv/python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$cache_base/uv/tools}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$cache_base/.conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$cache_base/.conda/pkgs}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$cache_base/.apptainer}"

mkdir -p \
  "$HF_HOME" \
  "$PIP_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$UV_TOOL_DIR" \
  "$CONDA_ENVS_PATH" \
  "$CONDA_PKGS_DIRS" \
  "$APPTAINER_CACHEDIR"

# Optional lmod setup. GPU templates set MARQUIS_MODULES to CUDA by default;
# override or clear it if the active environment already carries the right stack.
if [[ -n "${MARQUIS_MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load $MARQUIS_MODULES
fi

# Optional project virtualenv. Example:
#   export MARQUIS_VENV=/path/to/your/project/.venv
if [[ -n "${MARQUIS_VENV:-}" ]]; then
  source "$MARQUIS_VENV/bin/activate"
else
  echo "MARQUIS_VENV is not set; using marquis commands from PATH"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-none}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-none}"
echo "TMPDIR=${TMPDIR:-not set}"
echo "HF_HOME=$HF_HOME"

(git rev-parse HEAD && git status --short) 2>/dev/null || true
