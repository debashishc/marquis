# SLURM

The templates in `slurm/templates/` are generic SLURM batch scripts.
Filled copies belong in `slurm/internal/`, which is ignored by Git.

Use this flow from the repo root:

```bash
mkdir -p slurm/internal
cp slurm/templates/*.sbatch slurm/internal/
mkdir -p logs/slurm
```

Then edit the copied files:

- replace `CHANGE_ME_ACCOUNT`;
- keep CPU-only jobs on `--partition=cpu`;
- keep GPU jobs on `--partition=gpu` and name the GPU type if your cluster
  requires it, for example `--gres=gpu:a100:1`;
- set `--cpus-per-task`, `--mem`, and `--time` to the measured workload;
- set `MARQUIS_VENV` to the project virtualenv if the `marquis-*` commands are
  not already on `PATH`;
- set `MAGMAR_VIDEO_ROOT`, `MAGMAR_QUERIES_JSONL`, and any artifact paths.

The templates use `#!/bin/bash -l` so lmod is available inside the job. GPU
templates load `cuda/toolkit/12.8.1-1` through `MARQUIS_MODULES` by default; set
`MARQUIS_MODULES=""` if the active environment already provides CUDA.

`slurm/templates/cluster_env.sh` keeps Hugging Face, uv, pip, conda, and
Apptainer caches under `MARQUIS_CACHE_BASE` (defaults to `$HOME/.cache/marquis`).
If you have fast scratch storage, submit with:

```bash
export MARQUIS_CACHE_BASE=/scratch/$USER
```

The templates launch work with `srun` so SLURM accounts the job step and passes
signals cleanly. They do not add `--signal` or `--requeue` by default because
the MARQUIS wrappers do not checkpoint on walltime signals.

For large datasets with many small files, stage inputs into `$TMPDIR` at job
start and copy outputs back before the job exits.

Useful checks:

```bash
sinfo -p gpu -o "%G %D %t %m %c"
squeue --me
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocCPUs,AllocTRES -X
```

Do not run heavy preprocessing, model loading, or GPU work on login nodes. Use
`sbatch`, `salloc`, or `srun --pty`.
