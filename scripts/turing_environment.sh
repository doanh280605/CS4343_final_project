#!/usr/bin/env bash
# Source only inside a SLURM allocation, from the repository root.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit the setup/training script with sbatch, not bash on a login node}"
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f configs/final-study.yaml
module load python/3.12.10/bgfyood uv/0.7.2 cuda12.6/toolkit/12.6.3
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LANDCOVER_SCRATCH="/scratch/$USER/cs4343-landcover"
export UV_CACHE_DIR="$LANDCOVER_SCRATCH/cache/uv"
export TORCH_HOME="$LANDCOVER_SCRATCH/cache/torch"
mkdir -p .cache outputs "$LANDCOVER_SCRATCH"
# Prevent two jobs from changing this checkout's environment/study simultaneously.
exec 9>.cache/turing-job.lock
flock -n 9 || { echo 'Another setup/training job is using this checkout.' >&2; exit 1; }
