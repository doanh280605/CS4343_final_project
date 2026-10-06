#!/bin/bash -l
#SBATCH --job-name=landcover-a30
#SBATCH --account=ece341x
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A30
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/turing-train-%j.log

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/turing_environment.sh
phase="${1:-train}"
case "$phase" in train|test) ;; *) echo 'Use train or test.' >&2; exit 1 ;; esac
test -f outputs/turing-study/protocol.json
mkdir -p outputs/turing-study/sessions
nvidia-smi > "outputs/turing-study/sessions/gpu-$SLURM_JOB_ID.txt"
.venv-turing/bin/python - <<'PY'
import json
import os
from pathlib import Path

import torch

from landcover.engine import provenance

assert torch.cuda.is_available(), 'SLURM did not provide a usable CUDA GPU'
x = torch.ones(32, 32, device='cuda')
assert (x @ x).mean().item() == 32
record = {
    **provenance(),
    'gpu': torch.cuda.get_device_name(0),
    'cuda': torch.version.cuda,
    'slurm_job': os.environ['SLURM_JOB_ID'],
    'slurm_account': os.environ.get('SLURM_JOB_ACCOUNT'),
    'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
}
path = Path('outputs/turing-study/sessions') / f"session-{record['slurm_job']}.json"
path.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
PY
if [[ "$phase" == test ]]; then
    .venv-turing/bin/python -m landcover.cli study-test \
        --study outputs/turing-study --allow-test
else
    .venv-turing/bin/python -m landcover.cli study-run \
        --study outputs/turing-study --retry-incomplete
fi
