#!/usr/bin/env bash
# Run on the Turing login node; only submits work, never trains there.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
bundle="${1:-$HOME/landcover-colab.zip}"
test -f "$bundle" || { echo "Missing team bundle: $bundle" >&2; exit 1; }
bundle="$(realpath "$bundle")"
mkdir -p outputs
setup_job="$(sbatch --parsable scripts/turing_setup.sbatch "$bundle")"
setup_job="${setup_job%%;*}"
echo "Setup job: $setup_job"
train_job="$(sbatch --parsable --dependency="afterok:$setup_job" \
    --kill-on-invalid-dep=yes scripts/turing_train.sbatch)"
echo "Training job: $train_job"
echo 'Monitor with: squeue --me'
echo "Setup log: outputs/turing-setup-$setup_job.log"
echo "Training log: outputs/turing-train-${train_job%%;*}.log"
