#!/bin/bash -l
#SBATCH --job-name=landcover-setup
#SBATCH --account=ece341x
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/turing-setup-%j.log

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/turing_environment.sh
bundle="${1:?Pass the path to the existing landcover-colab.zip bundle}"
test -f "$bundle"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [[ ! -x .venv-turing/bin/python ]]; then
    uv venv --python "$(command -v python3)" .venv-turing
fi
uv pip install --python .venv-turing/bin/python -r configs/colab-requirements.txt
uv pip install --python .venv-turing/bin/python \
    'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
    --constraint configs/colab-requirements.txt \
    --index-url https://download.pytorch.org/whl/cu124
.venv-turing/bin/python scripts/turing_prepare.py \
    --bundle "$bundle" --raw-root "$LANDCOVER_SCRATCH/raw"
uv pip freeze --python .venv-turing/bin/python > "outputs/turing-study/setup-packages-$SLURM_JOB_ID.txt"
echo 'Setup complete. Submit scripts/turing_train.sbatch to start training.'
