#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONUNBUFFERED=1

if [[ ! -f outputs/final-study/protocol.json ]]; then
    uv run --no-sync landcover study-prepare
fi
uv run --no-sync landcover study-run "$@"
