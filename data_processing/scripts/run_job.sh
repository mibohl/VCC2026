#!/usr/bin/env bash
# All paths are relative to the project root, independently of the caller's directory.
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.."
: "${SLURM_JOB_ID:?Run data work through Slurm}"
if [[ -n "${LMOD_ROOT:-}" ]]; then
    . "$LMOD_ROOT/lmod/init/bash"
    ml eth_proxy
fi
if [[ -n "${PIXI_HOME:-}" ]]; then
    export PATH="$PIXI_HOME/bin:$PATH"
fi
export PYTHONPATH="$PWD/data_processing/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
bash data_processing/scripts/disk_guard.sh $$ &
guard=$!
trap 'kill "$guard" 2>/dev/null || true' EXIT
"$@"
