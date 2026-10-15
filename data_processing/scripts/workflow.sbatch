#!/usr/bin/env bash
#SBATCH -p normal.120h
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000M
#SBATCH -t 5-00:00:00
#SBATCH -J vcc_stats
set -euo pipefail
# Submit from the project root: sbatch data_processing/scripts/workflow.sbatch [targets...]
cd "${SLURM_SUBMIT_DIR:?}"
test -f data_processing/workflow/Snakefile
mkdir -p logs/slurm
bash data_processing/scripts/run_job.sh pixi run snakemake -s data_processing/workflow/Snakefile --workflow-profile "data_processing/workflow/profiles/${WORKFLOW_PROFILE:-slurm}" --rerun-incomplete --keep-going "$@"
