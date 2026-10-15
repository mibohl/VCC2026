# scripts

Preparation scripts and job wrappers of the pipeline. Shell wrappers change to the project root first, so
they work from any directory.

| Script | Purpose |
| --- | --- |
| `workflow.sbatch` | Slurm entry point: runs Snakemake with a workflow profile (`sbatch data_processing/scripts/workflow.sbatch [targets]`) |
| `run_job.sh` | Wraps every job: checks it runs under Slurm, sets threads, caches and `PYTHONPATH`, starts `disk_guard.sh` |
| `disk_guard.sh` | Cancels a job when `data/raw/_clean_tmp` grows past 245 GB |
| `dry_run.sh` | Snakemake dry run (`pixi run dry-run`) |
| `slurm_status.sh` | Job status callback for the Snakemake cluster executor |
| `prepare_data.py` | Prerequisite steps called by the workflow: H1 download/validation, vocabulary, probe set, Orion, Feng, DLD1, VIPerturb, Ensembl map |
| `vocabulary.py` | The frozen 18,533-gene VCC2026 vocabulary |
| `fetch_raw.py` | Downloads every raw source into `data/raw/<source>/` with a receipt |
| `fetch_cd4t_obs.py`, `fetch_cd4t_s3_lanes.py` | CD4 T helpers: per-cell metadata and the two Stim8hr lanes truncated on GEO |
| `summarize_scperturb.py` | scPerturb study table and count-selection helpers |
| `gse337988_coldata.R`, `viperturb_extract.R`, `mcfaline_extract.R` | Convert RDS objects (DLD1; VIPerturb and Jiang; McFaline-Figueroa) into flat files for Python |
