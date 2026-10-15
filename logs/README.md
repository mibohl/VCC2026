# logs

Job logs. Not in git (only this README is).

| Path | Contents |
| --- | --- |
| `stats/<context>.log` | One log per `summarize` job (Snakemake `log:` directive) |
| `slurm/<jobid>.out` | Slurm output of workflow jobs (LeoMed profile) |
| `<name>_<jobid>.log` | Slurm logs of single analysis jobs (mergeability, QC checks, ...) |
| `local/` | Logs of ad hoc jobs, together with some throwaway scripts and fixtures they ran |
| `coverage_estimate/run.log` | Output of the early coverage estimate |
