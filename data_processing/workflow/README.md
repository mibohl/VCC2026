# workflow

| File | Purpose |
| --- | --- |
| `Snakefile` | Rules for H1 download/validation, vocabulary, source metadata, gene mapping and one `summarize` rule per context |
| `config.yaml` | One entry per context: reader (`src/clean/readers.py`), reader arguments and Slurm resources |
| `contexts.csv` | Descriptive metadata per context: cell line, condition, modality, chemistry, lab, source, license |
| `profiles/slurm/` | Default Slurm profile (Euler partitions) |
| `profiles/leomed/` | LeoMed profile, selected with `WORKFLOW_PROFILE=leomed` |

The Snakefile expects to run from the project root, so its paths (`data/...`, `logs/...`) are relative to
the root. Launch it through `data_processing/scripts/workflow.sbatch`, or check it with `pixi run dry-run`.
Adding a context means a new `config.yaml` entry, a matching `contexts.csv` row and, if needed, a reader.
