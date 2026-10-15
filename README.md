# VCC 2026

Virtual Cell Challenge 2026 repo on tunguska

## Layout

| Folder | What it holds |
| --- | --- | 
| [`data_processing/`](data_processing/) | Snakemake workflow, readers and download scripts that turn public perturbation screens into per-context summaries |
| [`data/`](data/) | Processed summaries (`<context>.h5` / `.json`), raw downloads, H1 and VCC2026 reference files |
| [`tools/`](tools/) | Reusable parts: the cell generator and the H1 evaluator | 
| [`experiments/`](experiments/) | The playground: analyses built on the summaries (mergeability, coverage, QC, H1 baselines) |
| [`results/`](results/) | Outputs of experiments and evaluations |
| [`logs/`](logs/) | Slurm, Snakemake and ad hoc job logs |

## Setup

```bash
pixi install --locked
pixi run smoke      # small synthetic checks, no downloads
pixi run dry-run    # build the Snakemake DAG without running anything
```

## Conventions

- Run everything from the project root. Scripts locate the root from their own path, so paths such as
  `data/<context>.h5` and `results/...` resolve the same way from any folder.
- Code that turns raw data into `data/` lives in `data_processing/`. Code meant to be imported or reused by
  several analyses lives in `tools/`. Everything exploratory starts in `experiments/<topic>/` and writes to
  `results/<topic>/`.
- Large files never go into git: `data/`, `results/` and `logs/` are ignored apart from their README.

