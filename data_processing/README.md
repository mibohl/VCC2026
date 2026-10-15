# data_processing

Snakemake workflow that downloads the public perturbation screens and condenses each context into a summary
file in `data/`. Run every command below from the project root.

| Folder | Contents |
| --- | --- |
| [`workflow/`](workflow/) | `Snakefile`, `config.yaml` (contexts, readers, resources), `contexts.csv` (metadata), Slurm profiles |
| [`src/`](src/) | Ingesters, downloads, remote HDF5, and `clean/` (readers, gene/label mapping, statistics, `run.py`) |
| [`scripts/`](scripts/) | Preparation and fetch scripts, R extractors, Slurm wrappers |
| [`tests/`](tests/) | Synthetic smoke test and a known-answer check |

`AUDIT.md` (local only, not in git) records the audit of this pipeline.

## Setup

```bash
pixi install --locked
pixi run smoke
pixi run dry-run
```

## Prepare H1 or process all data

```bash
# Downloads the full original Training, Validation, and Test H5AD files (~34 GB).
sbatch data_processing/scripts/workflow.sbatch prepare_h1

# Generates all prerequisites and computes all context summaries.
sbatch data_processing/scripts/workflow.sbatch

# A single context, including its prerequisites.
sbatch data_processing/scripts/workflow.sbatch data/adamson__k562.h5
```

Snakemake builds a dependency graph from the requested outputs. The default `all`
target prepares the source metadata, vocabulary and gene mapping, then runs the
summary rule for each of the 56 contexts in `workflow/config.yaml`. `prepare_h1`
downloads and validates only the three H1 splits. Existing outputs are reused;
missing or outdated outputs trigger their dependent rules. The Slurm profile submits
individual rules with their declared CPU, memory and runtime limits; per-rule logs
record progress and failures (`logs/stats/<context>.log`). Use
`WORKFLOW_PROFILE=leomed` for the LeoMed profile. H1 prediction evaluation is launched
separately (see `tools/h1_evaluator/`).

Summary outputs are `data/<context>.h5` and `data/<context>.json`.
The HDF5 contains the ordered VCC2026 vocabulary, measured-gene mask, control statistics,
and per-target mean CPM, LFC, NB dispersion, MWU statistics, cell/depth summaries,
and split-half reliability.
- Unmeasured genes remain NaN.
- Library sizes use measured VCC2026 vocabulary genes.
- Cells below 1,000 UMI are removed.

The contexts cover H1, Adamson, three Frangieh conditions, Sunshine, two Feng screens,
DLD1, hESC, VIPerturb, two Orion lines, Replogle 2020, Replogle GWPS plus four Replogle/Nadig
contexts, KOLF, three Marson CD4 T-cell conditions, three McFaline-Figueroa glioblastoma lines
and 30 Jiang 2025 line-by-pathway contexts.

Feng annotation pairs come from the authors' LFC table; effect estimates are not used.

Staged non-H1 count files are deleted after successful summary generation. KOLF,
Replogle/Nadig, and CD4 inputs are streamed through HTTP ranges. Orion shards are staged
one at a time with prefetch. Receipts in `data/provenance/` retain source URLs, sizes,
SHA-256 hashes, and available upstream MD5s.
Failed jobs retain partial downloads for inspection/resume.

## Small verification only

`pixi run smoke` uses 24 genes, 32 controls, and 40 synthetic perturbed cells, plus tiny
metadata fixtures. It checks CPM/LFC/MWU calculations, integer count generation,
prediction validation, preservation of large LFCs, stale/incomplete runs, shared controls, and global
moment merging. `pixi run dry-run` constructs the DAG without downloads or Slurm submission.
Routine verification never processes real matrices or launches full H1 scoring.

`tests/known_answer.py --context <context>` recomputes one context's summary with SciPy and with
cell-eval2's own DE on the same cells and writes `results/known_answer_<context>.json`.
