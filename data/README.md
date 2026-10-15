# data

Inputs and processed data. Not in git (only this README is); everything here is produced by
`data_processing/` or downloaded by it. About 726 GB in total, most of it in `raw/`.

| Path | Contents | Produced by |
| --- | --- | --- |
| `<context>.h5`, `<context>.json` | One summary per context (56): vocabulary-ordered control statistics, per-target mean CPM, LFC, dispersion, MWU statistics, split-half reliability; the JSON holds QC counts and provenance | `summarize` rule (`data_processing/src/clean/run.py`) |
| `vocab_ensembl.tsv` | Ensembl ids of the VCC2026 vocabulary, used for gene matching | `ensembl` rule |
| `vcc2025/` | The three H1 splits (`adata_{Training,Validation,Test}.h5ad`), `validated.json`, and `h1_reference_20k.h5ad` | `prepare_h1` rule; `tools/h1_evaluator/h1_reference_20k.py` |
| `vcc2026/gene_names.csv` | The frozen 18,533-gene VCC2026 vocabulary | `vocabulary` rule |
| `vcc2026/controls/` | VCC2026 validation bundle (panel `vcc2026-val-1`): control cells of contexts A, B, C, `pert_counts.csv`, `manifest.json` | challenge download |
| `orion_probe/` | Orion probe gene metadata | `prepare_orion` rule |
| `raw/<source>/` | Raw downloads and author files per source; staged count files are deleted after their summary is written | `data_processing/scripts/fetch_raw.py`, prepare rules |
| `raw/_clean_tmp/` | Scratch space of running summaries (watched by `disk_guard.sh`) | `summarize` rule |
| `provenance/` | One download receipt per raw file: URL, size, SHA-256, upstream MD5 | `data_processing/src/downloads.py` |

Receipt file names are the SHA-256 of the raw file's absolute path at download time; the receipts here
were written on the cluster (`/cluster/work/boeva/VCC/VCC2026/data/raw/...`).
