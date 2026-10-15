# src

Library code of the pipeline. `data_processing/scripts/run_job.sh` puts this folder on `PYTHONPATH`; scripts
that run on their own add it to `sys.path` themselves.

| Module | Purpose |
| --- | --- |
| `downloads.py` | Verified, resumable downloads with receipts in `data/provenance/` |
| `remote_h5.py`, `add_dispersion.py` | Read-only HDF5 over HTTP byte ranges and AnnData categorical helpers |
| `ingest_*.py` | Source-specific parsing (Feng, hESC GSE295214, DLD1 GSE337988, Orion, VIPerturb) |
| `clean/readers.py` | One reader per source; yields control and perturbed count blocks with labels and batches |
| `clean/genes.py`, `clean/labels.py` | Mapping of native genes and target labels onto the 18,533-gene VCC2026 vocabulary |
| `clean/stats.py` | Streaming statistics engine and the summary HDF5 writer |
| `clean/run.py` | Entry point of the `summarize` rule: one context in, `data/<context>.h5` and `.json` out |
