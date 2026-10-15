# coverage

How well do the summarized contexts cover a challenge panel (its 18,533 genes and its target list), and what
does a missing gene cost? The panel comes from a bundle folder with `manifest.json`, `pert_counts.csv` and
`context_<X>.h5ad`, such as `data/vcc2026/controls/`.

| Script | Purpose | Output |
| --- | --- | --- |
| `coverage_tables.py` | Per-target, per-gene and per-context coverage tables of a panel | `results/coverage/<panel_id>/` |
| `missing_gene_cost.py` | Cost of each context's missing panel genes on H1's significant gene sets | `results/coverage/<panel_id>/` |
| `coverage_report.py` | One-page HTML report from the tables above (plus mergeability, when present) | `<panel dir>/report.html` |
| `estimate_coverage.py` | Early estimate from raw metadata only, before any summaries existed | `results/coverage_estimate/` |

```bash
pixi run python experiments/coverage/coverage_tables.py --bundle data/vcc2026/controls
pixi run python experiments/coverage/missing_gene_cost.py --bundle data/vcc2026/controls
pixi run python experiments/coverage/coverage_report.py results/coverage/vcc2026-val-1
pixi run python experiments/coverage/estimate_coverage.py results/coverage_estimate
```

`coverage_tables.py` marks contexts added after branch `main` by reading `main:workflow/config.yaml` through
git, so it must run inside the repository.
