# vcc_decouple

Generates synthetic perturbed cell populations (raw counts) for the Virtual Cell Challenge, where the
**pseudobulk** carries one log2 fold-change vector and the **per-cell average** carries another.

- PDS and MSE read the pooled pseudobulk (sum of counts, so deep cells weigh more): it delivers `lfc_bulk`.
- The four DE metrics read the per-cell CPM average and a Wilcoxon test: they see `lfc_de`.

The gap between the two comes from tilting each gene toward deep or shallow cells.
Pass `lfc_bulk == lfc_de` to get an ordinary population with no decoupling.

Single file, needs only `numpy` and `scipy` (plus `anndata` if you read or write `.h5ad`).

## Quick start

```python
import anndata as ad, numpy as np
import vcc_decouple as vd

ctrl = ad.read_h5ad("context_A.h5ad")       # non-targeting control cells, raw counts
controls = vd.fit(ctrl.X)                   # fit once (about 20 s), reuse for every call

X, obs_target, report = vd.generate(
    controls, lfc_bulk, lfc_de,             # [T, G] (or [G]) log2 fold changes vs control
    target_genes=targets, gene_names=ctrl.var_names,
)
ad.AnnData(X=X, obs={"target_gene": obs_target}, var={"gene": ctrl.var_names}).write_h5ad("pred_A.h5ad")
```

`X` is CSR float32 holding exact integer counts, with `n_cells` rows per target in input order.

## `generate()` arguments

| argument | default | meaning |
|---|---|---|
| `controls` | required | `vd.fit(counts)` result, or a raw control count matrix (fitted on the fly) |
| `lfc_bulk` | required | log2 FC of the pooled composition vs the control's pooled composition (PDS, MSE) |
| `lfc_de` | required | log2 FC of the mean per-cell CPM vs the control's (DE metrics) |
| `target_genes` | `None` | target names for `obs_target` (needed for `is_A`) |
| `gene_names` | `None` | gene names in column order (needed for `is_A`) |
| `n_cells` | `400` | cells per target |
| `seed` | `0` | output is deterministic per (seed, target index), for any `n_jobs` |
| `n_jobs` | all CPUs | worker processes (fork; shared memory on Linux/macOS) |
| `stretch` | `1.0` | depth spread around the fixed median. `1.0` = real control depths; `2.0` = fixed stretch; `"auto"` = per target, the smallest stretch that brings the bulk vector into reach |
| `stretch_tol` | `0.01` | for `"auto"`: allowed share of the bulk signal left out of reach |
| `min_depth`, `max_depth` | `1e3`, `9e5` | depth clip for stretched cells (cell-eval2 rejects cells above 1e6 counts) |
| `confidence` | `None` | `[T, G]` or `[G]` values in (0, 1): how sure you are that a gene moves in the direction of its `lfc_de`. Above 0.5 makes up AND down genes more significant, below 0.5 less, 0.5 = unchanged (see below) |
| `conf_k` | `0.86` | slope: dispersion multiplier `exp(-k * logit(c))` for up genes, its inverse for down genes (c = 0.9: up 0.15x, down 6.6x) |
| `conf_max_widen` | `16` | cap on the widening (down genes at high c, up genes at low c); widening costs DE fold-change precision |
| `conf_eps` | `0.1` | log2 scale of the dead zone: genes with delivered \|lfc_de\| well below it are left alone |
| `exact_bulk` | `True` | pin each gene's group total to its expectation (pseudobulk delivered almost noise-free) |
| `bmax` | `4.0` | maximum per-gene tilt |
| `is_A` | `False` | test the cells against the supplied controls and compare with published val-A statistics; needs exactly 300 targets |

## Output report

- `report["delivery"]`: one dict per target, with:
  - `bulk_strength`, `bulk_cos`: measured pseudobulk in the scorer's log1p(5e4 x) space, i.e. the delivered share of
    the intended delta energy (1 = full) and its direction cosine (1 = same shape)
  - `de_r`, `de_mae`: realised vs requested DE fold changes
  - `bulk_r`, `bulk_mae`: realised vs requested pooled fold changes
  - `n_clip`: genes whose tilt hit `bmax`
  - `split_err`: leftover error of the expected split
  - `depth_min`, `depth_median`, `depth_max`: depths used
- `report["stretch"]`: the stretch used per target.
- `report["split_sigma"]`: per-target dissimilarity σ (sd of `lfc_bulk - lfc_de` over tested genes).
- `report["expected"]`: rough expected DE call stability (`calls_kept`), call count and pseudobulk fidelity per
  target, looked up from the val-A sweep (`stretch=1.0` or `"auto"` only).
- `report["A"]` (with `is_A`): DE genes per target (min / median / mean, harmonic mean, share <= 10), pseudobulk
  energy, UMI and genes per cell, each next to the val-A reference.

## How far apart may the two vectors be?

`generate()` checks this up front and warns when:
- a target's σ exceeds 0.12 (the DE limit) or the pseudobulk limit (0.33 at `stretch=1.0`, 1.04 for `"auto"`).
  The warning includes a rough expectation for DE call stability, call count and, when in reach, pseudobulk fidelity.
- more than 5 % of a target's pseudobulk signal is out of tilt reach. σ cannot predict this case (sparse, large
  splits), so check the measured `bulk_strength` / `bulk_cos` printed after generation.

To check without generating, use `vd.split_sigma(...)`, `vd.expected_fidelity(sigma, stretch)` and
`vd.split_reach(...)`; the last one's `energy_out` is the share of the intended pseudobulk signal that no tilt can reach.

Measured on val-A controls (24 targets, split = per-gene noise of size σ in log2):

| σ (per-gene split) | pseudobulk, `stretch=1.0` | pseudobulk, `stretch="auto"` | DE calls |
|---|---|---|---|
| <= 0.12 | intact | intact | intact (changes no more than a resample) |
| 0.12 to 0.33 | intact | intact | drift, more calls |
| 0.33 to 1.0 | loses strength | intact | drift strongly |

What σ = 0.12 means in practice:
- A typical gene differs by about ±9 % between the two vectors, and 95 % of genes by less than ±18 %.
- Overall, the two vectors still correlate at about r = 0.86.

Stretching widens the pooled range but costs DE precision: DE fold-change correlation falls from 0.97 to 0.80 at stretch 3.
Stretch 1.25 is nearly free.

## Caveats
- **Confidence acts on any gene with |lfc_de| at or above about `conf_eps`.** Small, noisy values become calls too
  (val A/B/C: genes with lfc_de noise of sd 0.12 went from 5 % to 18 % called at c = 0.9). Give c > 0.5 only to genes
  you intend as DE, or raise `conf_eps` for a dense, noisy `lfc_de`.
- **Vectors are delivered up to one global offset.** Each vector is renormalised to a composition, so a vector
  with a net shift arrives shifted (`*_offset` in `delivery`).
- **Cost.** About 5 s per target per core. Peak memory is about 19 GB for 300 x 400 cells (is_A included).

## Self-check

```bash
python vcc_decouple.py   # synthetic data, a few minutes
```
