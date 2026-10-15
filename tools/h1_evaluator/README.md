# h1_evaluator

Scores LFC predictions for the 300 H1 (VCC2025) targets against the real cells with cell-eval2, using the
VCC2026 preset. Requires the three H1 splits in `data/vcc2025/` (`sbatch data_processing/scripts/workflow.sbatch prepare_h1`).

| File | Purpose |
| --- | --- |
| `evaluate_h1.py` | The evaluator (runs inside Slurm only) |
| `evaluate_h1.sbatch` | Slurm wrapper: `sbatch tools/h1_evaluator/evaluate_h1.sbatch --predictions ... --out ...` |
| `encoder.py` | Control/depth fits and the count encoder that turns predicted LFCs into cells |
| `score.py` | Six-member internal H1 scaling with the published VCC2026 midpoint anchors |
| `example_predictions.py` | Writes a zero-LFC prediction, or the H1 summary's own LFCs as an in-sample oracle |
| `h1_reference_20k.py` | Builds `data/vcc2025/h1_reference_20k.h5ad`, a 2026-style reference thinned to 20k UMI (`--reference`) |

## Prediction NPZ fields

| Field | Shape | Meaning |
| --- | --- | --- |
| `genes` | `[G]` | Exact native H1 gene axis, in H5AD order |
| `perts` | `[300]` | All unique H1 target symbols; any order |
| `lfc` | `[300, G]` | Finite predicted log2 fold changes |
| `source` | scalar string | Prediction provenance |
| `lfc_bulk` | `[300, G]`, optional | Separate pooled-bulk LFC channel |

## Usage

Run from the project root:

```bash
pixi run python tools/h1_evaluator/example_predictions.py --out results/zero.npz
sbatch tools/h1_evaluator/evaluate_h1.sbatch --predictions results/zero.npz --out results/zero_eval --seed 0
```

The example uses only gene/target labels. The evaluator uses the shared H1 controls once, scores all 300
targets, and retains reference split identity. The evaluator uses original UMI counts for controls and perturbed references, with seed 0,
up to 400 reference cells and 400 generated cells per target. Cell selection does not
alter the counts of retained cells. Predicted LFCs are not clipped; the separate bulk
channel retains its gain of 0.25. Old evaluation settings cannot reuse completed blocks.

DE runs in 12 blocks of 25 targets. PDS ranking and corrected MSE are calculated once
on the complete 300-target panel using the blocks' sufficient statistics. Outputs include
`per_pert.csv`, `agg.csv`, `score.csv`, `summary.json`, `metadata.json`, and `DONE`.
Completed blocks can resume only with identical recorded inputs/settings.

Options `--holdout` (cells a held-out prediction was built from, see `experiments/h1_baselines/`) and
`--reference` (an alternative reference such as the 20k-UMI file) cannot be combined.
