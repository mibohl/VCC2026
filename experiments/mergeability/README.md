# mergeability

Do shared targets have the same effects across contexts, relative to each context's own split-half
reliability, and does pooling contexts help predict a held-out one? Outputs go to `results/mergeability*/`.

| Script | Purpose | Output |
| --- | --- | --- |
| `mergeability.py` | Pairwise agreement of shared targets' effects between contexts | `results/mergeability/` (`pairs.csv`, `pair_summary.csv`) |
| `mergeability_design.py` | Pair agreement grouped by what differs: line, lab, chemistry, condition | `results/mergeability/design/` |
| `pooling_loco.py` | Leave-one-context-out: predict a held-out context from the others | `results/mergeability/loco/` |
| `effect_space_check.py` | Split-half agreement recomputed in the scorer's CPM and group-sum spaces | `results/mergeability/space_check/` |
| `bulk_order_check.py` | Averaged per-cell CPM vs summed-count fold changes per target | `results/mergeability/bulk_order_check.csv` |
| `donor_split_check.py` | CD4 T reliability from a random split vs a split by donor | `results/mergeability/cd4t_donor_split/` |

```bash
pixi run python experiments/mergeability/mergeability.py --effects auto
pixi run python experiments/mergeability/mergeability_design.py
pixi run python experiments/mergeability/pooling_loco.py --held-out vcc2025__H1
pixi run python experiments/mergeability/bulk_order_check.py adamson__k562 replogle_nadig__k562
pixi run python experiments/mergeability/donor_split_check.py Rest
```

`effect_space_check.py` and `donor_split_check.py` re-read raw cells, so they need the raw files and are
best run through Slurm. Older runs kept for comparison: `results/mergeability_{kolf,pooled,pre_m1,pre_rerun,test,m1_test}/`.
