# results

Outputs of experiments and evaluations. Not in git (only this README is). Each folder is written by one
script; see the experiment's README for details.

| Folder | Written by |
| --- | --- |
| `mergeability/` (`design/`, `loco/`, `space_check/`, `cd4t_donor_split/`, ...) | `experiments/mergeability/` |
| `mergeability_{kolf,pooled,pre_m1,pre_rerun,test,m1_test}/` | Earlier `mergeability.py` runs kept for comparison |
| `coverage/<panel_id>/`, `coverage_test/` | `experiments/coverage/coverage_tables.py`, `missing_gene_cost.py`, `coverage_report.py` |
| `coverage_estimate/` | `experiments/coverage/estimate_coverage.py` (early estimate from raw metadata) |
| `qc/` | `experiments/qc/` |
| `h1_oracle/`, `h1_heldout/`, `h1_heldout_pre_zerofix/`, `h1_bundle_20k/` | H1 evaluations (`tools/h1_evaluator/`) of the baselines in `experiments/h1_baselines/` |
| `known_answer_<context>.json` | `data_processing/tests/known_answer.py` |

New experiments write to `results/<topic>/`, matching their folder in `experiments/`.
