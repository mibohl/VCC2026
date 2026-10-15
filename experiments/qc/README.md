# qc

Do the summaries agree with what the authors published, and are their fold changes well behaved?
Outputs go to `results/qc/` or to the console.

| Script | Purpose | Output |
| --- | --- | --- |
| `compare_authors.py` | Our per-target fold changes against the authors' released ones; sources `replogle`, `dld1`, `jiang`, `feng`, `cd4t` | `results/qc/compare_authors_<sources>.csv` |
| `dld1_halves.py` | DLD1: the authors' Low1 vs Low2 fold changes per promoter, and ours against each | console |
| `eps_check.py` | Per context: how extreme fold changes are below and above the 5 CPM control gate | `results/qc/eps_check.csv` |
| `kolf_check.py` | Same-target vs other-target split-half correlation, before and after removing the mean response | console |

```bash
pixi run python experiments/qc/compare_authors.py replogle dld1 jiang feng cd4t
pixi run python experiments/qc/dld1_halves.py
pixi run python experiments/qc/eps_check.py
pixi run python experiments/qc/kolf_check.py kolf__hiPSC
```
