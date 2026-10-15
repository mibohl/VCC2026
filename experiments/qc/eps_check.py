"""Per context: how extreme fold changes are for genes below and above the 5 CPM control gate."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(0)
rows = []
for f in sorted((ROOT / "data").glob("*.h5")):
    with h5py.File(f) as h:
        key = "pert/lfc_batchmatched" if "pert/lfc_batchmatched" in h else "pert/lfc"
        T = h[key].shape[0]
        idx = np.sort(rng.choice(T, min(300, T), replace=False))
        lfc = h[key][idx]
        cpm = h["control/mean_cpm"][:]
        meas = h["measured"][:].astype(bool)
    for name, g in (("below_gate", meas & (cpm <= 5)), ("zero_in_controls", meas & (cpm == 0)), ("above_gate", meas & (cpm > 5))):
        v = np.abs(lfc[:, g])
        v = v[np.isfinite(v)]
        rows.append(dict(context=f.stem, genes=name, n_genes=int(g.sum()), median_abs=np.median(v) if v.size else np.nan,
                         p99_abs=np.quantile(v, .99) if v.size else np.nan, frac_abs_gt_5=(v > 5).mean() if v.size else np.nan))
df = pd.DataFrame(rows)
out = ROOT / "results/qc"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "eps_check.csv", index=False)
print(df.pivot(index="context", columns="genes", values=["n_genes", "p99_abs", "frac_abs_gt_5"]).round(3).to_string())
