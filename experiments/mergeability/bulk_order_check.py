"""Per target: fold change from averaged per-cell CPM against fold change from summed raw counts, and the matching log deltas."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "data_processing/src/clean")]
from effect_space_check import half_sums  # noqa: E402
from genes import load_vocab  # noqa: E402

EPS, GATE = 1e-9, 5.0


def main(contexts):
    vocab = load_vocab()
    rows = []
    for c in contexts:
        with h5py.File(ROOT / f"data/{c}.h5") as h:
            targets = [x.decode() if isinstance(x, bytes) else str(x) for x in h["pert/targets"][:]]
            nlp = h["pert/nlog10p"][:]
        hs = half_sums(c, targets, vocab, 0)
        K = len(targets)
        pair = lambda a: a[0:2 * K:2] + a[1:2 * K:2]
        S, F, Q, n, L = pair(hs["S"]), pair(hs["F"]), pair(hs["Q"]), pair(hs["n"]), pair(hs["L"])
        cS, cF, cQ, cn, cL = (hs[k][-2] + hs[k][-1] for k in ("S", "F", "Q", "n", "L"))
        mean_cpm_c = cF / cn * 1e6
        gated = np.isfinite(mean_cpm_c) & (mean_cpm_c > GATE)
        lfc_mean = np.log2((F / n[:, None] * 1e6 + EPS) / (mean_cpm_c + EPS))
        lfc_pool = np.log2((S / L[:, None] * 1e6 + EPS) / (cS / cL * 1e6 + EPS))
        d_cell = Q / n[:, None] - cQ / cn
        d_bulk = np.log1p(5e4 * S / L[:, None]) - np.log1p(5e4 * cS / cL)
        for i, t in enumerate(targets):
            g = gated & np.isfinite(lfc_mean[i]) & np.isfinite(lfc_pool[i])
            if n[i] < 20 or g.sum() < 100:
                continue
            a, b = lfc_mean[i, g], lfc_pool[i, g]
            sig = nlp[i, g] > 3
            rows.append(dict(context=c, target=t, cells=int(n[i]), depth_ratio=(L[i] / n[i]) / (cL / cn),
                             r_lfc=np.corrcoef(a, b)[0, 1], med_abs_diff_lfc=np.median(np.abs(a - b)),
                             med_abs_lfc=np.median(np.abs(a)),
                             r_lfc_sig=np.corrcoef(a[sig], b[sig])[0, 1] if sig.sum() > 10 else np.nan,
                             n_sig=int(sig.sum()), r_delta=np.corrcoef(d_cell[i, g], d_bulk[i, g])[0, 1],
                             r_lfc_vs_bulkdelta=np.corrcoef(b, d_bulk[i, g])[0, 1],
                             slope_bulkdelta_on_celldelta=np.polyfit(d_cell[i, g], d_bulk[i, g], 1)[0]))
        print(c, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/mergeability/bulk_order_check.csv"
    df.to_csv(out, index=False)
    q = df.groupby("context")[["cells", "depth_ratio", "r_lfc", "med_abs_diff_lfc", "med_abs_lfc", "r_lfc_sig", "r_delta",
                               "r_lfc_vs_bulkdelta", "slope_bulkdelta_on_celldelta"]].median()
    print(q.round(3).to_string())
    for c, d in df.groupby("context"):
        strong = d[d.n_sig >= 100]
        print(c, "targets", len(d), "strong", len(strong), "r_lfc strong median", round(strong.r_lfc.median(), 3),
              "r_lfc 5th pct", round(d.r_lfc.quantile(.05), 3), "depth ratio 5-95%", d.depth_ratio.quantile([.05, .95]).round(2).tolist())
        w = d.nsmallest(5, "r_lfc")[["target", "cells", "depth_ratio", "r_lfc", "r_lfc_sig"]]
        print(w.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
