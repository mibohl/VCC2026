"""CD4 T reliability from a random cell split against a split by donor (D1+D2 vs D3+D4), same targets and genes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "data_processing/src"), str(ROOT / "data_processing/src/clean")]
from readers import CD4T

EXTRA = ["NRAS", "PARP14", "IFFO2", "MPP5", "OFD1", "INSR", "RASAL3", "SLC38A2", "KAT6A", "NKX2-5", "PPFIA4"]
GROUP = {"D1": 0, "D2": 0, "D3": 1, "D4": 1}


def lognorm(X):
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(tot, 1)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def main(cond="Rest", n_random=300, out=ROOT / "results/mergeability/cd4t_donor_split"):
    rng = np.random.default_rng(0)
    r = CD4T(cond)
    G = None
    csum, ccount = None, np.zeros(2)
    for X, bat in r.controls():
        g = np.array([GROUP[b] for b in bat])
        L = lognorm(X)
        if csum is None:
            G = L.shape[1]
            csum, cpm = np.zeros((2, G)), np.zeros(G)
        for k in (0, 1):
            csum[k] += np.asarray(L[g == k].sum(0)).ravel()
            ccount[k] += (g == k).sum()
        tot = np.asarray(X.sum(1)).ravel()
        cpm += np.asarray((sp.diags(1e6 / np.maximum(tot, 1)) @ X).sum(0)).ravel()
    gated = cpm / ccount.sum() > 5
    ctrl = csum / ccount[:, None]
    targets = sorted(set(rng.choice(sorted(set(r.targets)), n_random, replace=False)) | (set(EXTRA) & set(r.targets)))
    ti = {t: i for i, t in enumerate(targets)}
    T = len(targets)
    s_donor, n_donor = np.zeros((T, 2, G)), np.zeros((T, 2))
    s_half, n_half = np.zeros((T, 2, G)), np.zeros((T, 2))
    for X, lab, bat in r.cells():
        keep = np.array([t in ti for t in lab])
        if not keep.any():
            continue
        L = lognorm(X[keep])
        t = np.array([ti[x] for x in lab[keep]])
        g = np.array([GROUP[b] for b in bat[keep]])
        h = rng.integers(0, 2, len(t))
        for split, s, n in ((g, s_donor, n_donor), (h, s_half, n_half)):
            for k in (0, 1):
                m = split == k
                M = sp.csr_matrix((np.ones(m.sum()), (t[m], np.arange(m.sum()))), shape=(T, m.sum()))
                s[:, k] += (M @ L[m]).toarray()
                np.add.at(n[:, k], t[m], 1)
    pooled = ctrl.T @ (ccount / ccount.sum())
    rows = []
    for i, t in enumerate(targets):
        if n_donor[i].min() < 20 or n_half[i].min() < 20:
            continue
        dd = [s_donor[i, k, gated] / n_donor[i, k] - ctrl[k, gated] for k in (0, 1)]
        dh = [s_half[i, k, gated] / n_half[i, k] - pooled[gated] for k in (0, 1)]
        rows.append(dict(target=t, extra=t in EXTRA, cells_donor_1_2=int(n_donor[i, 0]), cells_donor_3_4=int(n_donor[i, 1]),
                         r_random_halves=np.corrcoef(*dh)[0, 1], r_donor_split=np.corrcoef(*dd)[0, 1]))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{cond}.csv", index=False)
    print(df.groupby("extra")[["r_random_halves", "r_donor_split"]].median().round(3))
    print(df[df.extra].round(3).to_string(index=False))
    print("random targets: share with r_random_halves > 0.3:", (df[~df.extra].r_random_halves > .3).mean().round(3),
          " r_donor_split > 0.3:", (df[~df.extra].r_donor_split > .3).mean().round(3))


if __name__ == "__main__":
    main(*sys.argv[1:2])
