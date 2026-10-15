"""Leave-one-context-out: predict a held-out context's target effects from the other contexts only."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from mergeability_design import CHEMISTRY, LAB, PLURIPOTENT, condition_kind

ROOT = Path(__file__).resolve().parents[2]
HELD_OUT = ["vcc2025__H1", "gse295214__hesc", "kolf__hiPSC", "viperturb__k562", "replogle_nadig__rpe1",
            "cd4t_marson2025__Rest", "orion__hct116"]
GATE, ALPHA, MIN_REL = 5.0, 0.05, 0.3
N_MEAN_ROWS, N_BOOT = 500, 1000
BINS = [0, 10, 100, 1000, np.inf]


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def rows(h, key, idx):
    """Rows idx (any order) of a [T, G] dataset as float32."""
    o = np.argsort(idx)
    out = h[key][np.asarray(idx)[o]].astype(np.float32)
    return out[np.argsort(o)]


class Context:
    """One summary: targets, measured genes, effect rows and reliabilities read on demand."""

    def __init__(self, path):
        self.name, self.path = path.stem, path
        with h5py.File(path) as h:
            self.targets = _s(h["pert/targets"][:])
            self.measured = h["measured"][:].astype(bool)
            self.gated = h["control/mean_cpm"][:] > GATE
            self.bm = "pert/half_delta_bm_A" in h
            rk = "pert/reliability_sb_batchmatched" if self.bm else "pert/reliability_sb"
            self.rel = pd.Series(h[rk][:].astype(float), index=self.targets)
        self.index = pd.Series(np.arange(len(self.targets)), index=self.targets)

    def halves(self, targets):
        k = "half_delta_bm" if self.bm else "half_delta"
        with h5py.File(self.path) as h:
            i = self.index[targets].to_numpy()
            return rows(h, f"pert/{k}_A", i), rows(h, f"pert/{k}_B", i)

    def effects(self, targets):
        a, b = self.halves(targets)
        return (a + b) / 2

    def significant(self, targets, cols):
        with h5py.File(self.path) as h:
            nlp = rows(h, "pert/nlog10p", self.index[targets].to_numpy())[:, cols]
        q = np.vstack([false_discovery_control(np.clip(10 ** -np.nan_to_num(r, nan=0.0), 0, 1)) for r in nlp])
        return (q < ALPHA).sum(1)


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def weighted_mean(stack, weights):
    """Per-gene weighted mean over sources, ignoring sources that do not measure the gene; 0 where none does."""
    w = np.where(np.isfinite(stack), weights[:, None], 0.0)
    s = np.nansum(np.nan_to_num(stack) * w, 0)
    tot = w.sum(0)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 0.0)


def design_rank(meta, held, sources):
    """Sources ordered by shared line, then pluripotency, lab, condition and chemistry with the held-out context."""
    m = meta.set_index("context")
    h = m.loc[held]
    key = lambda s: (m.loc[s].cell_line == h.cell_line, (m.loc[s].cell_line in PLURIPOTENT) == (h.cell_line in PLURIPOTENT),
                     m.loc[s].lab == h.lab, m.loc[s].cond == h.cond, m.loc[s].chem == h.chem)
    return sorted(sources, key=key, reverse=True)


def predict(held, ctxs, meta, a, rng):
    """Per-target scores of every predictor for one held-out context."""
    H = ctxs[held]
    family = meta.set_index("context").dataset
    sources = [c for c in ctxs if c != held and not (a.exclude_same_dataset and family[c] == family[held])]
    pool = sorted(set().union(*[set(ctxs[c].targets) for c in sources]))
    targets = [t for t in H.targets if t in set(pool)]
    if len(targets) > a.max_targets:
        targets = sorted(rng.choice(targets, a.max_targets, replace=False))
    cols = np.flatnonzero(H.measured & H.gated)
    hA, hB = H.halves(targets)
    hA, hB = hA[:, cols], hB[:, cols]
    n_sig = H.significant(targets, cols)
    gene_pos = pd.Series(np.arange(len(cols)), index=np.flatnonzero(H.measured & H.gated))
    with h5py.File(H.path) as h:
        vocab = pd.Series(np.arange(len(H.measured)), index=_s(h["genes"][:]))
    eff, mean_resp, rel, scale = {}, {}, {}, {}
    for s in sources:
        S = ctxs[s]
        have = [t for t in targets if t in S.index.index]
        sample = rng.choice(S.targets, min(N_MEAN_ROWS, len(S.targets)), replace=False)
        mr = np.nanmean(S.effects(sorted(sample))[:, cols], 0)
        mr[~S.measured[cols]] = np.nan
        mean_resp[s] = mr
        if have:
            e = S.effects(have)[:, cols]
            e[:, ~S.measured[cols]] = np.nan
            eff[s] = pd.Series(list(e), index=have)
            rel[s] = S.rel[have]
    cons = weighted_mean(np.vstack(list(mean_resp.values())), np.ones(len(mean_resp)))
    total = {t: np.zeros(len(cols)) for t in targets}
    count = {t: np.zeros(len(cols)) for t in targets}
    for s in eff:
        for t, v in eff[s].items():
            total[t] += np.nan_to_num(v)
            count[t] += np.isfinite(v)
    for s in eff:
        num = den = 0.0
        for t, v in eff[s].items():
            n = count[t] - np.isfinite(v)
            ok = np.isfinite(v) & (n > 0)
            y = (total[t][ok] - v[ok]) / n[ok]
            num, den = num + (v[ok] * y).sum(), den + (y * y).sum()
        scale[s] = num / den if den > 0 and num > 0 else 1.0
    order = design_rank(meta, held, list(eff))
    out = []
    for i, t in enumerate(targets):
        own = np.ones(len(cols), bool)
        if t in vocab.index and vocab[t] in gene_pos.index:
            own[gene_pos[vocab[t]]] = False
        src = [s for s in eff if t in eff[s].index]
        stack = np.vstack([eff[s][t] for s in src])
        r = np.array([rel[s][t] for s in src])
        w = np.clip(r, 0.01, 0.99) / (1 - np.clip(r, 0.01, 0.99))
        centered = np.vstack([eff[s][t] - mean_resp[s] for s in src]) / np.array([scale[s] for s in src])[:, None]
        preds = dict(mean_response=cons, nearest_context=eff[next(s for s in order if t in eff[s].index)][t],
                     average=weighted_mean(stack, np.ones(len(src))), precision_weighted=weighted_mean(stack, w),
                     calibrated=weighted_mean(centered, w) + cons, own_half_A=hA[i])
        ceiling = pearson(hA[i, own], hB[i, own])
        for p, v in preds.items():
            v = np.nan_to_num(np.asarray(v, float))
            rr = pearson(v[own], hB[i, own])
            out.append(dict(held_out=held, target=t, predictor=p, r=rr, ceiling=ceiling,
                            r_norm=rr / ceiling if ceiling > 0 else np.nan, n_sources=len(src),
                            n_sig=int(n_sig[i]), rel_held_out=H.rel.get(t, np.nan)))
    return out


def boot_median(v, rng):
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.median(v) if len(v) else np.nan), np.nan, np.nan
    m = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], 1)
    return np.median(v), *np.percentile(m, [2.5, 97.5])


def summarize(df, rng):
    """Median normalized r per held-out context, predictor and strength bin, over targets that replicate."""
    df = df[df.ceiling >= MIN_REL].assign(strength=lambda d: pd.cut(d.n_sig, BINS, right=False,
                                          labels=["<10 DE", "10-99 DE", "100-999 DE", ">=1000 DE"]).astype(str))
    res = []
    for st, d in [("all", df)] + list(df.groupby("strength")):
        for (ho, p), g in d.groupby(["held_out", "predictor"]):
            m, lo, hi = boot_median(g.r_norm.to_numpy(float), rng)
            res.append(dict(held_out=ho, predictor=p, strength=st, n_targets=len(g), r_norm=m, lo=lo, hi=hi,
                            r=g.r.median(), n_sources_median=g.n_sources.median()))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--held-out", nargs="*", default=HELD_OUT)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--max-targets", type=int, default=1000)
    ap.add_argument("--exclude-same-dataset", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default=str(ROOT / "results/mergeability/loco"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    meta = pd.read_csv(ROOT / "data_processing/workflow/contexts.csv")
    meta = meta.assign(lab=meta.dataset.map(lambda d: LAB.get(d, d)), cond=meta.condition.map(condition_kind),
                       chem=meta.dataset.map(CHEMISTRY))
    ctxs = {f.stem: Context(f) for f in sorted(Path(a.summaries).glob("*.h5")) if f.stem in set(meta.context)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for held in a.held_out:
        res = predict(held, ctxs, meta, a, rng)
        pd.DataFrame(res).to_csv(out / f"per_target_{held}.csv", index=False)
        allrows += res
        print(f"{held}: {len({r['target'] for r in res})} targets", flush=True)
    df = pd.DataFrame(allrows)
    s = summarize(df, rng)
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(s[s.strength == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
