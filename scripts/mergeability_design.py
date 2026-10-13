"""Pair agreement grouped by what differs between two contexts: line, lab, chemistry, condition."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
LAB = {"adamson": "Weissman", "replogle2020": "Weissman", "replogle_gwps": "Weissman", "replogle_nadig": "Weissman",
       "gse295214": "Weissman"}
CHEMISTRY = {"adamson": "10x 3'", "replogle2020": "10x 3'", "replogle_gwps": "10x 3'", "replogle_nadig": "10x 3'",
             "frangieh": "10x 3'", "gse337988": "10x 3'", "kolf": "10x 3'", "sunshine": "10x 3'", "feng": "10x 5'",
             "orion (X-Atlas/Orion)": "10x 5'", "vcc2025": "Flex", "cd4t_marson2025": "Flex", "viperturb": "Flex",
             "jiang2025": "Parse", "mcfaline2023": "sci-RNA-seq3"}
PLURIPOTENT = {"H1", "hESC", "KOLF2.1J", "hPSC lines, pooled donors"}
FACTORS = ["same_line", "same_lab", "same_chemistry", "same_condition"]
N_BOOT = 1000
CONTRASTS = [
    ("upper bound: same lab, line, condition, other screen",
     [("replogle_gwps__k562", "replogle_nadig__k562"), ("replogle2020__k562", "replogle_gwps__k562"),
      ("replogle2020__k562", "replogle_nadig__k562")]),
    ("K562, same lab, older chemistry", [("adamson__k562", "replogle_gwps__k562"), ("adamson__k562", "replogle_nadig__k562"),
                                         ("adamson__k562", "replogle2020__k562")]),
    ("K562, other lab, Flex", [("replogle_gwps__k562", "viperturb__k562"), ("replogle_nadig__k562", "viperturb__k562"),
                               ("adamson__k562", "viperturb__k562")]),
    ("K562, other lab, Parse, stimulated",
     [(f"jiang2025_{s}__k562", o) for s in ("ifnb", "ifng", "ins", "tgfb", "tnfa")
      for o in ("replogle_gwps__k562", "replogle_nadig__k562", "viperturb__k562")]),
    ("same lab, other line: Nadig", [("replogle_nadig__k562", "replogle_nadig__rpe1"), ("replogle_nadig__hepg2", "replogle_nadig__k562"),
                                     ("replogle_nadig__jurkat", "replogle_nadig__k562"), ("replogle_nadig__hepg2", "replogle_nadig__rpe1"),
                                     ("replogle_nadig__jurkat", "replogle_nadig__rpe1"), ("replogle_nadig__hepg2", "replogle_nadig__jurkat")]),
    ("same lab, other line: Orion", [("orion__hct116", "orion__hek293t")]),
    ("same lab, other line: McFaline", [("mcfaline2023__a172", "mcfaline2023__t98g"), ("mcfaline2023__a172", "mcfaline2023__u87mg"),
                                        ("mcfaline2023__t98g", "mcfaline2023__u87mg")]),
    ("same lab and line, other condition: CD4 T", [("cd4t_marson2025__Rest", "cd4t_marson2025__Stim8hr"),
                                                   ("cd4t_marson2025__Rest", "cd4t_marson2025__Stim48hr"),
                                                   ("cd4t_marson2025__Stim48hr", "cd4t_marson2025__Stim8hr")]),
    ("same lab and line, other condition: Frangieh", [("frangieh__control", "frangieh__ifng"), ("frangieh__coculture", "frangieh__control"),
                                                      ("frangieh__coculture", "frangieh__ifng")]),
    ("pluripotent, other line and lab", [("gse295214__hesc", "vcc2025__H1"), ("kolf__hiPSC", "vcc2025__H1"), ("gse295214__hesc", "kolf__hiPSC"),
                                         ("feng_fit__hpsc", "gse295214__hesc"), ("feng_fit__hpsc", "vcc2025__H1"), ("feng_fit__hpsc", "kolf__hiPSC"),
                                         ("feng_nonfit__hpsc", "vcc2025__H1"), ("feng_nonfit__hpsc", "kolf__hiPSC")]),
]


def condition_kind(c):
    return "unstimulated" if str(c).startswith("unstimulated") else str(c)


def tag(ps, meta):
    """Pair rows with the design factors and the design class of each pair."""
    m = meta.set_index("context")
    a, b = m.loc[ps.a], m.loc[ps.b]
    lab = lambda d: d.dataset.map(lambda x: LAB.get(x, x)).to_numpy()
    chem = lambda d: d.dataset.map(CHEMISTRY).to_numpy()
    out = ps.assign(line_a=a.cell_line.to_numpy(), line_b=b.cell_line.to_numpy(), chemistry_a=chem(a), chemistry_b=chem(b),
                    same_line=a.cell_line.to_numpy() == b.cell_line.to_numpy(), same_lab=lab(a) == lab(b),
                    same_chemistry=chem(a) == chem(b),
                    same_condition=a.condition.map(condition_kind).to_numpy() == b.condition.map(condition_kind).to_numpy(),
                    same_modality=a.perturbation_modality.str[:7].to_numpy() == b.perturbation_modality.str[:7].to_numpy(),
                    pluripotent=a.cell_line.isin(PLURIPOTENT).to_numpy() & b.cell_line.isin(PLURIPOTENT).to_numpy())
    out["design"] = np.select(
        [out.same_line & out.same_lab & out.same_condition, out.same_line & out.same_lab, out.same_line,
         out.same_lab & out.same_condition, out.same_lab, out.pluripotent],
        ["same lab, line and condition", "same lab and line, other condition", "same line, other lab",
         "same lab and condition, other line", "same lab, other line and condition", "pluripotent, other line and lab"],
        "other lab and line")
    return out


def weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def summarize(g, rng):
    """Median r_disatt over pairs weighted by reliable targets, with a bootstrap interval over pairs."""
    x, w = g.r_disatt.to_numpy(), g.n_reliable.to_numpy(float)
    boot = [weighted_median(x[i], w[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))] if len(x) > 1 else [x[0]]
    return pd.Series(dict(n_pairs=len(g), n_reliable=int(w.sum()), r_disatt=weighted_median(x, w),
                          lo=np.quantile(boot, .025), hi=np.quantile(boot, .975), q1=g.r_disatt.quantile(.25),
                          q3=g.r_disatt.quantile(.75), r_centered=weighted_median(g.r_centered.to_numpy(), w),
                          slope_lfc=g.slope_lfc.median()))


def regression(t, rng):
    """r_disatt on the four factors, weighted by reliable targets, with a bootstrap over pairs."""
    X = sm.add_constant(t[FACTORS].astype(float))
    fit = sm.WLS(t.r_disatt, X, weights=t.n_reliable).fit()
    boot = []
    for i in rng.integers(0, len(t), (N_BOOT, len(t))):
        s = t.iloc[i]
        boot.append(sm.WLS(s.r_disatt, sm.add_constant(s[FACTORS].astype(float), has_constant="add"), weights=s.n_reliable).fit().params)
    boot = pd.DataFrame(boot)
    n_with = {f: int(t[f].sum()) for f in FACTORS}
    return pd.DataFrame(dict(term=fit.params.index, coef=fit.params.to_numpy(), lo=boot.quantile(.025).to_numpy(),
                             hi=boot.quantile(.975).to_numpy(), n_pairs_with=[len(t)] + [n_with[f] for f in FACTORS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mergeability", default=str(ROOT / "results/mergeability"))
    ap.add_argument("--pooled", default=str(ROOT / "results/mergeability_pooled"))
    ap.add_argument("--min-reliable", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or Path(a.mergeability) / "design")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    meta = pd.read_csv(ROOT / "workflow/contexts.csv")
    ps = pd.read_csv(Path(a.mergeability) / "pair_summary.csv")
    ps = ps[ps.strength == "all"].drop(columns="strength")
    pooled = pd.read_csv(Path(a.pooled) / "pair_summary.csv")
    pooled = pooled[pooled.strength == "all"][["a", "b", "r_disatt", "n_reliable"]]
    ps = ps.merge(pooled.rename(columns={"r_disatt": "r_disatt_pooled", "n_reliable": "n_reliable_pooled"}), on=["a", "b"], how="left")
    t = tag(ps, meta)
    t.to_csv(out / "pairs_tagged.csv", index=False)
    use = t[(t.n_reliable >= a.min_reliable) & t.r_disatt.notna()]
    by_design = use.groupby("design").apply(summarize, rng=rng, include_groups=False).reset_index().sort_values("r_disatt", ascending=False)
    by_design.to_csv(out / "by_design.csv", index=False)
    key = {frozenset(p): i for i, p in enumerate(zip(t.a, t.b))}
    rows, summ = [], []
    for name, pairs in CONTRASTS:
        idx = [key[frozenset(p)] for p in pairs if frozenset(p) in key]
        g = t.iloc[idx]
        rows += [dict(contrast=name, **r) for r in g[["a", "b", "n_targets", "n_reliable", "r", "r_centered", "r_disatt", "r_disatt_lo",
                                                     "r_disatt_hi", "r_disatt_pooled", "slope_lfc", "slope_lfc_lo", "slope_lfc_hi",
                                                     "effects"]].to_dict("records")]
        g = g[(g.n_reliable >= a.min_reliable) & g.r_disatt.notna()]
        if len(g):
            summ.append(dict(contrast=name, **summarize(g, rng).to_dict()))
    pd.DataFrame(rows).to_csv(out / "contrasts_pairs.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "contrasts.csv", index=False)
    reg = regression(use, rng)
    reg.to_csv(out / "regression.csv", index=False)
    cross = use.groupby(FACTORS).size().rename("n_pairs").reset_index()
    cross.to_csv(out / "factor_combinations.csv", index=False)
    pd.set_option("display.width", 200)
    print(by_design.round(3).to_string(index=False))
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(reg.round(3).to_string(index=False))
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
