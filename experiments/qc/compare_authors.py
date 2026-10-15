"""Per target: our fold-change vector against the one the authors released, on genes above 5 CPM in our controls."""
import io
import sys
import zipfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
GATE, EPS, SIG = 5.0, 1e-9, 0.05


def ours(context):
    with h5py.File(ROOT / f"data/{context}.h5") as h:
        s = lambda k: [x.decode() if isinstance(x, bytes) else str(x) for x in h[k][:]]
        key = "pert/lfc_batchmatched" if "pert/lfc_batchmatched" in h else "pert/lfc"
        keep = h["measured"][:].astype(bool) & (h["control/mean_cpm"][:] > GATE)
        genes = np.array(s("genes"))[keep]
        return pd.DataFrame(h[key][:][:, keep], index=s("pert/targets"), columns=genes), key


def feng():
    d = pd.read_csv(RAW / "feng/GenomeWideScreen_LFC_byGene.tsv.gz", sep="\t",
                    usecols=["Target", "Expressed_Gene_Symbol", "lfc", "pval_adj"])
    lfc = d.pivot_table(index="Target", columns="Expressed_Gene_Symbol", values="lfc", aggfunc="first")
    sig = d.pivot_table(index="Target", columns="Expressed_Gene_Symbol", values="pval_adj", aggfunc="first") < SIG
    return {c: (lfc, sig) for c in ("feng_fit__hpsc", "feng_nonfit__hpsc")}


def jiang():
    out = {}
    with zipfile.ZipFile(RAW / "jiang2025/authors_de/DE_results_all_pathway.zip") as z:
        for name in z.namelist():
            if not name.endswith("_pathway_DE_results.txt") or "__MACOSX" in name:
                continue
            target, pw = Path(name).name.split("_")[:2]
            d = pd.read_csv(io.BytesIO(z.read(name)), sep=r"\s+", index_col=0)
            for col in d.columns[d.columns.str.startswith("log2FC_")]:
                line = col.split("_", 1)[1]
                ctx = f"jiang2025_{pw.lower().rstrip('1')}__{line.lower()}"
                out.setdefault(ctx, {})[target] = (d[col], d[f"p_cell_type{line}"] < SIG)
    return {c: (pd.DataFrame({t: v[0] for t, v in m.items()}).T, pd.DataFrame({t: v[1] for t, v in m.items()}).T)
            for c, m in out.items()}


def replogle():
    a = ad.read_h5ad(RAW / "replogle2022/K562_essential_raw_bulk_01.h5ad")
    x = pd.DataFrame(np.asarray(a.X, float), index=a.obs_names, columns=a.var["gene_name"].astype(str).to_numpy()
                     if "gene_name" in a.var else a.var_names)
    cpm = x.div(x.sum(1), axis=0) * 1e6
    ctrl = cpm[a.obs["core_control"].astype(str).eq("True").to_numpy()].mean()
    lfc = np.log2((cpm + EPS) / (ctrl + EPS))
    gene = a.obs["gene"].astype(str).to_numpy() if "gene" in a.obs else a.obs_names.str.split("_").str[1]
    keep = ~a.obs["core_control"].astype(str).eq("True").to_numpy()
    lfc = lfc[keep].set_axis(gene[keep], axis=0)
    lfc = lfc[~lfc.index.duplicated(keep=False)]
    return {"replogle_nadig__k562": (lfc, None)}


def dld1():
    lfc = pd.read_csv(RAW / "gse337988/authors_de/GSE337988_sublib2_de_matrices_lfc_matrix_Low1.csv.gz", index_col=0).T
    p1 = lfc[lfc.index.str.endswith("_P1")]
    return {"gse337988__dld1": (p1.set_axis(p1.index.str[:-3], axis=0), None)}


def cd4t():
    a = ad.read_h5ad(RAW / "cd4t_marson2025/GWCD4i.DE_stats.h5ad", backed="r")
    tcol, genes = "target_contrast_gene_name", a.var["gene_name"].astype(str).to_numpy()
    out = {}
    for cond in ("Rest", "Stim8hr", "Stim48hr"):
        rows = np.flatnonzero(a.obs["culture_condition"].astype(str).to_numpy() == cond)
        lfc = pd.DataFrame(a.layers["log_fc"][rows], index=a.obs[tcol].astype(str).to_numpy()[rows], columns=genes)
        sig = pd.DataFrame(a.layers["adj_p_value"][rows] < SIG, index=lfc.index, columns=genes)
        out[f"cd4t_marson2025__{cond}"] = (lfc[~lfc.index.duplicated()], sig[~sig.index.duplicated()])
    return out


def compare(context, theirs, sig):
    us, key = ours(context)
    theirs = theirs.loc[:, ~theirs.columns.duplicated(keep=False)]
    sig = None if sig is None else sig.loc[:, ~sig.columns.duplicated(keep=False)]
    targets = us.index.intersection(theirs.index)
    genes = us.columns.intersection(theirs.columns)
    rows = []
    for t in targets:
        g = genes.drop(t, errors="ignore")
        x, y = us.loc[t, g].to_numpy(float), theirs.loc[t, g].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            continue
        s = sig.loc[t, g].to_numpy(bool) & ok if sig is not None and t in sig.index else None
        rows.append(dict(context=context, ours=key, target=t, n_genes=int(ok.sum()), r=np.corrcoef(x[ok], y[ok])[0, 1],
                         n_sig=int(s.sum()) if s is not None else np.nan,
                         r_sig=np.corrcoef(x[s], y[s])[0, 1] if s is not None and s.sum() >= 10 else np.nan,
                         sd_ratio=np.std(x[ok]) / np.std(y[ok])))
    return rows


def main(sources):
    load = dict(feng=feng, jiang=jiang, replogle=replogle, dld1=dld1, cd4t=cd4t)
    rows = []
    for src in sources:
        for ctx, (theirs, sig) in load[src]().items():
            if (ROOT / f"data/{ctx}.h5").exists():
                rows += compare(ctx, theirs, sig)
                print(ctx, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/qc"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"compare_authors_{'_'.join(sources)}.csv", index=False)
    summ = df.groupby("context").agg(targets=("target", "size"), genes=("n_genes", "median"), r=("r", "median"),
                                     r_sig=("r_sig", "median"), targets_r_sig=("r_sig", "count"), sd_ratio=("sd_ratio", "median"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
