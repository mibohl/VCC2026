"""DLD1: the authors' Low1 against Low2 fold changes per promoter, and ours against each."""
import numpy as np
import pandas as pd
from compare_authors import RAW, ours

a, b = (pd.read_csv(RAW / f"gse337988/authors_de/GSE337988_sublib2_de_matrices_lfc_matrix_{k}.csv.gz", index_col=0).T for k in ("Low1", "Low2"))
us, _ = ours("gse337988__dld1")
p = a.index.intersection(b.index)
p = p[p.str.endswith("_P1")]
genes = a.columns.intersection(b.columns).intersection(us.columns)
rows = []
for q in p:
    t = q[:-3]
    x, y = a.loc[q, genes].to_numpy(float), b.loc[q, genes].to_numpy(float)
    r_ab = np.corrcoef(x, y)[0, 1]
    if t in us.index:
        z = us.loc[t, genes].to_numpy(float)
        ok = np.isfinite(z)
        rows.append(dict(target=t, r_low1_low2=r_ab, r_ours_low1=np.corrcoef(z[ok], x[ok])[0, 1], r_ours_low2=np.corrcoef(z[ok], y[ok])[0, 1]))
df = pd.DataFrame(rows)
print(len(df), "targets,", len(genes), "genes")
print(df.drop(columns="target").median().round(3).to_string())
