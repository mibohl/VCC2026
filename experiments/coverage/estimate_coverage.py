"""Preliminary gene and panel-target coverage per source, from raw metadata only (no counts processed).

Gene match: exact symbol, then Ensembl id via the Flex v1.0.1 probe set (the panel's own key), then unique
case-insensitive symbol. Target match: exact or unique case-insensitive symbol, after HLA_X -> HLA-X.
Usage: python estimate_coverage.py OUT_DIR
"""
import sys, re, glob, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'data_processing/src'), str(ROOT / 'data_processing/src/clean')]
import numpy as np, pandas as pd, h5py
import pyarrow.parquet as pq
from add_dispersion import cat, open_figshare
from readers import index_of

R = str(ROOT / 'data')
OUT = sys.argv[1]
vocab = pd.read_csv(f'{R}/vcc2026/controls/gene_names.csv').gene_name.astype(str).to_numpy()
V = set(vocab)
up = pd.Series(vocab).str.upper(); dup = set(up[up.duplicated(keep=False)])
ui = {u: g for u, g in zip(up, vocab) if u not in dup}
p = pd.read_csv(f'{R}/raw/probeset/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2020-A.csv', comment='#', dtype=str)
e2s = dict(zip(p.gene_id.str.replace('DEPRECATED_', '', regex=False), p.probe_id.str.split('|').str[1]))
panel = pd.read_csv(f'{R}/vcc2026/controls/pert_counts.csv').target_gene.astype(str).tolist()


def map_genes(sym, ens=None):
    hit = set()
    sym = np.asarray(sym).astype(str)
    ens = None if ens is None else np.asarray(ens).astype(str)
    for i, s in enumerate(sym):
        if s in V:
            hit.add(s); continue
        e = None if ens is None else ens[i].split('.')[0]
        if e in e2s:
            hit.add(e2s[e]); continue
        if s.split('.')[0] in e2s:
            hit.add(e2s[s.split('.')[0]]); continue
        if s.upper() in ui:
            hit.add(ui[s.upper()])
    return hit


def map_targets(labels):
    out = set()
    for t in pd.unique(pd.Series(labels).fillna('').astype(str)):
        t2 = t.replace('_', '-') if re.fullmatch(r'HLA_[A-Z0-9]+', t) else t
        if t2 in V:
            out.add(t2)
        elif t2.upper() in ui:
            out.add(ui[t2.upper()])
    return out


def var_genes(h):
    keys = set(h['var'].keys()); index = index_of(h, 'var')
    sym = next((cat(h, f'var/{k}') for k in ('gene_name', 'gene_symbol', 'feature_name') if k in keys), index)
    ens = next((cat(h, f'var/{k}') for k in ('gene_id', 'gene_ids', 'ensembl_id') if k in keys), None)
    if ens is None and str(index[0]).startswith('ENSG'):
        ens = index
    return sym, ens


rows = []
def add(name, genes, targets, note=''):
    rows.append(dict(source=name, genes=genes, targets=targets, note=note))
    print(f'{name}: genes {len(genes) if genes is not None else "n/a"}, '
          f'panel targets {len(targets & set(panel)) if targets is not None else "n/a"}', flush=True)


# H1 2025: gene axis from Training (the only split with gene_id; validate_h1 asserts identical axes), targets from all
h = h5py.File(f'{R}/vcc2025/adata_Training.h5ad'); sym, ens = var_genes(h); t = set()
for s in ('Training', 'Validation', 'Test'):
    t |= map_targets(cat(h5py.File(f'{R}/vcc2025/adata_{s}.h5ad'), 'obs/target_gene'))
add('vcc2025_H1', map_genes(sym, ens), t)

for name, f in [('replogle_K562_essential', 'replogle2022/K562_essential_raw_singlecell_01.h5ad'),
                ('replogle_RPE1_essential', 'replogle2022/rpe1_raw_singlecell_01.h5ad'),
                ('nadig_Jurkat', 'nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad'),
                ('nadig_HepG2', 'nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad')]:
    h = h5py.File(f'{R}/raw/{f}'); sym, ens = var_genes(h)
    add(name, map_genes(sym, ens), map_targets(cat(h, 'obs/gene')))

# genome-wide K562: the authors' pseudobulk file has the same gene axis and one row per perturbation
import os
if os.path.exists(f'{R}/raw/replogle2022/K562_gwps_raw_bulk_01.h5ad'):
    h = h5py.File(f'{R}/raw/replogle2022/K562_gwps_raw_bulk_01.h5ad'); sym, ens = var_genes(h)
    lab = pd.Series(index_of(h, 'obs')).astype(str).str.split('_').str[1]
    add('replogle_K562_genomewide', map_genes(sym, ens), map_targets(lab), 'targets from the pseudobulk file (gene part of the row id)')
else:
    add('replogle_K562_genomewide', None, None, 'pending: download not finished')
add('kolf_iPSC', None, None, 'pending: 189 GB file not downloaded yet; metadata stream fails while the tunnel is saturated')

for name, f in [('adamson_K562', 'scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad'),
                ('frangieh_melanoma', 'scperturb/FrangiehIzar2021_RNA.h5ad')]:
    h = h5py.File(f'{R}/raw/{f}'); sym, ens = var_genes(h)
    lab = pd.Series(cat(h, 'obs/perturbation')).astype(str)
    lab = lab.str.split('_').str[0] if name.startswith('adamson') else lab
    add(name, map_genes(sym, ens), map_targets(lab))

f = sorted(glob.glob(f'{R}/raw/cd4t_marson2025/*.h5'))[0]; m = h5py.File(f)['matrix']
ft = m['features/feature_type'][:].astype(str); gex = ft == 'Gene Expression'
t = set(); obs = glob.glob(f'{R}/raw/cd4t_marson2025/obs/*.obs.parquet')
for o in obs:
    d = pd.read_parquet(o, columns=['guide_group', 'perturbed_gene_name'])
    t |= map_targets(d.perturbed_gene_name[d.guide_group == 'targeting single sgRNA'])
add('cd4t_marson2025', map_genes(m['features/name'][:].astype(str)[gex], m['features/id'][:].astype(str)[gex]), t,
    f'targets from {len(obs)} of 12 metadata files')

m = h5py.File(sorted(glob.glob(f'{R}/raw/gse295214/*.h5'))[0])['matrix']
ft = m['features/feature_type'][:].astype(str); nm = m['features/name'][:].astype(str); gex = ft == 'Gene Expression'
add('gse295214_hESC', map_genes(nm[gex], m['features/id'][:].astype(str)[gex]),
    map_targets(pd.Series(nm[~gex]).str.split('_').str[0]), 'targets = guide library')

rd = pd.read_csv(f'{R}/raw/gse337988/low_rowdata.csv.gz', dtype=str)
cd = pd.read_csv(f'{R}/raw/gse337988/low_coldata.csv.gz', usecols=['assignment_v2'], dtype=str)
add('gse337988_DLD1', map_genes(rd.Symbol, rd.feature), map_targets(cd.assignment_v2))

for name, cnt, meta in [('feng_fitness', 'GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz', 'GW_Fitness_meta.tsv.gz'),
                        ('feng_nonfitness', 'GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz', 'GW_NonFitness_meta.tsv.gz')]:
    out = subprocess.run(f"pigz -dc -p 4 {R}/raw/feng/{cnt} | cut -d, -f1 | tail -n +2", shell=True,
                         capture_output=True, text=True).stdout.split()
    feats = [x.split(':') for x in out if x.endswith(':Gene-Expression')]
    mt = pd.read_csv(f'{R}/raw/feng/{meta}', sep='\t', usecols=['Guide_Call']).Guide_Call.astype(str).str.rsplit('_', n=1).str[0]
    add(name, map_genes([f[1] for f in feats], [f[0] for f in feats]), map_targets(mt))

man = pd.read_csv(f'{R}/raw/viperturb/genome_wide_manifest.txt', sep=None, engine='python')
tcol = next(c for c in man.columns if 'gene' in c.lower())
add('viperturb_K562', None, map_targets(man[tcol]), f'genes need the RDS extraction; targets from manifest column {tcol}')

om = pd.read_parquet(f'{R}/orion_probe/gene_metadata.parquet'); og = map_genes(om.gene_name, om.ensembl_id)
shards = sorted(glob.glob(f'{R}/raw/orion/*.parquet'))
for line in ('HCT116', 'HEK293T'):
    mine = [x for x in shards if f'/{line}_' in x]; tl = set()
    for s in mine:
        tl |= map_targets(pq.read_table(s, columns=['gene_target']).column(0).to_pylist())
    add(f'orion_{line}', og, tl, f'targets from {len(mine)} of {109 if line == "HCT116" else 223} shards only')

# outputs
summ, long_g, long_t = [], [], []
for r in rows:
    miss_t = None if r['targets'] is None else [x for x in panel if x not in r['targets']]
    miss_g = None if r['genes'] is None else sorted(V - r['genes'])
    summ.append(dict(source=r['source'],
                     genes_measured=None if miss_g is None else len(r['genes']),
                     genes_missing=None if miss_g is None else len(miss_g),
                     panel_targets_measured=None if miss_t is None else 300 - len(miss_t),
                     panel_targets_missing=None if miss_t is None else len(miss_t),
                     note=r['note'], missing_panel_targets='' if miss_t is None else ';'.join(miss_t)))
    long_g += [dict(source=r['source'], missing_gene=x) for x in (miss_g or [])]
    long_t += [dict(panel_target=x, source=r['source'], measured=None if r['targets'] is None else x in r['targets'])
               for x in panel]
pd.DataFrame(summ).to_csv(f'{OUT}/coverage_summary.csv', index=False)
pd.DataFrame(long_g).to_csv(f'{OUT}/genes_missing_by_source.csv', index=False)
T = pd.DataFrame(long_t).pivot(index='panel_target', columns='source', values='measured')
T.to_csv(f'{OUT}/panel_targets_by_source.csv')
cnt = T.fillna(False).astype(bool).sum(1)
print('\npanel targets measured in N sources:', cnt.value_counts().sort_index().to_dict())
print('panel targets measured nowhere:', list(cnt[cnt == 0].index))
G = pd.DataFrame({r['source']: pd.Series(vocab).isin(r['genes']).to_numpy() for r in rows if r['genes'] is not None}, index=vocab)
G.to_csv(f'{OUT}/panel_genes_by_source.csv')
gc = G.sum(1)
print('panel genes measured in N sources:', gc.value_counts().sort_index().to_dict())
print('panel genes measured nowhere:', int((gc == 0).sum()))
