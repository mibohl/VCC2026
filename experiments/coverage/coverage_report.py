"""One-page HTML report of a panel's coverage, built only from the CSVs that coverage_tables.py wrote,
plus, when its CSVs exist, the mergeability of the contexts (experiments/mergeability/mergeability.py, effect_space_check.py).

    pixi run python experiments/coverage/coverage_report.py results/coverage/<panel_id> [--mergeability results/mergeability]
                                               [--pooled results/mergeability_pooled]      -> <dir>/report.html

The page is self-contained (data embedded as JSON, charts drawn in the browser) and follows the claude.ai artifact
page contract, so it can be opened locally or published as it is.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Lab and chemistry families for grouping pairs (several datasets share a lab: Adamson, Replogle 2020, 2022 and the
# hESC screen are all Weissman lab).
LAB = {"adamson": "Weissman", "replogle2020": "Weissman", "replogle_gwps": "Weissman", "replogle_nadig": "Weissman",
       "gse295214": "Weissman"}
CHEM = {"adamson": "3'", "replogle2020": "3'", "replogle_gwps": "3'", "replogle_nadig": "3'", "frangieh": "3'",
        "gse337988": "3'", "kolf": "3'", "sunshine": "3'", "feng": "5'", "orion (X-Atlas/Orion)": "5'",
        "vcc2025": "Flex", "cd4t_marson2025": "Flex", "viperturb": "Flex", "jiang2025": "Parse",
        "mcfaline2023": "sci-RNA-seq3", "gse295214": "unknown"}
# What one batch is in each dataset, in words (the readers record the column they read as batch_key).
ORIGIN = {
    "vcc2025": "10x Flex run (sample-multiplexed batch)", "adamson": "none: the deposit is a single 10x lane",
    "frangieh": "none used: no lane or batch column is read from the scPerturb file",
    "feng": "pool x sampling day x donor iPSC line (HipSci donors pooled in one screen)",
    "gse337988": "10x channel", "gse295214": "10x lane (one GEO sample file each)", "viperturb": "10x lane",
    "orion (X-Atlas/Orion)": "GEM sample (one parquet shard each)", "replogle_gwps": "10x GEM group (droplet lane)",
    "replogle_nadig": "10x GEM group (droplet lane)", "kolf": "sub-library (obs/batch: ALPHA, BETA, GAMMA; every target lies in one)",
    "cd4t_marson2025": "blood donor (4 donors; each condition is its own context)",
    "jiang2025": "biological replicate (Rep1, Rep2)", "replogle2020": "10x GEM group (barcode suffix)",
    "sunshine": "10x GEM group", "mcfaline2023": "replicate"}
PROGRESS = [
    dict(step="M1", what="Sound pair table: bootstrap intervals, depth-free fold-change slope, pooled vs batch-matched, scorer-space check", status="done (2026-10-13)"),
    dict(step="M2", what="Group pairs by what differs (line, lab, chemistry, condition); weighted regression of agreement on the four factors", status="done (2026-10-13)"),
    dict(step="M3", what="Which harmonization closes the gap: mean response removed, batch matching; scale calibration is tested in M4 because correlation ignores scale", status="done for centering and batch matching"),
    dict(step="KOLF", what="Every KOLF target lies in one of three sub-libraries (ALPHA, BETA, GAMMA) while controls are 55% ALPHA. With obs/batch as batch key its reliability falls from 0.38 to 0.018, and its agreement with H1 rises from 0.12 to 0.66, with hESC from 0.29 to 0.70", status="fixed (2026-10-13)"),
    dict(step="M4", what="Leave one context out for 7 contexts: predict a held-out context's effects from other datasets only", status="done (2026-10-13), first version"),
    dict(step="CD4 T", what="Random cell halves agree at median r 0.45 for random targets, a split by donor (D1+D2 vs D3+D4) at 0.04 (80% vs 3.4% of targets above 0.3): CD4 T's reliability is founder noise shared by both halves, not knockdown effect", status="confirmed; fix needed (donor-split reliability, donor-averaged effects)"),
    dict(step="M5", what="Write-up and pooling decision", status="after the CD4 T fix"),
]
PLURI = {"H1", "hESC", "KOLF2.1J", "hPSC lines, pooled donors"}
# Named pairs shown with their intervals, in reading order: (group, a, b).
NAMED = [
    ("upper bound: same lab, line and condition", "replogle_gwps__k562", "replogle_nadig__k562"),
    ("upper bound: same lab, line and condition", "replogle2020__k562", "replogle_gwps__k562"),
    ("upper bound: same lab, line and condition", "replogle2020__k562", "replogle_nadig__k562"),
    ("K562, same lab, older chemistry", "adamson__k562", "replogle_gwps__k562"),
    ("K562, same lab, older chemistry", "adamson__k562", "replogle_nadig__k562"),
    ("K562, other lab (Flex)", "replogle_gwps__k562", "viperturb__k562"),
    ("K562, other lab (Flex)", "replogle_nadig__k562", "viperturb__k562"),
    ("K562, other lab (Flex)", "adamson__k562", "viperturb__k562"),
    ("K562, other lab, stimulated (Parse)", "jiang2025_ifng__k562", "replogle_gwps__k562"),
    ("K562, other lab, stimulated (Parse)", "jiang2025_tgfb__k562", "replogle_gwps__k562"),
    ("same lab, other line", "replogle_nadig__k562", "replogle_nadig__rpe1"),
    ("same lab, other line", "replogle_nadig__hepg2", "replogle_nadig__k562"),
    ("same lab, other line", "replogle_nadig__jurkat", "replogle_nadig__k562"),
    ("same lab, other line", "orion__hct116", "orion__hek293t"),
    ("same lab and line, other condition", "cd4t_marson2025__Rest", "cd4t_marson2025__Stim8hr"),
    ("same lab and line, other condition", "cd4t_marson2025__Rest", "cd4t_marson2025__Stim48hr"),
    ("same lab and line, other condition", "frangieh__control", "frangieh__ifng"),
    ("pluripotent, other line", "gse295214__hesc", "vcc2025__H1"),
    ("pluripotent, other line", "kolf__hiPSC", "vcc2025__H1"),
    ("pluripotent, other line", "gse295214__hesc", "kolf__hiPSC"),
    ("pluripotent, other line", "feng_fit__hpsc", "gse295214__hesc"),
]


def pair_group(ra, rb):
    """Design group of a pair of contexts from their contexts.csv rows."""
    same_line = ra.cell_line == rb.cell_line
    same_lab = LAB.get(ra.dataset, ra.dataset) == LAB.get(rb.dataset, rb.dataset)
    same_cond = ra.condition == rb.condition
    if same_line and same_lab:
        return "same lab and line" + ("" if same_cond else ", other condition")
    if same_line:
        return "same line, other lab"
    if ra.cell_line in PLURI and rb.cell_line in PLURI:
        return "pluripotent, other line"
    return "same lab, other line" if same_lab else "other lab and line"


def merge_data(mdir, pooled, inv):
    """Everything the mergeability section shows; None when the M1 rerun has not written its CSVs yet."""
    f = Path(mdir) / "pair_summary.csv"
    if not f.exists():
        return None
    ps = pd.read_csv(f)
    if "r_disatt_lo" not in ps.columns:
        return None
    meta = pd.read_csv(ROOT / "data_processing/workflow/contexts.csv").set_index("context")
    allb = ps[ps.strength == "all"].copy()
    allb["group"] = [pair_group(meta.loc[a], meta.loc[b]) for a, b in zip(allb.a, allb.b)]
    allb["chem"] = [" vs ".join(sorted({CHEM.get(meta.loc[c].dataset, "?") for c in (a, b)})) for a, b in zip(allb.a, allb.b)]
    allb["depth_ratio"] = allb[["depth_a", "depth_b"]].max(axis=1) / allb[["depth_a", "depth_b"]].min(axis=1)
    cols = ["a", "b", "group", "chem", "effects", "n_targets", "n_reliable", "r", "r_centered", "r_disatt", "r_disatt_lo",
            "r_disatt_hi", "slope", "slope_lfc", "slope_lfc_lo", "slope_lfc_hi", "depth_a", "depth_b", "depth_ratio"]
    key = {(a, b): i for i, (a, b) in enumerate(zip(allb.a, allb.b))}
    named = []
    for grp, a, b in NAMED:
        i = key.get((a, b), key.get((b, a)))
        if i is not None:
            named.append({**allb.iloc[i][cols].to_dict(), "label": f"{a} vs {b}", "group": grp})
    bins = ps[ps.strength != "all"]
    named_bins = []
    for _, a, b in NAMED[:1] + NAMED[3:4] + NAMED[5:6] + NAMED[10:11] + NAMED[14:15] + NAMED[17:18]:
        g = bins[((bins.a == a) & (bins.b == b)) | ((bins.a == b) & (bins.b == a))]
        for _, r in g.iterrows():
            named_bins.append(dict(pair=f"{r.a} vs {r.b}", strength=r.strength, n_targets=int(r.n_targets),
                                   n_reliable=int(r.n_reliable), r=r.r, r_disatt=r.r_disatt, lo=r.r_disatt_lo,
                                   hi=r.r_disatt_hi))
    rel = allb[allb.n_reliable >= 10]
    groups = (rel.groupby("group").r_disatt.agg(n_pairs="size", median="median", q1=lambda x: x.quantile(.25),
                                                q3=lambda x: x.quantile(.75)).reset_index().sort_values("median", ascending=False))
    depth = rel[["a", "b", "depth_a", "depth_b", "slope", "slope_lfc"]].dropna()
    depth = depth[(depth.slope > 0) & (depth.slope_lfc > 0)]
    depth = depth.assign(x=np.log2(depth.depth_b / depth.depth_a), y_half=np.log2(depth.slope),
                         y_lfc=np.log2(depth.slope_lfc))
    out = dict(pairs=allb[cols + []].sort_values("r_disatt", ascending=False).to_dict("records"), named=named,
               bins=named_bins, groups=groups.to_dict("records"), depth=depth.to_dict("records"))
    batched = inv[inv.mean_reliability_batchmatched.notna()]
    out["n8_rel"] = batched[["context", "mean_reliability", "mean_reliability_batchmatched"]].to_dict("records")
    pf = Path(pooled) / "pair_summary.csv" if pooled else None
    if pf is not None and pf.exists():
        pp = pd.read_csv(pf)
        pp = pp[pp.strength == "all"].set_index(["a", "b"])
        m = allb[allb.effects.str.contains("batch-matched")].set_index(["a", "b"])
        j = m.join(pp[["r", "r_disatt", "rel_a", "rel_b", "n_reliable"]], rsuffix="_pooled", how="inner").reset_index()
        j = j[(j.n_reliable >= 10) & (j.n_reliable_pooled >= 10)]
        out["n8_pairs"] = j[["a", "b", "effects", "rel_a_pooled", "rel_a", "rel_b_pooled", "rel_b", "r_disatt_pooled",
                             "r_disatt"]].to_dict("records")
    sf = Path(mdir) / "space_check/summary.csv"
    if sf.exists():
        out["space"] = pd.read_csv(sf).to_dict("records")
    dd = Path(mdir) / "design"
    if (dd / "contrasts.csv").exists():
        out["design"] = {k: _records(pd.read_csv(dd / f"{k}.csv")) for k in ("contrasts", "by_design", "regression")}
    lf = Path(mdir) / "loco/summary.csv"
    if lf.exists():
        lo = pd.read_csv(lf)
        out["loco"] = _records(lo[lo.strength == "all"])
        out["loco_bins"] = _records(lo[lo.strength != "all"])
    out["progress"] = PROGRESS
    return out


def _records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel_dir")
    ap.add_argument("--mergeability", default=str(ROOT / "results/mergeability"))
    ap.add_argument("--pooled", default=str(ROOT / "results/mergeability_pooled"))
    a = ap.parse_args()
    d = Path(a.panel_dir)
    inv = pd.read_csv(d / "inventory.csv")
    tc = pd.read_csv(d / "targets_coverage.csv")
    gc = pd.read_csv(d / "genes_coverage.csv")
    mat = pd.read_csv(d / "targets_by_context_cells.csv").set_index("panel_target")
    pid = d.name
    unstim = inv.condition.str.startswith("unstimulated")
    gate_cols = [c for c in gc.columns if c.startswith("gated_in_") and c != "gated_in_any"]
    order = tc.panel_target.tolist()                                   # least covered first
    ctx = inv.sort_values(["dataset", "cell_line", "context"])
    data = dict(
        panel=pid, n_targets=len(tc), n_genes=len(gc), n_contexts=len(inv), n_datasets=int(inv.dataset.nunique()),
        n_lines=int(inv.cell_line.nunique()), n_unstim=int(unstim.sum()),
        n_crispri=int(inv.perturbation_modality.str.startswith("CRISPRi").sum()),
        gates=[c.replace("gated_in_", "") for c in gate_cols],
        targets=tc[["panel_target", "rank_least_covered", "n_contexts", "n_datasets", "n_cell_lines",
                    "n_unstimulated_contexts", "total_cells", "median_cells_per_context", "datasets",
                    "cell_lines"]].fillna("").to_dict("records"),
        contexts=ctx[["context", "dataset", "cell_line", "condition", "perturbation_modality", "n_targets",
                      "n_panel_targets", "n_genes_measured", "n_missing_genes_above_gate_ABC",
                      "median_cells_per_target", "mean_reliability_batchmatched"]].fillna("").to_dict("records"),
        matrix=dict(rows=order, cols=ctx.context.tolist(),
                    values=mat.loc[order, ctx.context].to_numpy().astype(int).tolist()),
        gene_hist_all=gc.n_contexts_measured.value_counts().sort_index().to_dict(),
        gene_hist_gated=gc[gc.gated_in_any].n_contexts_measured.value_counts().sort_index().to_dict(),
        n_gated=int(gc.gated_in_any.sum()),
        genes_few=gc[gc.gated_in_any].nsmallest(30, "n_contexts_measured", keep="first")[
            ["gene", "n_contexts_measured", "n_datasets_measured", *gate_cols, "contexts_missing"]].to_dict("records"),
    )
    data["gene_hist_all"] = {int(k): int(v) for k, v in data["gene_hist_all"].items()}
    data["gene_hist_gated"] = {int(k): int(v) for k, v in data["gene_hist_gated"].items()}
    data["merge"] = merge_data(a.mergeability, a.pooled, inv)
    new = inv[inv.added_in_audit.astype(str) == "True"] if "added_in_audit" in inv else inv.iloc[:0]
    data["new_contexts"], data["new_datasets"] = new.context.tolist(), sorted(new.dataset.unique().tolist())
    # panel genes measured per experiment, and a per-gene lookup (missing contexts as indices into ctx_names)
    gcols = [c for c in inv.columns if c.startswith("n_missing_genes_above_gate_") and c != "n_missing_genes_above_gate_ABC"]
    gx = inv.assign(chem=inv.dataset.map(CHEM).fillna("?")).sort_values("n_genes_measured")
    data["gene_ctx"] = gx[["context", "dataset", "chem", "cell_line", "n_native_genes", "n_genes_measured",
                           "pct_panel_genes_measured", "n_panel_genes_missing", "n_missing_genes_above_gate_ABC",
                           *gcols, "n_missing_genes_without_included_probe", "n_genes_matched_ensembl",
                           "n_genes_matched_hgnc"]].to_dict("records")
    cost = d / "missing_gene_cost_summary.csv"
    if cost.exists():
        cs = pd.read_csv(cost)[["context", "set_share_median", "magnitude_share_median"]]
        gm = pd.DataFrame(data["gene_ctx"]).merge(cs, on="context", how="left")
        data["gene_ctx"] = gm.astype(object).where(gm.notna(), None).to_dict("records")
    data["gate_letters"] = [c.split("_")[-1] for c in gcols]
    names = inv.context.tolist()
    pos = {c: i for i, c in enumerate(names)}
    data["ctx_names"] = names
    bc = pd.read_csv(d / "batch_coverage.csv")
    agg = bc.groupby("context").agg(n_batches=("batch", "size"), ctrl_min=("n_controls", "min"),
                                    ctrl_median=("n_controls", "median"), ctrl_max=("n_controls", "max"),
                                    pert_median=("n_perturbed_cells", "median"), targets_median=("n_targets", "median"),
                                    examples=("batch", lambda x: "; ".join(map(str, x[:3]))))
    bt = inv[["context", "dataset", "batch_key", "median_batch_shift_rms"]].join(agg, on="context")
    bt["origin"] = bt.dataset.map(ORIGIN).fillna("")
    data["batch_ctx"] = bt.sort_values(["dataset", "context"]).to_dict("records")
    data["batch_rows"] = {c: g[["batch", "n_controls", "n_perturbed_cells", "n_targets"]].values.tolist()
                          for c, g in bc.groupby("context")}
    data["gene_lookup"] = {r["gene"]: [";".join(str(pos[c]) for c in str(r["contexts_missing"]).split(";") if c in pos),
                                    "".join(g for g in data["gates"] if r[f"gated_in_{g}"] in (True, "True"))]
                           for r in gc.to_dict("records")}
    html = TEMPLATE.replace("__DATA__", json.dumps(_clean(data), default=str)).replace("__PANEL__", pid)
    (d / "report.html").write_text(html)
    print(f"wrote {d / 'report.html'}")


def _clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN and infinities to None (JSON has no NaN)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


TEMPLATE = r"""<title>Panel Coverage __PANEL__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;650&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: summary band, then three stacked sections (contexts, targets, genes), each a heading, a chart and a table. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --fg: #1a1d24; --muted: #5d6573; --rule: #d9dde4;
  --accent: #3b5bdb; --accent-soft: #dbe4ff; --warm: #c2410c; --heat0: #eef0f4; --new: #0f766e; --new-soft: #e3f6f2;
  --font-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --font-body: "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12151b; --surface: #1a1e26; --fg: #e6e9ef; --muted: #9aa3b2; --rule: #2c323d;
  --accent: #7c9cff; --accent-soft: #26304d; --warm: #fb923c; --heat0: #222833; --new: #2dd4bf; --new-soft: #133430; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #12151b; --surface: #1a1e26; --fg: #e6e9ef; --muted: #9aa3b2; --rule: #2c323d;
  --accent: #7c9cff; --accent-soft: #26304d; --warm: #fb923c; --heat0: #222833; --new: #2dd4bf; --new-soft: #133430; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15.5px; line-height: 1.55; }
.wrap { max-width: 76rem; margin: 0 auto; padding-inline: 16px; padding-block: 2rem 4rem; }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; line-height: 1.2; margin: 0; }
h1 { font-size: clamp(1.7rem, 3.5vw, 2.3rem); font-weight: 650; }
h2 { font-size: 1.35rem; font-weight: 650; margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.02rem; font-weight: 650; margin-top: 1.4rem; }
p { max-width: 70ch; margin: .5rem 0; }
.eyebrow { font-family: var(--font-mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.mono, code { font-family: var(--font-mono); font-size: .85em; }
.band { display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: .8rem; margin: 1.4rem 0 .4rem; }
.band > div { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: .8rem 1rem; display: grid; gap: .1rem; min-width: 0; }
.band .v { font-family: var(--font-display); font-size: 1.7rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.band .k { font-size: .82rem; color: var(--muted); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: 1.2rem; margin-top: 1rem; }
.grid2 > * { min-width: 0; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem; }
.cap { font-size: .85rem; color: var(--muted); margin-top: .4rem; }
.tablebox { overflow-x: auto; margin-top: .8rem; max-height: 34rem; overflow-y: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
th, td { text-align: left; padding: .38rem .6rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { position: sticky; top: 0; background: var(--surface); font-family: var(--font-mono); font-weight: 500; font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); z-index: 1; }
td.n { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--font-mono); font-size: .82rem; }
td.wrap { min-width: 16rem; }
.pill { display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 999px; background: var(--accent-soft); color: var(--fg); white-space: nowrap; }
.pill.new { background: var(--new); color: var(--surface); font-weight: 600; }
tr.new td { background: var(--new-soft); }
tr.new td:first-child { box-shadow: inset 3px 0 0 var(--new); }
.pill.stim { background: transparent; border: 1px solid var(--warm); color: var(--warm); }
.chart svg { display: block; width: 100%; height: auto; }
.heatbox { overflow-x: auto; }
canvas { display: block; max-width: none; }
.controls { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; align-items: center; margin-top: .6rem; font-size: .88rem; }
.controls label { display: flex; gap: .4rem; align-items: center; }
select { font: inherit; padding: .3rem .5rem; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--fg); max-width: 100%; }
input[type=search] { font: inherit; padding: .3rem .5rem; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--fg); min-width: 12rem; }
input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
#tip { position: fixed; pointer-events: none; background: var(--fg); color: var(--bg); font-size: .78rem; padding: .3rem .5rem; border-radius: 4px; font-family: var(--font-mono); }
</style>

<div class="wrap">
  <div class="eyebrow">VCC 2026 · panel <span id="pid"></span> · built from results/coverage and results/mergeability CSVs</div>
  <h1>What our contexts cover of the VCC 2026 panel</h1>
  <p>For every target and gene the challenge asks us to predict: in how many of our summarized contexts it was perturbed or measured. A target counts as covered in a context when the context has cells for it, of any number.</p>
  <div class="band" id="band"></div>
  <p id="newNote"></p>
  <p class="cap">Sections: <a href="#contexts">contexts</a> · <a href="#targets">perturbations</a> · <a href="#genes">genes</a> · <a href="#genesPerContext">genes per experiment</a> · <a href="#batches">batches</a> · <a href="#merge">can the contexts be pooled?</a></p>

  <h2 id="contexts">Contexts</h2>
  <p>Each context is one cell line in one condition from one dataset. Stimulated and co-culture contexts are marked; drug-vehicle contexts count as unstimulated.</p>
  <div class="tablebox"><table id="ctxTable"></table></div>

  <h2 id="targets">Perturbations of the panel</h2>
  <div class="grid2">
    <div class="panel"><h3>Contexts per target</h3><div class="chart" id="hCtx"></div><div class="cap">Bars: all contexts; outlined: unstimulated contexts only.</div></div>
    <div class="panel"><h3>Distinct cell lines per target</h3><div class="chart" id="hLine"></div><div class="cap">Several contexts of one line (for example K562 in 7 datasets) count once.</div></div>
  </div>
  <h3>Every target, least covered first</h3>
  <p>Ranked by distinct cell lines, then contexts, then total cells.</p>
  <div class="controls"><label for="q">Find a target <input id="q" type="search" placeholder="e.g. FZD2"></label><span class="cap" id="qn"></span></div>
  <div class="tablebox"><table id="tTable"></table></div>

  <h3>Cells per target and context</h3>
  <p>Rows: the 300 targets, least covered at the top. Columns: contexts grouped by dataset. Darker = more cells; empty = not perturbed. Hover for counts.</p>
  <div class="panel heatbox"><canvas id="heat"></canvas></div>

  <h2 id="genes">Genes of the panel</h2>
  <p>Every panel gene must be predicted in every context. A context that does not measure a gene cannot teach its response there. Genes below 5 mean CPM in the 2026 controls are never tested by the DE metrics, so the genes that matter most are those above that gate in at least one of <span id="gates"></span>.</p>
  <div class="grid2">
    <div class="panel"><h3>Contexts measuring each gene</h3><div class="chart" id="hGene"></div><div class="cap">Filled: genes above the 2026 expression gate (<span id="ngated"></span>); outlined: all panel genes.</div></div>
    <div class="panel"><h3>Gated genes measured in the fewest contexts</h3><div class="tablebox" style="max-height:22rem"><table id="gTable"></table></div></div>
  </div>

  <h3 id="genesPerContext">Panel genes measured in each experiment</h3>
  <p>VCC 2026 asks for every one of the <span id="npanel"></span> panel genes (the 10x Flex probe-set genes) in every context. This table shows, for each of our experiments, how many of them its deposit actually measures, and how many of the missing ones matter: those above the 5 CPM gate in a 2026 validation context, where the DE metrics test them. Genes without an included Flex probe are rarely measured by Flex experiments and are usually below the gate anyway.</p>
  <p>Replogle 2022 and Nadig 2025 (K562, RPE1, Jurkat, HepG2) were sequenced genome-wide, but the authors deposited only the genes expressed in each line (every kept gene averages at least about 0.1 UMI per cell); the other genes would need realignment of the raw reads (AUDIT S8).</p>
  <p>Cost of the missing genes, measured on H1 (which itself measures 18,080 panel genes): for each H1 target, the share of its significant genes (after the 5 CPM gate and BH at 0.05) that the experiment does not measure, and the share of their summed |log2 fold change|. Medians over H1's targets; a missing gene is then predicted as no change. Small values mean the gaps fall mostly on genes that rarely respond.</p>
  <div class="tablebox" style="max-height:30rem"><table id="gctxTable"></table></div>
  <h3>Look up a gene</h3>
  <div class="controls"><label for="gq">Gene <input id="gq" type="search" placeholder="e.g. NOTCH2NLB"></label></div>
  <div class="panel" id="gqOut" style="margin-top:.6rem"><span class="cap">Type a panel gene symbol to see which experiments measure it.</span></div>

  <h2 id="batches">Batches: where the cells of each experiment come from</h2>
  <p>A batch is the unit cells were processed in together: a donor, a droplet lane, a replicate. Cells of one batch share technical (and for donors, genetic) differences, so a target whose cells come mostly from one batch can show that batch's difference as an effect. The pipeline therefore also computes batch-matched fold changes, where each target is compared with controls mixed in its own batch proportions (AUDIT S7, N8). Batch shift: the root mean square difference between the pooled and the batch-matched log2 fold changes per target, median over targets; large values mean the batch mix matters.</p>
  <div class="tablebox" style="max-height:30rem"><table id="batchTable"></table></div>
  <h3>Batches of one experiment</h3>
  <div class="controls"><label for="bsel">Experiment <select id="bsel"></select></label><span class="cap" id="bselN"></span></div>
  <div class="tablebox" style="max-height:22rem"><table id="batchRows"></table></div>

  <h2 id="merge">Can the contexts be pooled? Mergeability, step M1</h2>
  <div id="mergeBody">
  <p>For two contexts that perturbed the same target, we correlate the two effect vectors (each gene's change against that context's own controls) over the genes both measure above 5 CPM in both control pools, leaving out the target's own gene. Each context also gets a <b>reliability</b>: how well one random half of its cells agrees with the other half. The <b>disattenuated correlation</b> r / √(reliability<sub>A</sub> × reliability<sub>B</sub>) is about 1 when the two contexts measure the same effect up to noise, and lower when lab, chemistry, cell line or condition change the effect.</p>
  <p>Per pair: the median over the shared targets that have reliability ≥ 0.3 in both contexts, with a bootstrap 95% interval over those targets. Effects are batch-matched where a context has batches (AUDIT N8).</p>
  <div class="band" id="mband"></div>
  <div class="panel"><h3>Progress of the mergeability steps</h3><div class="tablebox"><table id="progTable"></table></div></div>
  <div class="panel"><h3>What M1 shows</h3><ul>
    <li>The check passes: Replogle genome-wide vs essential (same lab, line and condition) is the top pair, but even there r_disatt is about 0.91 and no pair's interval reaches 1, so effects never transfer perfectly.</li>
    <li>Same line, other lab or chemistry: Adamson vs Replogle K562 about 0.74; Replogle vs VIPerturb (Flex) about 0.57. Other lines: Jurkat or HepG2 about 0.6, K562 vs RPE1 about 0.44, H1 vs hESC about 0.62.</li>
    <li>Scale: on the depth-free fold change the slopes shrink (H1 vs hESC 2.10 to 1.28; K562 vs RPE1 2.61 to 1.72), so much of the earlier scale gap was sequencing depth; a per-context scale remains.</li>
    <li>Batch matching (N8) raises CD4 T agreement across states by 0.02 to 0.06; over all pairs the median change is about +0.005.</li>
    <li>Expression space: the K562 pairs agree across the three spaces; H1 vs hESC reads about 0.15 higher in the scorer's spaces than in the stored CP10k deltas (6x depth gap).</li>
  </ul></div>

  <h3>Agreement of selected pairs</h3>
  <p>The first rows are the upper bound: the same lab, cell line and condition, different screens. Every other comparison should be read against it.</p>
  <div class="panel chart" id="forest"></div>

  <h3>All pairs by design</h3>
  <p>Pairs with at least 10 reliable shared targets, grouped by what differs between the two contexts. Lab families: Adamson, Replogle 2020 and 2022 and the hESC screen are all Weissman lab.</p>
  <div class="tablebox" style="max-height:20rem"><table id="grpTable"></table></div>

  <h3>Agreement by effect strength</h3>
  <p>Targets binned by the number of significant genes in the weaker context. Weak targets have little signal, so their correlations are noisy, and they are most of the shared targets in many pairs.</p>
  <div class="tablebox" style="max-height:24rem"><table id="binTable"></table></div>

  <h3>Scale: does sequencing depth explain the slopes?</h3>
  <p>The stored half deltas (mean log1p CP10k per cell) grow with sequencing depth; the log2 fold change of mean CPM, which the scorer's DE metrics use, should not. Each point is a pair: x is log2 of the depth ratio of its controls (b over a), y is log2 of its Deming slope (b on a). If depth drives the scale, the half-delta slopes (filled) follow x and the fold-change slopes (outlined) do not.</p>
  <div class="panel chart" id="depthPlot"></div>
  <div class="cap" id="depthCap"></div>

  <h3>Batch matching (N8): pooled against batch-matched effects</h3>
  <p>Mean reliability per context with batches, against the pooled controls and against controls mixed to each target's own batch mix. A drop means part of the pooled reliability came from the batch mix rather than the perturbation.</p>
  <div class="tablebox" style="max-height:20rem"><table id="n8Table"></table></div>
  <p id="n8PairsNote"></p>
  <div class="tablebox" style="max-height:20rem"><table id="n8Pairs"></table></div>

  <h3>Expression space: does the statistic change the answer?</h3>
  <p>The scorer reads effects in per-cell CPM (DE metrics: log2 mean CPM ratio) and in group sums (pds and mse: log1p of 50,000 × gene sum / total). For three pairs the halves were recomputed from the cells in all three spaces, against pooled controls.</p>
  <div class="tablebox"><table id="spaceTable"></table></div>

  <h3>M2: agreement by what differs between two contexts</h3>
  <p>Named contrasts: each row is a set of pairs that differ in one thing. The value is the median r_disatt over the pairs, each pair weighted by its number of reliable targets, with a bootstrap 95% interval over pairs. r and r centered are the median raw correlations over all shared targets (most of them weak, hence the low values), before and after removing each context's mean response (M3); r_disatt pooled is the same pairs with pooled instead of batch-matched effects.</p>
  <div class="tablebox" style="max-height:24rem"><table id="contrastTable"></table></div>
  <h3>M2: all pairs grouped by design</h3>
  <div class="tablebox" style="max-height:20rem"><table id="designTable"></table></div>
  <h3>M2: how much each shared factor adds</h3>
  <p>One weighted regression over all pairs with at least 10 reliable targets: r_disatt on whether the two contexts share the cell line, the lab, the chemistry and the condition (weights: reliable targets; 95% interval from 1,000 bootstrap resamples of pairs). The factors overlap (no pair shares the line but differs in lab and chemistry), so read the coefficients as a description, not as separate causes.</p>
  <div class="tablebox"><table id="regTable"></table></div>
  <p><b>M3 so far.</b> Removing each context's mean response barely changes the correlations (r and r centered columns), so the gap between contexts is not a shared, target-independent shift. Batch matching raises agreement where batches are unbalanced (pluripotent pairs, through KOLF's sub-libraries: 0.31 pooled against 0.70 batch-matched; CD4 T: 0.50 to 0.52) and changes little elsewhere. Scale calibration cannot change a correlation, so it is tested in M4 instead.</p>
  <p><b>KOLF (fixed).</b> KOLF read as highly reliable (median split-half 0.38 at about 220 cells per target, against 0.05 for Orion) yet agreed with almost nothing (0.12 with H1). Its deposit has three sub-libraries (obs/batch: ALPHA, BETA, GAMMA); every target lies entirely in one of them, and the controls are 55% ALPHA, so against pooled controls each BETA and GAMMA target carried the sub-library difference, which both halves share. With obs/batch as the batch key, KOLF's reliability falls to 0.018 and it agrees with the other pluripotent contexts at 0.65 to 0.70 (H1 0.66, hESC 0.70, Feng 0.65).</p>
  <p><b>CD4 T (open).</b> CD4 T targets replicate between random halves of their cells (median r 0.45 for 300 random genome-wide targets, about 0.9 for targets such as NKX2-5 or IFFO2 that cannot act in T cells), but between donors D1+D2 and D3+D4 the same targets agree at a median of 0.04 (80% against 3.4% of targets above 0.3). Lanes and donor imbalance do not explain it. The likely cause is founder composition: each guide's cells descend from a limited number of transduced T cells with their own subtype mix, and a random cell split puts the same founders in both halves. CD4 T's reliabilities, its r_disatt values (also between its own conditions, which may share founders) and its weight in M4 are therefore not trustworthy until reliability is computed across donors and effects are averaged over donors.</p>

  <h3>M4: predicting a held-out context from the others</h3>
  <p>Each held-out context contributes only its controls (as a 2026 context would); other contexts of the same dataset are left out too. For each target, a prediction from the other contexts is correlated with the held-out context's half B, and divided by its own ceiling (half A against half B), so 1 means as good as a replicate. Targets whose ceiling is below 0.3 are left out. Predictors: <i>mean response</i> (the average effect over all perturbations, the scorer's baseline kind); <i>nearest context</i> (the source most similar by design: same line, then lab, condition, chemistry); <i>average</i> (plain mean over the contexts with the target); <i>precision weighted</i> (weights from each source's reliability); <i>calibrated</i> (precision weighted after removing each source's mean response and dividing by its scale, then adding the pooled mean response back). <i>own half A</i> is the check: it must be about 1.</p>
  <div id="locoBody"><p>M4 has not finished yet.</p></div>
  <p><b>Reading M4 (first version).</b> The check works: own half A scores 1 everywhere. From other datasets alone, the best predictor reaches about 0.78 of the replicate ceiling for Orion HCT116, 0.43 to 0.46 for RPE1 and hESC, 0.19 for H1 and about 0 for CD4 T at rest (see the CD4 T note above). The plain average over contexts is the best or close to it almost everywhere: precision weighting and per-source calibration do not help yet, and the most similar context by design is usually worse than the average. Where prediction works (Orion, RPE1, VIPerturb), the mean response alone already gets most of it, so what transfers between datasets is mainly the response shared by most perturbations. Caveats: genome-wide screens have few targets that replicate well enough to score (KOLF 7, VIPerturb 5, Orion 35), so their rows are uncertain; H1 is deep (its own halves agree at about 0.84), which makes every noisy prediction look weak against it; KOLF's own row used its fixed summary, but its partners' agreement tables are being recomputed. CD4 T is the outlier and is under investigation (progress table).</p>

  <h3>Every pair</h3>
  <div class="controls"><label for="pq">Find a context <input id="pq" type="search" placeholder="e.g. k562"></label><span class="cap" id="pqn"></span></div>
  <div class="tablebox"><table id="pairTable"></table></div>
  </div>
</div>
<div id="tip" hidden></div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = v => v === "" || v === null || v === undefined ? "" : (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(2)) : esc(v));
// Contexts added in the audit (not in Michael's pipeline): shaded rows and a "new" tag next to their names.
const NEW = new Set(D.new_contexts), NEWTAG = ' <span class="pill new">new</span>';
const isNew = r => r && typeof r === "object" && !Array.isArray(r) && [r.context, r.a, r.b, ...(r.pair ? r.pair.split(" vs ") : [])].some(c => NEW.has(c));
$("newNote").innerHTML = D.new_contexts.length ? `Rows shaded and tagged${NEWTAG} are the ${D.new_contexts.length} contexts added during the audit (${D.new_datasets.map(esc).join(", ")}); the other ${D.n_contexts - D.new_contexts.length} come from Michael's pipeline. In the heatmap their column names are in the same color.` : "";
$("pid").textContent = D.panel; $("gates").textContent = D.gates.join(", "); $("ngated").textContent = D.n_gated.toLocaleString("en-US");
const minCtx = Math.min(...D.targets.map(t => t.n_contexts)), minLine = Math.min(...D.targets.map(t => t.n_cell_lines));
$("band").innerHTML = [
  [D.n_contexts, "contexts"], [D.n_datasets, "datasets"], [D.n_lines, "distinct cell lines"], [D.n_unstim, "unstimulated contexts"],
  [D.new_contexts.length, "contexts added in the audit"], [D.n_targets, "panel targets"], [D.targets.filter(t => t.n_contexts === 0).length, "targets in no context"],
  [minCtx, "fewest contexts for a target"], [minLine, "fewest cell lines for a target"]
].map(([v, k]) => `<div><span class="v">${v}</span><span class="k">${k}</span></div>`).join("");

function table(el, cols, rows) {
  el.innerHTML = "<thead><tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map(r => `<tr class="${isNew(r) ? "new" : ""}">` + cols.map(c => `<td class="${c[2] || ""}">${c[3] ? c[3](r) : fmt(r[c[0]])}` +
      (["context", "a", "b"].includes(c[0]) && !Array.isArray(r) && NEW.has(r[c[0]]) ? NEWTAG : "") + "</td>").join("") + "</tr>").join("") + "</tbody>";
}
const condPill = r => `<span class="pill ${r.condition.startsWith("unstimulated") ? "" : "stim"}">${esc(r.condition)}</span>`;
table($("ctxTable"), [["dataset", "dataset"], ["context", "context", "mono"], ["cell_line", "cell line"], ["condition", "condition", "", condPill],
  ["perturbation_modality", "modality"], ["n_targets", "targets", "n"], ["n_panel_targets", "panel targets", "n"],
  ["median_cells_per_target", "median cells / target", "n"], ["n_genes_measured", "panel genes measured", "n"],
  ["n_missing_genes_above_gate_ABC", "missing genes above gate", "n"], ["mean_reliability_batchmatched", "mean reliability", "n"]], D.contexts);

function renderTargets() {
  const q = $("q").value.trim().toUpperCase();
  const rows = q ? D.targets.filter(t => t.panel_target.toUpperCase().includes(q)) : D.targets;
  $("qn").textContent = q ? `${rows.length} of ${D.targets.length}` : "";
  table($("tTable"), [["rank_least_covered", "rank", "n"], ["panel_target", "target", "mono"], ["n_cell_lines", "cell lines", "n"],
    ["n_contexts", "contexts", "n"], ["n_unstimulated_contexts", "unstim.", "n"], ["n_datasets", "datasets", "n"],
    ["total_cells", "cells", "n"], ["median_cells_per_context", "median cells / context", "n"],
    ["cell_lines", "cell lines", "wrap", r => esc(String(r.cell_lines).split(";").join(", "))]], rows);
}
$("q").addEventListener("input", renderTargets); renderTargets();
table($("gTable"), [["gene", "gene", "mono"], ["n_contexts_measured", "contexts", "n"], ["n_datasets_measured", "datasets", "n"],
  ["gates", "above gate in", "", r => D.gates.filter(g => r["gated_in_" + g] === true || r["gated_in_" + g] === "True").join(" ")]], D.genes_few);

function hist(el, series, xlabel, ylabel) {   // series: [{counts:{x:n}, style:"fill"|"line"}]; labels carry the axis name and its unit
  const xs = [...new Set(series.flatMap(s => Object.keys(s.counts).map(Number)))].sort((a, b) => a - b);
  const lo = Math.min(...xs), hi = Math.max(...xs), ymax = Math.max(...series.flatMap(s => Object.values(s.counts)));
  const W = 560, H = 240, m = {l: 62, r: 10, t: 12, b: 44}, iw = W - m.l - m.r, ih = H - m.t - m.b, n = hi - lo + 1, bw = iw / n;
  const y = v => m.t + ih - v / ymax * ih;
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(ylabel)} by ${esc(xlabel)}">`;
  s += `<text transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle" font-size="12" style="fill:var(--fg)">${esc(ylabel)}</text>`;
  const step = Math.pow(10, Math.floor(Math.log10(ymax))) * (ymax / Math.pow(10, Math.floor(Math.log10(ymax))) > 5 ? 2 : 1);
  for (let t = 0; t <= ymax; t += step) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--rule)"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">${t.toLocaleString("en-US")}</text>`;
  series.forEach(se => { for (const [k, v] of Object.entries(se.counts)) { const x = m.l + (Number(k) - lo) * bw;
    s += se.style === "fill" ? `<rect x="${x + 1}" y="${y(v)}" width="${Math.max(bw - 2, 1)}" height="${y(0) - y(v)}" style="fill:var(--accent)" rx="1"><title>${k}: ${v}</title></rect>`
      : `<rect x="${x + 1.5}" y="${y(v)}" width="${Math.max(bw - 3, 1)}" height="${y(0) - y(v)}" style="fill:none;stroke:var(--warm);stroke-width:1.5"><title>${k}: ${v}</title></rect>`; } });
  const every = Math.max(1, Math.ceil(n / 14));
  for (let k = lo; k <= hi; k += every) s += `<text x="${m.l + (k - lo + .5) * bw}" y="${H - m.b + 15}" text-anchor="middle" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">${k}</text>`;
  s += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle" font-size="12" style="fill:var(--fg)">${esc(xlabel)}</text></svg>`;
  el.innerHTML = s;
}
const count = (arr, f) => arr.reduce((o, t) => (o[f(t)] = (o[f(t)] || 0) + 1, o), {});
hist($("hCtx"), [{counts: count(D.targets, t => t.n_contexts), style: "fill"}, {counts: count(D.targets, t => t.n_unstimulated_contexts), style: "line"}],
  "Contexts with cells for the target (number of contexts)", "Panel targets (count)");
hist($("hLine"), [{counts: count(D.targets, t => t.n_cell_lines), style: "fill"}],
  "Distinct cell lines with cells for the target (number of cell lines)", "Panel targets (count)");
hist($("hGene"), [{counts: D.gene_hist_gated, style: "fill"}, {counts: D.gene_hist_all, style: "line"}],
  "Contexts measuring the gene (number of contexts)", "Panel genes (count)");

// ---------------- panel genes per experiment
$("npanel").textContent = D.n_genes.toLocaleString("en-US");
const bar = r => `<div style="display:flex;align-items:center;gap:.5rem"><div style="flex:0 0 7rem;height:.6rem;background:var(--heat0);border-radius:2px"><div style="height:100%;width:${r.pct_panel_genes_measured}%;background:var(--accent);border-radius:2px"></div></div><span>${fmt(r.n_genes_measured)}</span></div>`;
table($("gctxTable"), [["context", "context", "mono"], ["chem", "chemistry"], ["n_native_genes", "genes in deposit", "n"],
  ["n_genes_measured", "panel genes measured", "", bar], ["pct_panel_genes_measured", "% of panel", "n", r => r.pct_panel_genes_measured.toFixed(1)],
  ["n_panel_genes_missing", "missing", "n"], ["n_missing_genes_above_gate_ABC", "missing, above gate in A/B/C", "n"],
  ...D.gate_letters.map(g => ["n_missing_genes_above_gate_" + g, "above gate in " + g, "n"]),
  ["n_missing_genes_without_included_probe", "missing, no Flex probe", "n"],
  ["set_share_median", "H1 DE genes lost (median %)", "n", r => r.set_share_median == null ? "" : (100 * r.set_share_median).toFixed(2)],
  ["magnitude_share_median", "H1 |LFC| lost (median %)", "n", r => r.magnitude_share_median == null ? "" : (100 * r.magnitude_share_median).toFixed(2)],
  ["n_genes_matched_ensembl", "matched by Ensembl id", "n"], ["n_genes_matched_hgnc", "matched by HGNC previous symbol", "n"]], D.gene_ctx);
const UPPER = Object.fromEntries(Object.keys(D.gene_lookup).map(g => [g.toUpperCase(), g]));
$("gq").addEventListener("input", () => {
  const q0 = $("gq").value.trim(), q = UPPER[q0.toUpperCase()] || q0, r = D.gene_lookup[q];
  if (!q0) { $("gqOut").innerHTML = '<span class="cap">Type a panel gene symbol to see which experiments measure it.</span>'; return; }
  if (!r) { $("gqOut").innerHTML = `<span class="cap">${esc(q)} is not a panel gene; try its current HGNC symbol.</span>`; return; }
  const miss = r[0] ? r[0].split(";").map(Number) : [], nm = D.ctx_names.length - miss.length;
  $("gqOut").innerHTML = `<b class="mono">${esc(q)}</b>: measured in ${nm} of ${D.ctx_names.length} experiments; ${r[1] ? "above the 5 CPM gate in " + r[1].split("").join(", ") : "below the gate in every 2026 validation context"}.` +
    (miss.length ? `<div class="cap" style="margin-top:.4rem">Not measured in: ${miss.map(i => `<span class="mono">${esc(D.ctx_names[i])}</span>`).join(", ")}</div>` : "");
});

// ---------------- batches
table($("batchTable"), [["context", "context", "mono"], ["origin", "what one batch is", "wrap"], ["batch_key", "column read", "mono"],
  ["n_batches", "batches", "n"], ["ctrl_min", "controls per batch: min", "n"], ["ctrl_median", "median", "n"], ["ctrl_max", "max", "n"],
  ["pert_median", "perturbed cells per batch (median)", "n"], ["targets_median", "targets per batch (median)", "n"],
  ["median_batch_shift_rms", "batch shift", "n"], ["examples", "example batches", "mono"]], D.batch_ctx);
const withB = D.batch_ctx.filter(r => D.batch_rows[r.context]);
$("bsel").innerHTML = withB.map(r => `<option value="${esc(r.context)}">${esc(r.context)}</option>`).join("");
const showB = () => { const c = $("bsel").value, R = D.batch_rows[c] || [];
  $("bselN").textContent = `${R.length} batches`;
  table($("batchRows"), [[0, "batch", "mono"], [1, "controls", "n"], [2, "perturbed cells", "n"], [3, "targets", "n"]], R); };
$("bsel").addEventListener("change", showB);
const cd4 = withB.find(r => r.context.startsWith("cd4t")); if (cd4) $("bsel").value = cd4.context; showB();

// ---------------- mergeability
const M = D.merge;
const f2 = v => v === null || v === undefined ? "" : Number(v).toFixed(2);
const ci = (v, lo, hi) => v === null || v === undefined ? "" : `${f2(v)} <span style="color:var(--muted)">[${f2(lo)}, ${f2(hi)}]</span>`;
const short = c => c.replace("cd4t_marson2025__", "CD4T ").replace("replogle_nadig__", "Nadig ").replace("replogle_gwps__", "Replogle GW ").replace("replogle2020__", "Replogle 2020 ").replace("jiang2025_", "Jiang ").replace("vcc2025__", "").replace("__", " ");
if (!M) { $("mergeBody").innerHTML = "<p>The M1 rerun has not finished yet; this section fills in when its CSVs exist.</p>"; }
else {
  const ub = M.named.find(r => r.a.startsWith("replogle_gwps") || r.b.startsWith("replogle_gwps"));
  $("mband").innerHTML = [
    [M.pairs.length, "pairs with ≥ 20 shared targets"], [M.pairs.filter(r => r.n_reliable >= 10).length, "pairs with ≥ 10 reliable targets"],
    [ub ? f2(ub.r_disatt) : "", "upper bound (Replogle GW vs essential)"],
  ].map(([v, k]) => `<div><span class="v">${v}</span><span class="k">${k}</span></div>`).join("");

  (function forest() {
    const R = M.named, W = 760, rh = 22, m = {l: 300, r: 70, t: 26, b: 34}, H = m.t + m.b + R.length * rh + new Set(R.map(r => r.group)).size * 16;
    const x = v => m.l + Math.max(0, Math.min(1.3, v)) / 1.3 * (W - m.l - m.r);
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="disattenuated correlation per pair">`;
    for (const t of [0, .25, .5, .75, 1, 1.25]) s += `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t - 6}" y2="${H - m.b + 4}" style="stroke:var(--rule)${t === 1 ? ";stroke-dasharray:4 3;stroke:var(--muted)" : ""}"/><text x="${x(t)}" y="${H - m.b + 18}" text-anchor="middle" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">${t}</text>`;
    s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="12" style="fill:var(--fg)">disattenuated correlation (1 = same effect up to noise)</text>`;
    let y = m.t, last = null;
    R.forEach(r => {
      if (r.group !== last) { y += 16; s += `<text x="8" y="${y - 4}" font-size="11.5" style="fill:var(--muted);font-family:var(--font-display);font-weight:650">${esc(r.group)}</text>`; last = r.group; }
      const cy = y + rh / 2;
      s += `<text x="16" y="${cy + 4}" font-size="12" style="fill:var(--fg)">${esc(short(r.a))} vs ${esc(short(r.b))}</text>`;
      if (r.r_disatt_lo !== null) s += `<line x1="${x(r.r_disatt_lo)}" x2="${x(r.r_disatt_hi)}" y1="${cy}" y2="${cy}" style="stroke:var(--accent);stroke-width:2"/>`;
      if (r.r_disatt !== null) s += `<circle cx="${x(r.r_disatt)}" cy="${cy}" r="4.5" style="fill:var(--accent)"><title>${esc(r.label)}: ${f2(r.r_disatt)} [${f2(r.r_disatt_lo)}, ${f2(r.r_disatt_hi)}], ${r.n_reliable} reliable targets</title></circle>`;
      s += `<text x="${W - m.r + 8}" y="${cy + 4}" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">n=${r.n_reliable}</text>`;
      y += rh;
    });
    $("forest").innerHTML = s + "</svg>";
  })();

  table($("grpTable"), [["group", "what differs"], ["n_pairs", "pairs", "n"], ["median", "median r_disatt", "n"], ["q1", "25%", "n"], ["q3", "75%", "n"]], M.groups);
  table($("binTable"), [["pair", "pair", "mono", r => esc(short(r.pair.split(" vs ")[0])) + " vs " + esc(short(r.pair.split(" vs ")[1]))], ["strength", "strength"],
    ["n_targets", "targets", "n"], ["n_reliable", "reliable", "n"], ["r", "median r", "n"], ["r_disatt", "r_disatt [95%]", "n", r => ci(r.r_disatt, r.lo, r.hi)]], M.bins);

  (function depthPlot() {
    const P = M.depth; if (!P.length) { $("depthPlot").textContent = "no pairs"; return; }
    const W = 640, H = 360, m = {l: 52, r: 14, t: 14, b: 42};
    const xs = P.map(p => p.x), ys = P.flatMap(p => [p.y_half, p.y_lfc]);
    const x0 = Math.min(...xs, -1), x1 = Math.max(...xs, 1), y0 = Math.max(Math.min(...ys, -1), -4), y1 = Math.min(Math.max(...ys, 1), 4);
    const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => m.t + (1 - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="slope against depth ratio">`;
    for (let t = Math.ceil(x0); t <= x1; t++) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--rule)"/><text x="${X(t)}" y="${H - m.b + 15}" text-anchor="middle" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">${t}</text>`;
    for (let t = Math.ceil(y0); t <= y1; t++) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}" style="stroke:var(--rule)"/><text x="${m.l - 6}" y="${Y(t) + 4}" text-anchor="end" font-size="11" style="fill:var(--muted);font-family:var(--font-mono)">${t}</text>`;
    s += `<line x1="${X(Math.max(x0, y0))}" y1="${Y(Math.max(x0, y0))}" x2="${X(Math.min(x1, y1))}" y2="${Y(Math.min(x1, y1))}" style="stroke:var(--muted);stroke-dasharray:4 3"/>`;
    P.forEach(p => { s += `<circle cx="${X(p.x)}" cy="${Y(p.y_half)}" r="2.6" style="fill:var(--accent);opacity:.7"><title>${esc(p.a)} vs ${esc(p.b)}: half-delta slope ${f2(Math.pow(2, p.y_half))}</title></circle>`;
      s += `<circle cx="${X(p.x)}" cy="${Y(p.y_lfc)}" r="2.6" style="fill:none;stroke:var(--warm);stroke-width:1.2;opacity:.8"><title>${esc(p.a)} vs ${esc(p.b)}: fold-change slope ${f2(Math.pow(2, p.y_lfc))}</title></circle>`; });
    s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" style="fill:var(--fg)">log2 control depth ratio (b / a)</text>`;
    s += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" font-size="12" style="fill:var(--fg)">log2 slope (b on a)</text></svg>`;
    $("depthPlot").innerHTML = s;
    const corr = (a, b) => { const n = a.length, ma = a.reduce((x, y) => x + y) / n, mb = b.reduce((x, y) => x + y) / n; let sab = 0, saa = 0, sbb = 0;
      for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };
    $("depthCap").textContent = `${P.length} pairs with ≥ 10 reliable targets. Correlation with the depth ratio: half-delta slopes ${f2(corr(xs, P.map(p => p.y_half)))}, fold-change slopes ${f2(corr(xs, P.map(p => p.y_lfc)))}. Dashed: slope equal to the depth ratio.`;
  })();

  table($("n8Table"), [["context", "context", "mono"], ["mean_reliability", "pooled", "n"], ["mean_reliability_batchmatched", "batch-matched", "n"],
    ["d", "change", "n", r => f2(r.mean_reliability_batchmatched - r.mean_reliability)]], M.n8_rel.slice().sort((a, b) => (a.mean_reliability_batchmatched - a.mean_reliability) - (b.mean_reliability_batchmatched - b.mean_reliability)));
  if (M.n8_pairs) {
    const P = M.n8_pairs.filter(r => r.a.startsWith("cd4t") || r.b.startsWith("cd4t"));
    const med = a => { const v = a.filter(x => x !== null).sort((x, y) => x - y); return v.length ? v[Math.floor(v.length / 2)] : null; };
    $("n8PairsNote").innerHTML = `Pairs involving CD4 T (${P.length} with ≥ 10 reliable targets both ways): median r_disatt pooled ${f2(med(P.map(r => r.r_disatt_pooled)))}, batch-matched ${f2(med(P.map(r => r.r_disatt)))}. Over all ${M.n8_pairs.length} pairs with batch-matched effects: pooled ${f2(med(M.n8_pairs.map(r => r.r_disatt_pooled)))}, batch-matched ${f2(med(M.n8_pairs.map(r => r.r_disatt)))}.`;
    table($("n8Pairs"), [["a", "a", "mono", r => esc(short(r.a))], ["b", "b", "mono", r => esc(short(r.b))], ["rel_a_pooled", "rel a pooled", "n"], ["rel_a", "rel a matched", "n"],
      ["rel_b_pooled", "rel b pooled", "n"], ["rel_b", "rel b matched", "n"], ["r_disatt_pooled", "r_disatt pooled", "n"], ["r_disatt", "r_disatt matched", "n"]], P.sort((x, y) => (y.r_disatt || 0) - (x.r_disatt || 0)));
  } else $("n8PairsNote").textContent = "The pooled-effects run has not finished yet.";
  if (M.space) table($("spaceTable"), [["a", "a", "mono", r => esc(short(r.a))], ["b", "b", "mono", r => esc(short(r.b))], ["space", "space", "mono"],
    ["n_reliable", "reliable targets", "n"], ["rel_a", "rel a", "n"], ["rel_b", "rel b", "n"], ["r", "median r", "n"],
    ["r_disatt", "r_disatt [95%]", "n", r => ci(r.r_disatt, r.r_disatt_lo, r.r_disatt_hi)], ["slope", "slope [95%]", "n", r => ci(r.slope, r.slope_lo, r.slope_hi)]], M.space);
  else $("spaceTable").outerHTML = "<p>The expression-space check has not finished yet.</p>";
  table($("progTable"), [["step", "step", "mono"], ["what", "what"], ["status", "status"]], M.progress);
  if (M.design) {
    const dc = [["n_pairs", "pairs", "n"], ["n_reliable", "reliable targets", "n"], ["r_disatt", "r_disatt [95%]", "n", r => ci(r.r_disatt, r.lo, r.hi)],
      ["r", "r", "n"], ["r_centered", "r centered", "n"], ["r_disatt_pooled", "r_disatt pooled", "n"], ["slope_lfc", "fold-change slope", "n"]];
    table($("contrastTable"), [["contrast", "contrast"], ...dc], M.design.contrasts);
    table($("designTable"), [["design", "what differs"], ...dc], M.design.by_design);
    table($("regTable"), [["term", "term", "mono"], ["coef", "added r_disatt [95%]", "n", r => ci(r.coef, r.lo, r.hi)], ["n_pairs_with", "pairs sharing it", "n"]], M.design.regression);
  }
  if (M.loco && M.loco.length) {
    const P = ["own_half_A", "calibrated", "precision_weighted", "average", "nearest_context", "mean_response"];
    const H = [...new Set(M.loco.map(r => r.held_out))];
    const cell = (h, p) => { const r = M.loco.find(x => x.held_out === h && x.predictor === p); return r ? ci(r.r_norm, r.lo, r.hi) : ""; };
    const n = h => { const r = M.loco.find(x => x.held_out === h); return r ? r.n_targets : ""; };
    $("locoBody").innerHTML = `<div class="tablebox"><table><thead><tr><th>held-out context</th><th>targets</th>${P.map(p => `<th>${p.replace(/_/g, " ")}</th>`).join("")}</tr></thead><tbody>` +
      H.map(h => `<tr><td class="mono">${esc(short(h))}</td><td class="n">${n(h)}</td>${P.map(p => `<td class="n">${cell(h, p)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
  }
  function renderPairs() {
    const q = $("pq").value.trim().toLowerCase();
    const rows = q ? M.pairs.filter(r => (r.a + " " + r.b).toLowerCase().includes(q)) : M.pairs;
    $("pqn").textContent = `${rows.length} of ${M.pairs.length}`;
    table($("pairTable"), [["a", "a", "mono"], ["b", "b", "mono"], ["group", "what differs"], ["chem", "chemistry"], ["n_targets", "shared", "n"], ["n_reliable", "reliable", "n"],
      ["r", "median r", "n"], ["r_disatt", "r_disatt [95%]", "n", r => ci(r.r_disatt, r.r_disatt_lo, r.r_disatt_hi)],
      ["slope_lfc", "fold-change slope [95%]", "n", r => ci(r.slope_lfc, r.slope_lfc_lo, r.slope_lfc_hi)]], rows);
  }
  $("pq").addEventListener("input", renderPairs); renderPairs();
}

// heatmap: log-scaled cell counts
(function () {
  // Axes: x = contexts (titled above the column labels), y = panel targets by coverage rank (titled on the left, ticked every 50 ranks),
  // color = cells per target in the context (legend below, log scale).
  const c = $("heat"), M = D.matrix, cw = 12, rh = 3, top = 196, left = 64, legendH = 72, dpr = window.devicePixelRatio || 1;
  const maxCells = Math.max(...M.values.flat()), minCells = Math.min(...M.values.flat().filter(v => v > 0));
  const gridW = M.cols.length * cw, gridH = M.rows.length * rh;
  const W = left + gridW + 8, H = top + gridH + legendH;
  c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px";
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement), col = n => css.getPropertyValue(n).trim();
  g.fillStyle = col("--surface"); g.fillRect(0, 0, W, H);
  const max = Math.log10(maxCells + 1), acc = col("--accent"), base = col("--heat0"), fg = col("--fg"), muted = col("--muted");
  const mono = "10px " + col("--font-mono"), body = "12px " + col("--font-body");
  M.rows.forEach((r, i) => M.cols.forEach((cname, j) => {
    const v = M.values[i][j]; g.fillStyle = v ? acc : base; g.globalAlpha = v ? 0.18 + 0.82 * Math.log10(v + 1) / max : 1;
    g.fillRect(left + j * cw, top + i * rh, cw - 1, rh); }));
  g.globalAlpha = 1;
  // x axis: title, then the rotated context names
  g.fillStyle = fg; g.font = body; g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillText("Context (dataset__cell line or condition), grouped by dataset", left, 14);
  g.fillStyle = muted; g.font = mono;
  M.cols.forEach((cname, j) => { g.save(); g.translate(left + j * cw + cw / 2 + 3, top - 6); g.rotate(-Math.PI / 2.6); g.fillStyle = NEW.has(cname) ? col("--new") : muted; g.fillText(cname, 0, 0); g.restore(); });
  // y axis: rank ticks every 50 targets, then a rotated title
  g.fillStyle = muted; g.font = mono; g.textAlign = "right"; g.textBaseline = "middle"; g.strokeStyle = col("--rule");
  for (let rank = 1; rank <= M.rows.length; rank += 50) {
    const yy = top + (rank - 1) * rh + rh / 2;
    g.beginPath(); g.moveTo(left - 4, yy); g.lineTo(left - 1, yy); g.stroke();
    g.fillText(String(rank), left - 7, yy); }
  g.save(); g.translate(14, top + gridH / 2); g.rotate(-Math.PI / 2);
  g.fillStyle = fg; g.font = body; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText("Panel target, by coverage rank (1 = least covered)", 0, 0); g.restore();
  // color legend: a log-scaled ramp with the unit
  const lx = left, ly = top + gridH + 34, lw = Math.min(220, gridW), lh = 10;
  g.fillStyle = fg; g.font = body; g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillText("Cells per target in the context (number of cells, log scale; 0 = not perturbed)", lx, ly - 10);
  g.fillStyle = base; g.fillRect(lx, ly, 14, lh);
  for (let px = 0; px < lw; px++) { const v = Math.pow(10, max * px / (lw - 1)) - 1;
    g.fillStyle = acc; g.globalAlpha = 0.18 + 0.82 * Math.log10(v + 1) / max; g.fillRect(lx + 20 + px, ly, 1, lh); }
  g.globalAlpha = 1; g.fillStyle = muted; g.font = mono; g.textBaseline = "top";
  g.textAlign = "center"; g.fillText("0", lx + 7, ly + lh + 3);
  g.textAlign = "left"; g.fillText("1", lx + 20, ly + lh + 3);
  g.textAlign = "right"; g.fillText(maxCells.toLocaleString("en-US"), lx + 20 + lw, ly + lh + 3);
  const tip = $("tip");
  c.addEventListener("mousemove", e => { const b = c.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    const j = Math.floor((x - left) / cw), i = Math.floor((y - top) / rh);
    if (i >= 0 && i < M.rows.length && j >= 0 && j < M.cols.length) { tip.hidden = false; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
      tip.textContent = `${M.rows[i]} · ${M.cols[j]}: ${M.values[i][j]} cells`; } else tip.hidden = true; });
  c.addEventListener("mouseleave", () => tip.hidden = true);
})();
</script>
"""

if __name__ == "__main__":
    main()
