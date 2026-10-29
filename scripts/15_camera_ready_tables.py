"""Camera-ready tables: reads saved outputs only (never trains or re-evaluates).

Inputs: the original sweeps (results/full_scale_calibration_stats.json,
results/full_scale_binary_sweep_stats.json), the camera-ready runs
(results/runs/cr_*.json from 13_camera_ready_experiments.py, latest file per
run name), the inference-only re-analysis (results/camera_ready_reanalysis.json
from 14) and the leakage screen (results/text_leakage_screen.json from 12).

Writes results/camera_ready_summary.md and LaTeX row blocks
results/cr_rows_*.tex that the paper's tables are built from. Cells are
mean±std over training seeds (pandas std, ddof=1, as src/report.py); a cell
with fewer than 3 finished seeds is marked with a dagger so a partial sweep
can never be mistaken for a finished one.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
METRICS = ("auroc", "auprc", "f1", "accuracy")
N_SEEDS = 3

ORIGINAL_NAMES = {"fusion=concat": "Fusion (GIN+text)", "graph-only": "Graph-only (GIN)",
                  "text-only": "Text-only", "fingerprint-mlp": "Fingerprint-MLP"}
CR_NAMES = {"fp-text": "Fingerprint+text", "gcn-graph-only": "GCN graph-only",
            "gcn-fusion=concat": "GCN fusion (GCN+text)", "fusion=gated": "Fusion, gated",
            "fusion=cross_attention": "Fusion, cross-attention",
            "filtered-fusion=concat": "Fusion, filtered text", "filtered-text-only": "Text-only, filtered text",
            "generic-fusion=concat": "Fusion, interaction wording removed",
            "generic-text-only": "Text-only, interaction wording removed"}


def load_rows():
    rows = []
    for dataset, fname in (("drugbank", "full_scale_calibration_stats.json"),
                           ("biosnap", "full_scale_binary_sweep_stats.json")):
        with open(os.path.join(RES, fname)) as f:
            for r in json.load(f)["summary_table"]:
                rows.append({"dataset": dataset, "model": ORIGINAL_NAMES[r["model"]], "seed": r["seed"],
                             "split": "T" if r["eval_split"] == "test" else r["eval_split"].upper(),
                             **{m: r[m] for m in METRICS}})
    latest = {}
    for path in glob.glob(os.path.join(RES, "runs", "cr_*.json")):
        with open(path) as f:
            run = json.load(f)
        name = run["result"]["run_name"]
        stamp = int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0])
        if name not in latest or stamp > latest[name][0]:
            latest[name] = (stamp, run)
    for _, run in latest.values():
        cfg, res = run["config"], run["result"]
        for split, m in res["metrics_by_split"].items():
            rows.append({"dataset": cfg["dataset"], "model": CR_NAMES[cfg["name"]], "seed": cfg["seed"],
                         "split": "T" if split == "test" else split.upper(), **{k: m[k] for k in METRICS}})
    return pd.DataFrame(rows)


def cell(vals, digits=3):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return {"mean": mean, "std": std, "n": len(vals),
            "txt": f"{mean:.{digits}f}$\\pm${std:.{digits}f}" + ("" if len(vals) == N_SEEDS else "$^\\dagger$")}


def agg(df, dataset, model, split, metric="auprc"):
    sel = df[(df.dataset == dataset) & (df.model == model) & (df.split == split)].sort_values("seed")
    return cell(list(sel[metric]))


def gap(df, dataset, model, split, metric="auprc"):
    sel = df[(df.dataset == dataset) & (df.model == model)]
    t = sel[sel.split == "T"].set_index("seed")[metric]
    s = sel[sel.split == split].set_index("seed")[metric]
    common = sorted(set(t.index) & set(s.index))
    return cell([t[i] - s[i] for i in common])


def latex_row(label, cells, best=None):
    parts = []
    for i, c in enumerate(cells):
        if c is None:
            parts.append("--")
        elif best is not None and best[i]:
            parts.append(f"\\textbf{{{c['txt']}}}")
        else:
            parts.append(c["txt"])
    return f"{label} & " + " & ".join(parts) + " \\\\"


def bold_mask(rows_cells, higher_is_better=True):
    """per column: True for the row(s) holding the best mean (ties at 3 d.p. all bold)"""
    n_cols = len(rows_cells[0])
    masks = [[False] * n_cols for _ in rows_cells]
    for j in range(n_cols):
        vals = [(round(r[j]["mean"], 3) if r[j] else None) for r in rows_cells]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        target = max(present) if higher_is_better else min(present)
        for i, v in enumerate(vals):
            masks[i][j] = v == target
    return masks


def plot_inductive_gap(df, out_path):
    """src/report.py's Figure 1 (same Okabe-Ito colors, size, error bars) with the
    camera-ready Fingerprint+text series added and the in-figure title dropped
    (the caption carries it). Palette checked with the dataviz validator: all
    checks pass; orange/purple are low-contrast on white, so the exact values
    are in Table 1."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
    import matplotlib.pyplot as plt
    models =[("Fusion (GIN+text)", "#0072B2"), ("Graph-only (GIN)", "#E69F00"), ("Text-only", "#009E73"),
              ("Fingerprint-MLP", "#D55E00"), ("Fingerprint+text", "#CC79A7")]
    splits = ["T", "S1", "S2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, (dataset, title) in zip(axes, (("drugbank", "DrugBank"), ("biosnap", "BioSNAP"))):
        x = np.arange(len(splits))
        width = 0.8 / len(models)
        for i, (m, color) in enumerate(models):
            cells = [agg(df, dataset, m, s) for s in splits]
            means = [c["mean"] if c else np.nan for c in cells]
            stds = [c["std"] if c and not np.isnan(c["std"]) else 0 for c in cells]
            ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=2, label=m, color=color,
                   edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(["Transductive", "S1", "S2"])
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_rows()
    plot_inductive_gap(df, os.path.join(ROOT, "figures", "inductive_gap_camera_ready.pdf"))
    out_md = ["# Camera-ready summary (generated by scripts/15_camera_ready_tables.py)", ""]
    blocks = {}

    # --- Table 1: AUPRC, both datasets, five structure/text models ---
    main_models = ["Fusion (GIN+text)", "Graph-only (GIN)", "Text-only", "Fingerprint-MLP", "Fingerprint+text"]
    cols = [("drugbank", "T"), ("drugbank", "S1"), ("drugbank", "S2"), ("biosnap", "T"), ("biosnap", "S1"), ("biosnap", "S2")]
    table = [[agg(df, d, m, s) for d, s in cols] for m in main_models]
    masks = bold_mask(table)
    blocks["main_auprc"] = [latex_row(m, c, b) for m, c, b in zip(main_models, table, masks)]

    # --- Table 2: gap, both datasets ---
    gap_lines = []
    for k, (dataset, label) in enumerate((("drugbank", "DrugBank"), ("biosnap", "BioSNAP"))):
        rows = [[gap(df, dataset, m, "S1"), gap(df, dataset, m, "S2")] for m in main_models]
        masks = bold_mask(rows, higher_is_better=False)
        if k:
            gap_lines.append("\\cmidrule(lr){1-4}")
        for i, (m, c, b) in enumerate(zip(main_models, rows, masks)):
            first = f"\\multirow{{{len(main_models)}}}{{*}}{{{label}}}" if i == 0 else ""
            gap_lines.append(latex_row(f"{first} & {m}", c, b))
    blocks["gap"] = gap_lines

    # --- controls table: DrugBank S1/S2 AUPRC (+ AUROC) ---
    groups = [
        ("Reference (Table 1)", ["Fusion (GIN+text)", "Graph-only (GIN)"]),
        ("Text-leakage control", ["Fusion, interaction wording removed", "Text-only, interaction wording removed",
                                  "Fusion, filtered text", "Text-only, filtered text"]),
        ("Structure encoder", ["Fingerprint+text", "Fingerprint-MLP", "GCN fusion (GCN+text)", "GCN graph-only"]),
        ("Fusion mechanism", ["Fusion, gated", "Fusion, cross-attention"]),
    ]
    ctrl_lines = []
    for title, models in groups:
        ctrl_lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for m in models:
            cells = [agg(df, "drugbank", m, "S1"), agg(df, "drugbank", m, "S2"),
                     agg(df, "drugbank", m, "S1", "auroc"), agg(df, "drugbank", m, "S2", "auroc")]
            ctrl_lines.append(latex_row(f"\\quad {m}", cells))
    blocks["controls"] = ctrl_lines

    # --- appendix: all metrics for the camera-ready models ---
    full_lines = []
    for dataset in ("drugbank", "biosnap"):
        for m in CR_NAMES.values():
            for split in ("T", "S1", "S2"):
                cells = [agg(df, dataset, m, split, met) for met in METRICS]
                if any(cells):
                    full_lines.append(latex_row(f"{dict(drugbank='DrugBank', biosnap='BioSNAP')[dataset]} & {m} & {split}", cells))
    blocks["new_models_all_metrics"] = full_lines

    # --- re-analysis tables (common classes, S1 partner, similarity strata) ---
    re_path = os.path.join(RES, "camera_ready_reanalysis.json")
    if os.path.exists(re_path):
        with open(re_path) as f:
            re = json.load(f)
        summ = re["summary"]

        def rc(label, key):
            v = summ.get(label, {}).get(key)
            if not v:
                return None
            a = v["auprc"] if "auprc" in v else v
            return {"mean": a["mean"], "std": a["std"], "n": a["n"],
                    "txt": f"{a['mean']:.3f}$\\pm${a['std']:.3f}" + ("" if a["n"] == N_SEEDS else "$^\\dagger$")}

        labels_re = {"Fusion (GIN+text)": "Fusion (GIN+text)", "Graph-only (GIN)": "Graph-only (GIN)",
                     "Text-only": "Text-only", "Fingerprint-MLP": "Fingerprint-MLP",
                     "Fingerprint+text": "Fingerprint+text", "GCN graph-only": "GCN graph-only",
                     "GCN fusion": "GCN fusion (GCN+text)"}
        present = [l for l in labels_re if l in summ]
        common_rows = [[rc(l, "T_common"), rc(l, "S1_common"), rc(l, "S2_common"),
                        rc(l, "gap_T_minus_S1_common_auprc"), rc(l, "gap_T_minus_S2_common_auprc")] for l in present]
        val_masks = bold_mask([r[:3] for r in common_rows])
        gap_masks = bold_mask([r[3:] for r in common_rows], higher_is_better=False)
        blocks["common_class"] = [latex_row(labels_re[l], r, vm + gm)
                                  for l, r, vm, gm in zip(present, common_rows, val_masks, gap_masks)]
        part_rows = [[rc(l, "S1_train_partner"), rc(l, "S1_val_partner"), rc(l, "S1_sim_low"), rc(l, "S1_sim_mid"),
                      rc(l, "S1_sim_high"), rc(l, "S2_sim_low"), rc(l, "S2_sim_high")] for l in present]
        blocks["partner_similarity"] = [latex_row(labels_re[l], r, m)
                                        for l, r, m in zip(present, part_rows, bold_mask(part_rows))]
        out_md += [f"Common evaluable class set: {len(re['common_classes'])} types; "
                   f"re-analysis sanity max |diff| = {re['sanity_max_abs_diff_auprc']:.1e}", "",
                   "```", json.dumps(re["composition"], indent=1), json.dumps(re["similarity"], indent=1), "```", ""]

    for name, lines in blocks.items():
        with open(os.path.join(RES, f"cr_rows_{name}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        out_md += [f"## {name}", "", "```latex", *lines, "```", ""]

    seeds_done = df[df.model.isin(CR_NAMES.values())].groupby(["dataset", "model", "split"])["seed"].nunique()
    out_md += ["## camera-ready runs finished (seeds per dataset/model/split)", "", "```", seeds_done.to_string(), "```"]
    with open(os.path.join(RES, "camera_ready_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_md) + "\n")
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
