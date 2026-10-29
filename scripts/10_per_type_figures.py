"""Plotting-only regeneration of the per-type cold-start figures for the paper.
Reads results/per_type_drugbank.csv exactly as written by
09_per_type_drugbank_breakdown.py -- no recomputation, no checkpoint loading,
no retraining. Produces:

  figures/per_type_hist.pdf    -- histogram of the per-type fusion-minus-
                                   graph-only AUPRC delta, S1/S2 panels. The
                                   main-paper figure: the finding is
                                   distributional ("diffuse, ~70% positive, no
                                   concentration"), so a distribution plot is
                                   the right visualization, not a labeled bar
                                   chart of all 90 types.
  figures/per_type_scatter.pdf -- per-type fusion AUPRC vs. graph-only AUPRC,
                                   S1/S2 panels, colored by mechanism bucket,
                                   with the y=x diagonal. Direct visual
                                   complement to the histogram.
  figures/per_type_full.pdf    -- the complete 62+28 labeled bar chart, for
                                   supplementary material (unchanged role).

per_type_hist.pdf and per_type_scatter.pdf replace figures/per_type_compact.pdf
as the main-paper figure: a labeled bar chart of 30-62 types per panel is the
wrong chart for a claim about a distribution's shape, and is also the figure
that kept overflowing a single IEEE column.
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("per_type_figures")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
CSV_PATH = os.path.join(RESULTS_DIR, "per_type_drugbank.csv")

SPLITS = ["S1", "S2"]
LOW_SUPPORT_LABEL = "low test support (n<20)"  # threshold baked into the precomputed `low_support` column
COLUMN_WIDTH = 3.4  # inches, single IEEE column

BUCKET_COLORS = {
    "PK (ADME)": "#0072B2",
    "PD (named activity)": "#E69F00",
    "Risk/severity (unspecified mechanism)": "#009E73",
    "Therapeutic efficacy (unspecified mechanism)": "#CC79A7",
}


def split_stats(df: pd.DataFrame, split_name: str) -> dict:
    sub = df[df["split"] == split_name]
    return {
        "n": len(sub),
        "pct_positive": float((sub["delta_auprc_mean"] > 0).mean() * 100),
        "mean_delta": float(sub["delta_auprc_mean"].mean()),
    }


def _short_label(type_id: int, label: str, max_len: int) -> str:
    """Every template here starts with boilerplate ("#Drug1 may ...", "The
    metabolism of #Drug2 ...") identical across most of the 86 types, so plain
    character truncation only loses information once max_len clears that
    shared prefix (~25-35 chars). Used by the full (supplementary) figure
    only, which has enough width for that to hold."""
    text = label if len(label) <= max_len else label[:max_len - 1] + "…"
    return f"[{type_id}] {text}"


def _bucket_legend_handles(include_low_support: bool = False):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BUCKET_COLORS.values()]
    labels = list(BUCKET_COLORS.keys())
    if include_low_support:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="dimgray", hatch="///"))
        labels.append(LOW_SUPPORT_LABEL)
    return handles, labels


# ---------------------------------------------------------------------------
# Main-paper figure: distribution of the per-type delta
# ---------------------------------------------------------------------------

def plot_hist(df: pd.DataFrame, out_path: str) -> dict:
    all_deltas = df["delta_auprc_mean"].values
    bin_edges = np.linspace(all_deltas.min(), all_deltas.max(), 15)
    stats = {s: split_stats(df, s) for s in SPLITS}

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.3), sharex=True)
    for ax, split_name in zip(axes, SPLITS):
        sub = df[df["split"] == split_name]
        ax.hist(sub["delta_auprc_mean"], bins=bin_edges, color="#5B5B5B", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        st = stats[split_name]
        ax.set_title(f"DrugBank {split_name}: {st['pct_positive']:.0f}% positive, "
                     f"mean {st['mean_delta']:+.2f}, n={st['n']}", fontsize=8.5)
        ax.set_ylabel("Number of types", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel("Δ AUPRC, Fusion (GIN+text) − Graph-only (GIN), per type", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")
    return stats


# ---------------------------------------------------------------------------
# Direct comparison: per-type fusion AUPRC vs. graph-only AUPRC
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 6.8))
    for ax, split_name in zip(axes, SPLITS):
        sub = df[df["split"] == split_name]
        colors = [BUCKET_COLORS[b] for b in sub["bucket"]]
        ax.scatter(sub["graphonly_auprc_mean"], sub["fusion_auprc_mean"], c=colors, s=16,
                   edgecolor="black", linewidth=0.3, zorder=3)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.text(0.04, 0.96, "fusion wins", transform=ax.transAxes, fontsize=6.5,
                color="gray", va="top", ha="left", style="italic")
        ax.text(0.96, 0.04, "graph-only wins", transform=ax.transAxes, fontsize=6.5,
                color="gray", va="bottom", ha="right", style="italic")
        n = len(sub)
        ax.set_title(f"DrugBank {split_name}: per-type AUPRC (n={n})", fontsize=8.5)
        ax.set_ylabel("Fusion (GIN+text) AUPRC", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel("Graph-only (GIN) AUPRC", fontsize=8)
    handles, labels = _bucket_legend_handles()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5,
               frameon=False, labelspacing=0.3, handletextpad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Supplementary figure: the complete labeled bar chart (unchanged role)
# ---------------------------------------------------------------------------

def _draw_full_panel(ax, rows: pd.DataFrame, title: str):
    y_positions = np.arange(len(rows))
    colors = [BUCKET_COLORS[b] for b in rows["bucket"]]
    bars = ax.barh(y_positions, rows["delta_auprc_mean"], xerr=rows["delta_auprc_std"], color=colors,
                    error_kw=dict(elinewidth=0.6, ecolor="black", capsize=1.2), height=0.75)
    for bar, low in zip(bars, rows["low_support"]):
        if low:
            bar.set_hatch("///")
            bar.set_edgecolor("dimgray")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short_label(tid, lbl, 62) for tid, lbl in zip(rows["type_id"], rows["label"])],
                        fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Δ AUPRC, Fusion (GIN+text) − Graph-only (GIN), mean ± std over 3 seeds", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8)


def plot_full(df: pd.DataFrame, out_path: str) -> None:
    counts = {s: len(df[df["split"] == s]) for s in SPLITS}
    height_ratios = [counts[s] for s in SPLITS]
    fig_height = sum(0.24 * h for h in height_ratios) + 3.4
    fig, axes = plt.subplots(2, 1, figsize=(10, fig_height), gridspec_kw={"height_ratios": height_ratios})

    for ax, split_name in zip(axes, SPLITS):
        rows = df[df["split"] == split_name].sort_values("delta_auprc_mean", ascending=True).reset_index(drop=True)
        title = f"DrugBank {split_name}: per-type fusion-minus-graph-only AUPRC ({counts[split_name]} evaluable types)"
        _draw_full_panel(ax, rows, title)

    handles, labels = _bucket_legend_handles(include_low_support=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"missing {CSV_PATH} -- run scripts/09_per_type_drugbank_breakdown.py first "
                          f"(this script only re-plots, it does not compute per-type metrics)")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    hist_stats = plot_hist(df, os.path.join(FIGURES_DIR, "per_type_hist.pdf"))
    plot_scatter(df, os.path.join(FIGURES_DIR, "per_type_scatter.pdf"))
    plot_full(df, os.path.join(FIGURES_DIR, "per_type_full.pdf"))

    print("\n=== per-panel summary stats (figures/per_type_hist.pdf) ===")
    for split_name, st in hist_stats.items():
        print(f"{split_name}: {st['pct_positive']:.0f}% positive, mean {st['mean_delta']:+.3f}, n={st['n']}")


if __name__ == "__main__":
    main()
