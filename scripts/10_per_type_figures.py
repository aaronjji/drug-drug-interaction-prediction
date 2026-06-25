"""Plotting-only regeneration of the per-type cold-start delta figure for the
paper. Reads results/per_type_drugbank.csv exactly as written by
09_per_type_drugbank_breakdown.py -- no recomputation, no checkpoint loading,
no retraining. Produces:

  figures/per_type_compact.pdf  -- top-15/bottom-15 per split (or all types if
                                    a split has <=30 evaluable), sized for a
                                    single IEEE column. For the main paper.
  figures/per_type_full.pdf     -- the complete 62+28 version, at a larger
                                    figure height/font for legibility on a
                                    full page. For supplementary material.

Same color coding (PK/PD/Risk/Efficacy bucket), low-support hatching (n<20,
per the precomputed `low_support` column), error bars, and x-axis label as
the original figures/per_type_delta.pdf.
"""
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
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
TOP_N = 15
SHOW_ALL_THRESHOLD = 30  # show every type instead of top/bottom-N if n_evaluable <= this
LOW_SUPPORT_LABEL = "low test support (n<20)"  # threshold baked into the precomputed `low_support` column

BUCKET_COLORS = {
    "PK (ADME)": "#0072B2",
    "PD (named activity)": "#E69F00",
    "Risk/severity (unspecified mechanism)": "#009E73",
    "Therapeutic efficacy (unspecified mechanism)": "#CC79A7",
}


def _short_label(type_id: int, label: str, max_len: int) -> str:
    """Full-text mode: every template here starts with boilerplate
    ("#Drug1 may ...", "The metabolism of #Drug2 ...") that's identical across
    most of the 86 types, so plain character truncation only loses information
    once max_len is large enough to clear that shared prefix (~25-35 chars)."""
    text = label if len(label) <= max_len else label[:max_len - 1] + "…"
    return f"[{type_id}] {text}"


def _core_phrase(label: str):
    """Strips the boilerplate ('#Drug1 may increase the ... activities of
    #Drug2.', 'The serum concentration of #Drug2 can be ... ', etc.) down to
    the direction + the actual mechanism/effect term, since at single-column
    width that shared boilerplate is what a naive character truncation keeps
    and the distinguishing word is what it cuts. Returns None if no known
    template pattern matches (caller falls back to plain truncation)."""
    tl = label.lower()
    direction = "+" if "increase" in tl else ("-" if ("decrease" in tl or "reduced" in tl) else "")

    m = re.search(r"the (.+?) activities of", label, re.IGNORECASE)
    if m:
        return f"{direction} {m.group(1).strip()}"
    # checked before the generic 'serum concentration' branch below: types 17/66
    # describe a change in absorption that secondarily affects concentration --
    # 'absorption' is the more specific, distinguishing term for those two
    if "absorption" in tl:
        return f"{direction} absorption"
    if "serum concentration of the active metabolites" in tl:
        return f"{direction} active-metabolite serum conc."
    if "serum concentration" in tl:
        return f"{direction} serum concentration"
    if "bioavailability" in tl:
        return f"{direction} bioavailability"
    if "metabolism" in tl:
        return f"{direction} metabolism"
    if "excretion rate" in tl:
        return f"{direction} excretion rate"
    if "therapeutic efficacy" in tl:
        return f"{direction} therapeutic efficacy"
    m2 = re.search(r"risk or severity of (.+?) can be", label, re.IGNORECASE)
    if m2:
        return f"{direction} risk: {m2.group(1).strip()}"
    return None


def _short_label_compact(type_id: int, label: str, max_len: int) -> str:
    """Narrow-column mode: extracts the mechanism/effect term instead of
    truncating raw characters, so the distinguishing word survives instead of
    being cut along with the shared '#Drug1 may increase the ...' boilerplate."""
    core = _core_phrase(label)
    text = core if core is not None else label
    if len(text) > max_len:
        text = text[:max_len - 1] + "…"
    return f"[{type_id}] {text}"


def _draw_panel(ax, rows: pd.DataFrame, gap_after: int, gap_text: str, label_max_len: int,
                 tick_fontsize: float, title: str, title_fontsize: float, xlabel_fontsize: float,
                 compact_labels: bool = False):
    """rows must already be in bottom-to-top plotting order (ascending delta
    within each contiguous block: bottom-N ascending, then top-N ascending).
    gap_after=index (0-based, into `rows`) of the last row of the bottom block,
    after which a blank omitted-types row is inserted, or None for no gap.

    Bars and y-tick labels are built from the SAME single pass over `rows` (one
    list of (position, bar-or-None, label) triples) so a bar at position k is
    guaranteed to carry the label for that exact row -- no separate index
    bookkeeping between bars and labels that could drift apart."""
    label_fn = _short_label_compact if compact_labels else _short_label
    entries = []  # (y_position, row-or-None, label_text)
    pos = 0
    for i in range(len(rows)):
        r = rows.iloc[i]
        entries.append((pos, r, label_fn(r["type_id"], r["label"], label_max_len)))
        pos += 1
        if gap_after is not None and i == gap_after:
            entries.append((pos, None, gap_text))
            pos += 1
    n_total = pos

    bar_entries = [(y, r) for y, r, _ in entries if r is not None]
    y_positions = [y for y, _ in bar_entries]
    colors = [BUCKET_COLORS[r["bucket"]] for _, r in bar_entries]
    deltas = [r["delta_auprc_mean"] for _, r in bar_entries]
    stds = [r["delta_auprc_std"] for _, r in bar_entries]
    bars = ax.barh(y_positions, deltas, xerr=stds, color=colors,
                    error_kw=dict(elinewidth=0.6, ecolor="black", capsize=1.2), height=0.75)
    for bar, (_, r) in zip(bars, bar_entries):
        if r["low_support"]:
            bar.set_hatch("///")
            bar.set_edgecolor("dimgray")

    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks([y for y, _, _ in entries])
    ax.set_yticklabels([lbl for _, _, lbl in entries], fontsize=tick_fontsize)
    for tick, (_, r, lbl) in zip(ax.get_yticklabels(), entries):
        if r is None:
            tick.set_style("italic")
            tick.set_color("gray")
            tick.set_fontsize(max(tick_fontsize - 0.5, 5.5))
    ax.set_ylim(-0.7, n_total - 0.3)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Δ AUPRC (fusion=concat − graph-only), mean ± std over 3 seeds", fontsize=xlabel_fontsize)
    ax.tick_params(axis="x", labelsize=max(tick_fontsize - 0.5, 6))


def _legend_handles_labels():
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BUCKET_COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="dimgray", hatch="///"))
    labels = list(BUCKET_COLORS.keys()) + [LOW_SUPPORT_LABEL]
    return handles, labels


def build_compact_rows(df: pd.DataFrame, split_name: str):
    """Returns (rows_in_plot_order, gap_after_index, gap_text, n_shown, n_total)."""
    sub = df[df["split"] == split_name]
    n_total = len(sub)
    if n_total <= SHOW_ALL_THRESHOLD:
        rows = sub.sort_values("delta_auprc_mean", ascending=True).reset_index(drop=True)
        return rows, None, None, n_total, n_total

    bottom = sub.sort_values("delta_auprc_mean", ascending=True).head(TOP_N)
    top = sub.sort_values("delta_auprc_mean", ascending=False).head(TOP_N).sort_values(
        "delta_auprc_mean", ascending=True)
    rows = pd.concat([bottom, top], ignore_index=True)
    n_omitted = n_total - len(rows)
    gap_text = f"⋯ {n_omitted} types omitted ⋯"
    return rows, len(bottom) - 1, gap_text, len(rows), n_total


def plot_compact(df: pd.DataFrame, out_path: str) -> dict:
    panel_data = {}
    for split_name in SPLITS:
        panel_data[split_name] = build_compact_rows(df, split_name)

    height_ratios = [len(panel_data[s][0]) + (1 if panel_data[s][1] is not None else 0) for s in SPLITS]
    fig_height = sum(0.225 * h for h in height_ratios) + 2.1
    fig, axes = plt.subplots(2, 1, figsize=(3.45, fig_height), gridspec_kw={"height_ratios": height_ratios})

    for ax, split_name in zip(axes, SPLITS):
        rows, gap_after, gap_text, n_shown, n_total = panel_data[split_name]
        if gap_after is not None:
            title = f"DrugBank {split_name}: top-{TOP_N} / bottom-{TOP_N} of {n_total} evaluable types"
        else:
            title = f"DrugBank {split_name}: all {n_total} evaluable types"
        _draw_panel(ax, rows, gap_after, gap_text, label_max_len=34, tick_fontsize=8.0,
                    title=title, title_fontsize=8.5, xlabel_fontsize=8.0, compact_labels=True)

    handles, labels = _legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.3, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")
    return {s: panel_data[s][0] for s in SPLITS}


def plot_full(df: pd.DataFrame, out_path: str) -> None:
    counts = {s: len(df[df["split"] == s]) for s in SPLITS}
    height_ratios = [counts[s] for s in SPLITS]
    fig_height = sum(0.24 * h for h in height_ratios) + 3.4
    fig, axes = plt.subplots(2, 1, figsize=(10, fig_height), gridspec_kw={"height_ratios": height_ratios})

    for ax, split_name in zip(axes, SPLITS):
        rows = df[df["split"] == split_name].sort_values("delta_auprc_mean", ascending=True).reset_index(drop=True)
        title = f"DrugBank {split_name}: per-type fusion-minus-graph-only AUPRC ({counts[split_name]} evaluable types)"
        _draw_panel(ax, rows, None, None, label_max_len=62, tick_fontsize=8.5,
                    title=title, title_fontsize=11, xlabel_fontsize=9.5)

    handles, labels = _legend_handles_labels()
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

    compact_path = os.path.join(FIGURES_DIR, "per_type_compact.pdf")
    shown = plot_compact(df, compact_path)

    full_path = os.path.join(FIGURES_DIR, "per_type_full.pdf")
    plot_full(df, full_path)

    for split_name in SPLITS:
        rows = shown[split_name].sort_values("delta_auprc_mean", ascending=False)
        print(f"\n=== {split_name}: types shown in compact figure (n={len(rows)}) ===")
        print(rows[["type_id", "label", "bucket", "n_test", "low_support", "delta_auprc_mean", "delta_auprc_std"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
