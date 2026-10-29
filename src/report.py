"""Builds paper-ready tables and figures purely from already-saved results
(results/full_scale_calibration_stats.json, full_scale_binary_sweep_stats.json,
biosnap_f1_threshold_check.json). Reads and formats only -- never retrains,
never calls train.py. If a number a paper table would want isn't in the saved
outputs, this prints a flag rather than fabricating or rerunning anything.
"""
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("report")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

MODEL_ORDER = ["fusion=concat", "graph-only", "text-only", "fingerprint-mlp"]
MODEL_LABELS = {"fusion=concat": "Fusion (GIN+text)", "graph-only": "Graph-only (GIN)",
                "text-only": "Text-only", "fingerprint-mlp": "Fingerprint-MLP"}
ABLATION_ORDER = ["fusion=concat", "graph-only", "text-only"]
METRICS = ["auroc", "auprc", "f1", "accuracy"]
METRIC_LABELS = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1", "accuracy": "Acc."}
# Okabe-Ito colorblind-safe palette
COLORS = {"fusion=concat": "#0072B2", "graph-only": "#E69F00",
          "text-only": "#009E73", "fingerprint-mlp": "#D55E00"}


def _missing(path):
    full = os.path.join(RESULTS_DIR, path)
    if not os.path.exists(full):
        log.warning(f"MISSING required input: {full} -- cannot build the outputs that depend on it")
        return True
    return False


def load_drugbank():
    if _missing("full_scale_calibration_stats.json"):
        return None
    with open(os.path.join(RESULTS_DIR, "full_scale_calibration_stats.json")) as f:
        d = json.load(f)
    summary = pd.DataFrame(d["summary_table"])
    gap = pd.DataFrame(d["transductive_inductive_gap"])
    return {"summary": summary, "gap": gap, "raw": d}


def load_biosnap():
    if _missing("full_scale_binary_sweep_stats.json"):
        return None
    with open(os.path.join(RESULTS_DIR, "full_scale_binary_sweep_stats.json")) as f:
        d = json.load(f)
    summary = pd.DataFrame(d["summary_table"])
    gap = pd.DataFrame(d["transductive_inductive_gap"])
    return {"summary": summary, "gap": gap, "raw": d}


def load_f1_threshold_check():
    if _missing("biosnap_f1_threshold_check.json"):
        return None
    with open(os.path.join(RESULTS_DIR, "biosnap_f1_threshold_check.json")) as f:
        return pd.DataFrame(json.load(f))


# ---------------------------------------------------------------------------
# Main results table: rows=model, columns=(split, metric), cells="mean+-std"
# ---------------------------------------------------------------------------

def build_main_results_table(summary: pd.DataFrame, split_order):
    agg = summary.groupby(["model", "eval_split"])[METRICS].agg(["mean", "std"])
    rows = []
    for model in MODEL_ORDER:
        if model not in summary["model"].unique():
            continue
        row = {"Model": MODEL_LABELS[model]}
        for split in split_order:
            for metric in METRICS:
                col = f"{split}_{metric}"
                if (model, split) in agg.index:
                    mean, std = agg.loc[(model, split), (metric, "mean")], agg.loc[(model, split), (metric, "std")]
                    row[col] = f"{mean:.3f}±{std:.3f}"
                else:
                    row[col] = "--"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    df.columns = pd.MultiIndex.from_tuples(
        [(s.upper() if s != "test" else "TRANSDUCTIVE", METRIC_LABELS[c.split("_")[-1]])
         for s in split_order for c in [f"{s}_{m}" for m in METRICS]])
    return df


def write_latex_table(df: pd.DataFrame, path: str, caption: str, label: str):
    latex = df.to_latex(multicolumn=True, multicolumn_format="c", escape=False,
                         caption=caption, label=label, position="t")
    latex = latex.replace("±", "$\\pm$")  # raw unicode needs fontspec/unicode-math; \pm is universally safe
    header = "% requires \\usepackage{booktabs} \\usepackage{multirow}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + latex)
    log.info(f"wrote {path}")


# ---------------------------------------------------------------------------
# Gap table: rows=(dataset, model), columns=(gap_to, metric), AUPRC+AUROC only
# ---------------------------------------------------------------------------

def build_gap_table(gap_drugbank: pd.DataFrame, gap_biosnap: pd.DataFrame):
    gap_metrics = ["auroc", "auprc"]
    rows = []
    for dataset_name, gap_df in [("DrugBank", gap_drugbank), ("BioSNAP", gap_biosnap)]:
        agg = gap_df.groupby(["model", "gap"])[gap_metrics].agg(["mean", "std"])
        for model in MODEL_ORDER:
            if model not in gap_df["model"].unique():
                continue
            row = {"Dataset": dataset_name, "Model": MODEL_LABELS[model]}
            for gap_to, gap_label in [("transductive_minus_s1", "S1"), ("transductive_minus_s2", "S2")]:
                for metric in gap_metrics:
                    col = f"{gap_label}_{metric}"
                    if (model, gap_to) in agg.index:
                        mean, std = agg.loc[(model, gap_to), (metric, "mean")], agg.loc[(model, gap_to), (metric, "std")]
                        row[col] = f"{mean:.3f}±{std:.3f}"
                    else:
                        row[col] = "--"
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["Dataset", "Model"])
    df.columns = pd.MultiIndex.from_tuples(
        [(g, METRIC_LABELS[c.split("_")[-1]]) for g in ["S1", "S2"] for c in [f"{g}_{m}" for m in gap_metrics]])
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _grouped_bar_panel(ax, summary, models_here, splits, split_labels):
    agg = summary.groupby(["model", "eval_split"])["auprc"].agg(["mean", "std"])
    x = np.arange(len(splits))
    width = 0.8 / len(models_here)
    for i, model in enumerate(models_here):
        means = [agg.loc[(model, s), "mean"] if (model, s) in agg.index else np.nan for s in splits]
        stds = [agg.loc[(model, s), "std"] if (model, s) in agg.index else 0 for s in splits]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=2,
               label=MODEL_LABELS[model], color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.05)


def plot_inductive_gap(drugbank_summary, biosnap_summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    splits, split_labels = ["test", "s1", "s2"], ["Transductive", "S1", "S2"]
    for ax, name, summary in zip(axes, ["DrugBank", "BioSNAP"], [drugbank_summary, biosnap_summary]):
        models_here = [m for m in MODEL_ORDER if m in summary["model"].unique()]
        _grouped_bar_panel(ax, summary, models_here, splits, split_labels)
        ax.set_title(name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Transductive vs. cold-start (S1/S2) AUPRC, mean ± std over 3 seeds", y=1.18, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")


def plot_ablation(drugbank_summary, biosnap_summary, out_path):
    # drawn at print size (paper includes it at \linewidth = 5.5in); the caption carries the title
    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9))
        splits, split_labels = ["s1", "s2"], ["S1", "S2"]
        for ax, name, summary in zip(axes, ["DrugBank", "BioSNAP"], [drugbank_summary, biosnap_summary]):
            models_here = [m for m in ABLATION_ORDER if m in summary["model"].unique()]
            _grouped_bar_panel(ax, summary, models_here, splits, split_labels)
            ax.set_title(name)
            ax.set_ylim(0, 1.0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    log.info(f"wrote {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    missing_flags = []

    drugbank = load_drugbank()
    biosnap = load_biosnap()
    f1_check = load_f1_threshold_check()

    if drugbank is None or biosnap is None:
        raise SystemExit("required results JSON missing -- run the sweeps first, this script does not retrain")

    # --- main results tables ---
    db_table = build_main_results_table(drugbank["summary"], split_order=["test", "s1", "s2"])
    db_table.to_csv(os.path.join(RESULTS_DIR, "main_results_drugbank.csv"))
    write_latex_table(db_table, os.path.join(RESULTS_DIR, "main_results_drugbank.tex"),
                       caption="DrugBank (multi-relational, 86 types): mean$\\pm$std over 3 seeds.",
                       label="tab:main-drugbank")

    bs_table = build_main_results_table(biosnap["summary"], split_order=["test", "s1", "s2"])
    bs_table.to_csv(os.path.join(RESULTS_DIR, "main_results_biosnap.csv"))
    write_latex_table(bs_table, os.path.join(RESULTS_DIR, "main_results_biosnap.tex"),
                       caption="BioSNAP ChCh-Miner (binary): mean$\\pm$std over 3 seeds.",
                       label="tab:main-biosnap")

    # --- gap table ---
    gap_table = build_gap_table(drugbank["gap"], biosnap["gap"])
    gap_table.to_csv(os.path.join(RESULTS_DIR, "gap_table.csv"))
    write_latex_table(gap_table, os.path.join(RESULTS_DIR, "gap_table.tex"),
                       caption="Transductive-minus-cold-start gap (AUROC, AUPRC), mean$\\pm$std over 3 seeds. Smaller is better (degrades less).",
                       label="tab:gap")

    # --- figures ---
    inductive_gap_path = os.path.join(FIGURES_DIR, "inductive_gap.pdf")
    plot_inductive_gap(drugbank["summary"], biosnap["summary"], inductive_gap_path)
    ablation_path = os.path.join(FIGURES_DIR, "ablation.pdf")
    plot_ablation(drugbank["summary"], biosnap["summary"], ablation_path)

    # --- flag anything a paper table would want that we don't have ---
    if "n_evaluable_classes_used" not in biosnap["summary"].columns or biosnap["summary"]["n_evaluable_classes_used"].isna().all():
        missing_flags.append("BioSNAP: no evaluable-class count column (not applicable -- binary task, no macro-averaging over classes)")
    n_seeds = drugbank["summary"]["seed"].nunique()
    if n_seeds < 5:
        missing_flags.append(f"Only {n_seeds} seeds run -- mean+-std with n={n_seeds} is informative but too "
                              f"few for a formal significance test (e.g. paired t-test); no p-values/CIs are computed.")
    missing_flags.append("No per-interaction-type (per-class) breakdown is saved for DrugBank -- only the "
                          "macro-averaged metric over evaluable classes. A reviewer asking 'which interaction "
                          "types does fusion help most on' can't be answered from saved outputs alone.")
    missing_flags.append("No statistical significance test (e.g. paired bootstrap or t-test across seeds) "
                          "between fusion and the next-best cold-start model is computed or saved.")

    print("\n=== MAIN RESULTS: DrugBank ===")
    print(db_table.to_string())
    print("\n=== MAIN RESULTS: BioSNAP ===")
    print(bs_table.to_string())
    print("\n=== GAP TABLE (AUROC, AUPRC; transductive-minus-cold-start) ===")
    print(gap_table.to_string())
    print(f"\nfigures: {inductive_gap_path}, {ablation_path}")
    print("\n=== METRICS A PAPER TABLE MIGHT WANT THAT AREN'T IN SAVED OUTPUTS ===")
    for flag in missing_flags:
        print(f"- {flag}")

    with open(os.path.join(RESULTS_DIR, "report_missing_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(missing_flags) + "\n")


if __name__ == "__main__":
    main()
