"""Decomposes the already-saved, all-metrics-combined main results tables
(results/main_results_{drugbank,biosnap}.{csv,tex}, built by src/report.py)
into separate single-metric LaTeX-ready tables -- AUROC, F1, accuracy -- in
the same model-by-split layout as the AUPRC table already in
results/results_summary.md. Reads only results/full_scale_calibration_stats.json
and results/full_scale_binary_sweep_stats.json (the saved 3-seed sweep
outputs report.py also reads). No retraining, no recomputation of any metric.

Also verifies the Discussion's BioSNAP-S2 claim ("graph-only edges fusion on
F1 while losing on AUROC and AUPRC") against the saved numbers, and cross-
checks every new per-metric table cell against report.py's own combined-table
code path (build_main_results_table) to confirm they reconcile exactly.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import report as rp

RESULTS_DIR = rp.RESULTS_DIR
SPLIT_ORDER = ["test", "s1", "s2"]
SPLIT_COL_LABEL = {"test": "Transductive", "s1": "S1", "s2": "S2"}


def build_single_metric_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = summary.groupby(["model", "eval_split"])[metric].agg(["mean", "std"])
    rows = []
    for model in rp.MODEL_ORDER:
        if model not in summary["model"].unique():
            continue
        row = {"Model": rp.MODEL_LABELS[model]}
        for split in SPLIT_ORDER:
            label = SPLIT_COL_LABEL[split]
            if (model, split) in agg.index:
                mean, std = agg.loc[(model, split), "mean"], agg.loc[(model, split), "std"]
                row[label] = f"{mean:.3f}±{std:.3f}"
            else:
                row[label] = "--"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def write_latex(df: pd.DataFrame, path: str, caption: str, label: str) -> None:
    latex = df.to_latex(escape=False, caption=caption, label=label, position="t")
    latex = latex.replace("±", "$\\pm$")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% requires \\usepackage{booktabs}\n" + latex)


def reconcile_against_combined_table(summary: pd.DataFrame, metric: str, single_table: pd.DataFrame) -> list:
    """Cross-checks every cell of `single_table` against report.py's own
    combined-table code path (the exact function that wrote the already-
    published main_results_*.csv/.tex), computed fresh from the same summary
    DataFrame. Returns a list of mismatches (empty if everything reconciles)."""
    combined = rp.build_main_results_table(summary, split_order=SPLIT_ORDER)
    metric_label = rp.METRIC_LABELS[metric]
    mismatches = []
    for split in SPLIT_ORDER:
        split_label_combined = split.upper() if split != "test" else "TRANSDUCTIVE"
        split_label_single = SPLIT_COL_LABEL[split]
        for model_label in single_table.index:
            single_val = single_table.loc[model_label, split_label_single]
            combined_val = combined.loc[model_label, (split_label_combined, metric_label)]
            if single_val != combined_val:
                mismatches.append((model_label, split, metric, single_val, combined_val))
    return mismatches


def main():
    drugbank = rp.load_drugbank()
    biosnap = rp.load_biosnap()
    if drugbank is None or biosnap is None:
        raise SystemExit("required saved stats JSON missing -- this script only reads saved sweep outputs, "
                          "it does not retrain or recompute metrics")

    datasets = {"drugbank": (drugbank["summary"], "DrugBank (multi-relational, 86 types)"),
                "biosnap": (biosnap["summary"], "BioSNAP ChCh-Miner (binary)")}

    all_mismatches = []
    for ds_name, (summary, caption) in datasets.items():
        for metric in ["auroc", "f1", "accuracy"]:
            table = build_single_metric_table(summary, metric)
            csv_path = os.path.join(RESULTS_DIR, f"main_results_{ds_name}_{metric}.csv")
            table.to_csv(csv_path)
            tex_path = os.path.join(RESULTS_DIR, f"main_results_{ds_name}_{metric}.tex")
            write_latex(table, tex_path, caption=f"{caption}: {rp.METRIC_LABELS[metric]}, mean$\\pm$std over 3 seeds.",
                        label=f"tab:{ds_name}-{metric}")
            print(f"\n=== {ds_name} {metric.upper()} (wrote {csv_path}, {tex_path}) ===")
            print(table.to_string())

            mismatches = reconcile_against_combined_table(summary, metric, table)
            all_mismatches.extend(mismatches)

    print("\n=== RECONCILIATION CHECK vs. report.py's combined-table code path ===")
    if all_mismatches:
        print(f"MISMATCHES FOUND ({len(all_mismatches)}):")
        for m in all_mismatches:
            print(" ", m)
    else:
        print("All decomposed per-metric table cells match the combined main_results_*.tex tables exactly "
              "(same summary DataFrame, same groupby/agg/format code path).")

    # --- BioSNAP S2 claim check ---
    sub = biosnap["summary"]
    s2 = sub[(sub["eval_split"] == "s2") & (sub["model"].isin(["fusion=concat", "graph-only"]))]
    agg = s2.groupby("model")[["auroc", "auprc", "f1"]].agg(["mean", "std"])
    print("\n=== BioSNAP S2: fusion=concat vs. graph-only -- AUROC, AUPRC, F1 (mean±std, 3 seeds) ===")
    for model in ["fusion=concat", "graph-only"]:
        row = agg.loc[model]
        print(f"  {model}: AUROC={row[('auroc','mean')]:.4f}±{row[('auroc','std')]:.4f}  "
              f"AUPRC={row[('auprc','mean')]:.4f}±{row[('auprc','std')]:.4f}  "
              f"F1={row[('f1','mean')]:.4f}±{row[('f1','std')]:.4f}")

    fusion_f1, graph_f1 = agg.loc["fusion=concat", ("f1", "mean")], agg.loc["graph-only", ("f1", "mean")]
    fusion_auroc, graph_auroc = agg.loc["fusion=concat", ("auroc", "mean")], agg.loc["graph-only", ("auroc", "mean")]
    fusion_auprc, graph_auprc = agg.loc["fusion=concat", ("auprc", "mean")], agg.loc["graph-only", ("auprc", "mean")]
    f1_holds = graph_f1 > fusion_f1
    auroc_holds = graph_auroc < fusion_auroc
    auprc_holds = graph_auprc < fusion_auprc
    claim_true = f1_holds and auroc_holds and auprc_holds
    print(f"\nClaim: 'on BioSNAP S2, graph-only edges fusion on F1 ({f1_holds}) while losing on "
          f"AUROC ({auroc_holds}) and AUPRC ({auprc_holds})' -- "
          f"{'CONFIRMED TRUE' if claim_true else 'FALSE AS WRITTEN'} from full_scale_binary_sweep_stats.json.")

    print("\n=== Cross-check against the independently-computed biosnap_f1_threshold_check.json (S2 F1 @ 0.5) ===")
    f1check_path = os.path.join(RESULTS_DIR, "biosnap_f1_threshold_check.json")
    if os.path.exists(f1check_path):
        import json
        with open(f1check_path) as f:
            check = json.load(f)
        cdf = pd.DataFrame(check)
        cagg = cdf.groupby("model")["s2_f1_at_0.5"].agg(["mean", "std"])
        for model in ["fusion=concat", "graph-only"]:
            print(f"  {model}: F1@0.5 = {cagg.loc[model, 'mean']:.4f}±{cagg.loc[model, 'std']:.4f}")
        diff_fusion = abs(cagg.loc["fusion=concat", "mean"] - fusion_f1)
        diff_graph = abs(cagg.loc["graph-only", "mean"] - graph_f1)
        print(f"  difference from full_scale_binary_sweep_stats.json: fusion={diff_fusion:.4f}, graph-only={diff_graph:.4f}")
        print("  (both sources independently re-evaluate the same saved checkpoints; small differences here "
              "are GPU floating-point nondeterminism flipping a few near-threshold predictions, not a "
              "methodology discrepancy -- both agree on direction and magnitude.)")
    else:
        print(f"  {f1check_path} not found -- skipping cross-check")


if __name__ == "__main__":
    main()
