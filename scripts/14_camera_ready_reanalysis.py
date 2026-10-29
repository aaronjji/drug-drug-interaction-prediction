"""Inference-only re-analysis of saved DrugBank checkpoints for the camera-ready
(no training). Reloads every finished checkpoint -- the original sweep's
fullscale_* runs and any finished cr_* runs from 13_camera_ready_experiments.py
-- and evaluates it on the persisted seed-0 splits:

  1. sanity: recomputed official metrics vs. the stored ones (must match);
  2. common class set [reviewer yfbP]: macro AUROC/AUPRC over the interaction
     types evaluable in ALL of transductive-test, S1 and S2, so the
     transductive-to-cold-start gap is no longer partly a change of class set;
  3. S1 by partner [yfbP, validation protocol]: inductive validation drugs
     never appear in a training pair, so an S1 pair whose non-test drug is a
     validation drug has two drugs unseen by gradient updates; S1 is split
     into train-partner and val-partner pairs;
  4. similarity strata [SM1A, analogue reliance]: each held-out drug's
     maximum ECFP4 Tanimoto similarity to any training drug; S1 pairs are
     binned into tertiles of the test drug's similarity, S2 pairs into halves
     of the mean of the two drugs' similarities.
Partitions (3, 4) are macro-averaged over the official evaluable classes of
that split that keep >= MIN_PARTITION_POSITIVES positives inside the
partition; every model is scored on the identical class set per partition.

Also writes the split-composition counts (both datasets) quoted in the paper.
Writes results/camera_ready_reanalysis.{json,md}.

--biosnap-redraw (run with PYTHONHASHSEED=0): the original BioSNAP sweep's
inductive negatives cannot be recreated (06's bucket sampler iterates Python
sets, whose order depends on the per-process hash seed), so the camera-ready
fingerprint+text BioSNAP runs were evaluated on the PYTHONHASHSEED=0 draw.
This re-scores the four original BioSNAP inductive checkpoints on that same
draw, so every BioSNAP S1/S2 number can be compared on identical pairs, and
reports how far each moves from its stored sweep value. Writes
results/biosnap_redraw_check.json.
"""
import argparse
import importlib.util
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import baselines as bl
import evaluate as ev
import experiment_utils as eu
import splits as sp
import train as tr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera_ready_reanalysis")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (0, 1, 2)
MIN_PARTITION_POSITIVES = 10
EVAL_BATCH = 512

# (label, checkpoint prefix, arch key understood by 13's Context.components / "fingerprint-mlp")
MODELS = [
    ("Fusion (GIN+text)", "fullscale_fusion=concat", "fusion=concat"),
    ("Graph-only (GIN)", "fullscale_graph-only", "graph-only"),
    ("Text-only", "fullscale_text-only", "text-only"),
    ("Fingerprint-MLP", "fullscale_fingerprint-mlp", "fingerprint-mlp"),
    ("Fingerprint+text", "cr_drugbank_fp-text", "fp-text"),
    ("GCN graph-only", "cr_drugbank_gcn-graph-only", "gcn-graph-only"),
    ("GCN fusion", "cr_drugbank_gcn-fusion=concat", "gcn-fusion=concat"),
    ("Fusion (gated)", "cr_drugbank_fusion=gated", "fusion=gated"),
    ("Fusion (cross-attention)", "cr_drugbank_fusion=cross_attention", "fusion=cross_attention"),
    ("Fusion, filtered text", "cr_drugbank_filtered-fusion=concat", "filtered-fusion=concat"),
    ("Text-only, filtered text", "cr_drugbank_filtered-text-only", "filtered-text-only"),
    ("Fusion, interaction wording removed", "cr_drugbank_generic-fusion=concat", "generic-fusion=concat"),
    ("Text-only, interaction wording removed", "cr_drugbank_generic-text-only", "generic-text-only"),
]


def load_module_from_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(ROOT, "scripts", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tanimoto_max(query_ids, ref_ids, fp_cache):
    ref = torch.tensor(np.stack([fp_cache[i] for i in ref_ids]), dtype=torch.float)
    q = torch.tensor(np.stack([fp_cache[i] for i in query_ids]), dtype=torch.float)
    inter = q @ ref.T
    union = q.sum(1, keepdim=True) + ref.sum(1).unsqueeze(0) - inter
    sim = inter / union.clamp(min=1)
    return dict(zip(query_ids, sim.max(1).values.numpy().astype(float)))


def composition(transductive, drug_split, inductive, id1, id2):
    d_train, d_val, d_test = drug_split["train"], drug_split["val"], drug_split["test"]
    val = inductive["val"]
    n_val_drugs = val[id1].isin(d_val).astype(int) + val[id2].isin(d_val).astype(int)
    s1 = inductive["s1"]
    partner = np.where(s1[id1].isin(d_test), s1[id2], s1[id1])
    out = {
        "n_drugs": {k: len(v) for k, v in drug_split.items()},
        "transductive_pairs": {n: int((transductive["split"] == n).sum()) for n in ("train", "val", "test")},
        "inductive_pairs": {n: len(df) for n, df in inductive.items()},
        "val_pairs_one_val_drug": int((n_val_drugs == 1).sum()),
        "val_pairs_two_val_drugs": int((n_val_drugs == 2).sum()),
        "s1_pairs_train_partner": int(np.isin(partner, list(d_train)).sum()),
        "s1_pairs_val_partner": int(np.isin(partner, list(d_val)).sum()),
    }
    covered = sum(out["inductive_pairs"].values())
    out["inductive_pairs_total"] = covered
    out["drug_sets_disjoint"] = not (d_train & d_val or d_train & d_test or d_val & d_test)
    return out


def macro(y, probs, classes):
    m = ev.compute_multiclass_metrics(np.asarray(y), probs, set(classes))
    return {"auroc": m["auroc"], "auprc": m["auprc"], "n_classes": m["n_evaluable_classes_used"]}


def partition_metrics(y, probs, mask, official_classes):
    y_part = np.asarray(y)[mask]
    classes = [c for c in official_classes if (y_part == c).sum() >= MIN_PARTITION_POSITIVES]
    if not classes:
        return None
    out = macro(y_part, probs[mask], classes)
    out["n_pairs"] = int(mask.sum())
    out["classes"] = classes
    return out


def summarize(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": len(arr)}


def biosnap_redraw_check(ctx, device):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run --biosnap-redraw with PYTHONHASHSEED=0 (the draw the camera-ready runs used)")
    with open(os.path.join(ROOT, "results", "full_scale_binary_sweep_stats.json")) as f:
        stored = pd.DataFrame(json.load(f)["summary_table"])
    m = ctx.model_cfg
    rows = []
    for arch in ("fusion=concat", "graph-only", "text-only", "fingerprint-mlp"):
        for seed in SEEDS:
            if arch == "fingerprint-mlp":
                model_factory, batch_fn = eu.make_fingerprint_components(
                    ctx.fp_cache, fp_dim=bl.FINGERPRINT_BITS, hidden_dim=m["hidden_dim"], head_hidden=m["head_hidden"],
                    dropout=m["dropout"], num_classes=None, binary=True)
            else:
                model_factory, batch_fn = ctx.components("biosnap", arch)
            model = model_factory().to(device)
            model.load_state_dict(torch.load(os.path.join(tr.CHECKPOINT_DIR, f"binary_{arch}_inductive_seed{seed}.pt"),
                                             weights_only=False))
            _, inductive = ctx.biosnap(seed)
            for split in ("s1", "s2"):
                metrics = tr.evaluate_split(model, inductive[split], "d1", "d2", "label", batch_fn, True, device, EVAL_BATCH)
                old = stored[(stored.model == arch) & (stored.regime == "inductive") & (stored.seed == seed)
                             & (stored.eval_split == split)].iloc[0]
                rows.append({"model": arch, "seed": seed, "split": split, **{k: metrics[k] for k in ev.METRIC_NAMES},
                             **{f"sweep_{k}": float(old[k]) for k in ev.METRIC_NAMES}})
                log.info(f"biosnap redraw {arch} seed={seed} {split}: auprc {old['auprc']:.4f} (sweep draw) -> "
                         f"{metrics['auprc']:.4f} (PYTHONHASHSEED=0 draw)")
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "split"])[["auprc", "sweep_auprc", "auroc", "sweep_auroc"]].agg(["mean", "std"])
    summary = {f"{a}|{s}": {f"{c0}_{c1}": float(v) for (c0, c1), v in row.items()} for (a, s), row in agg.iterrows()}
    out = {"rows": rows, "max_abs_auprc_shift": float((df["auprc"] - df["sweep_auprc"]).abs().max()),
           "summary": summary}
    with open(os.path.join(ROOT, "results", "biosnap_redraw_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"biosnap redraw: max |AUPRC shift| = {out['max_abs_auprc_shift']:.4f}; wrote results/biosnap_redraw_check.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="|-separated model labels to (re)evaluate (labels contain commas)")
    parser.add_argument("--biosnap-redraw", action="store_true", help="only run the BioSNAP negative-redraw check")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cr = load_module_from_script("13_camera_ready_experiments.py", "cr13")
    ctx = cr.Context(device, smoke=False)
    if args.biosnap_redraw:
        biosnap_redraw_check(ctx, device)
        return
    transductive, drug_split, inductive, evaluable = ctx.drugbank()
    trans_test = transductive[transductive["split"] == "test"].reset_index(drop=True)
    s1, s2 = inductive["s1"].reset_index(drop=True), inductive["s2"].reset_index(drop=True)

    comp = {"drugbank": composition(transductive, drug_split, inductive, "ID1", "ID2")}
    chch_t, chch_split, chch_ind = sp.load_splits(0, tag="chch")
    comp["biosnap_positives"] = composition(chch_t, chch_split, chch_ind, "d1", "d2")
    log.info(json.dumps(comp, indent=1))

    common = sorted(evaluable["transductive_test"] & evaluable["inductive_s1"] & evaluable["inductive_s2"])
    log.info(f"common evaluable class set: {len(common)} types {common}")

    d_train, d_val, d_test = drug_split["train"], drug_split["val"], drug_split["test"]
    s1_test_drug = np.where(s1["ID1"].isin(d_test), s1["ID1"], s1["ID2"])
    s1_partner = np.where(s1["ID1"].isin(d_test), s1["ID2"], s1["ID1"])
    s1_masks = {"train_partner": np.isin(s1_partner, list(d_train)),
                "val_partner": np.isin(s1_partner, list(d_val))}
    sim = tanimoto_max(sorted(d_test), sorted(d_train), ctx.fp_cache)
    s1_sim = np.array([sim[d] for d in s1_test_drug])
    s2_sim = np.array([(sim[a] + sim[b]) / 2 for a, b in zip(s2["ID1"], s2["ID2"])])
    q1, q2 = np.quantile(s1_sim, [1 / 3, 2 / 3])
    med2 = np.quantile(s2_sim, 0.5)
    sim_masks = {
        "s1": {"low": s1_sim <= q1, "mid": (s1_sim > q1) & (s1_sim <= q2), "high": s1_sim > q2},
        "s2": {"low": s2_sim <= med2, "high": s2_sim > med2},
    }
    sim_info = {"s1_tertile_cuts": [float(q1), float(q2)], "s2_median_cut": float(med2),
                "test_drug_max_sim_quartiles": [float(x) for x in np.quantile(list(sim.values()), [0.25, 0.5, 0.75])],
                "s1_bin_sizes": {k: int(v.sum()) for k, v in sim_masks["s1"].items()},
                "s2_bin_sizes": {k: int(v.sum()) for k, v in sim_masks["s2"].items()}}
    log.info(f"similarity strata: {sim_info}")

    with open(os.path.join(ROOT, "results", "full_scale_calibration_stats.json")) as f:
        stored_main = pd.DataFrame(json.load(f)["summary_table"])

    def stored_metrics(prefix, arch, regime, seed, split):
        if prefix.startswith("fullscale_"):
            row = stored_main[(stored_main["model"] == arch) & (stored_main["regime"] == regime)
                              & (stored_main["seed"] == seed) & (stored_main["eval_split"] == split)]
            return None if row.empty else {"auroc": float(row["auroc"].iloc[0]), "auprc": float(row["auprc"].iloc[0])}
        runs = sorted(f for f in os.listdir(tr.RUNS_DIR) if f.startswith(f"{prefix}_{regime}_seed{seed}_"))
        if not runs:
            return None
        with open(os.path.join(tr.RUNS_DIR, runs[-1])) as f:
            return json.load(f)["result"]["metrics_by_split"].get(split)

    def components_for(arch):
        if arch == "fingerprint-mlp":
            m = ctx.model_cfg
            return eu.make_fingerprint_components(ctx.fp_cache, fp_dim=bl.FINGERPRINT_BITS, hidden_dim=m["hidden_dim"],
                                                  head_hidden=m["head_hidden"], dropout=m["dropout"],
                                                  num_classes=86, binary=False)
        return ctx.components("drugbank", arch)

    def predict(prefix, arch, regime, seed, df):
        path = os.path.join(tr.CHECKPOINT_DIR, f"{prefix}_{regime}_seed{seed}.pt")
        if not os.path.exists(path):
            return None
        model_factory, batch_fn = components_for(arch)
        model = model_factory().to(device)
        model.load_state_dict(torch.load(path, weights_only=False))
        probs, labels = tr._predict_split(model, df, "ID1", "ID2", "Y", batch_fn, False, device, EVAL_BATCH)
        return probs, labels

    out_path = os.path.join(ROOT, "results", "camera_ready_reanalysis.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("per_run", {})
    only = set(args.only.split("|")) if args.only else None
    max_sanity_diff = 0.0
    for label, prefix, arch in MODELS:
        if only and label not in only:
            continue
        for seed in SEEDS:
            key = f"{label}|{seed}"
            rec = {}
            pt = predict(prefix, arch, "transductive", seed, trans_test)
            if pt is not None:
                probs, y = pt
                rec["T_official"] = macro(y, probs, evaluable["transductive_test"])
                rec["T_common"] = macro(y, probs, common)
                stored = stored_metrics(prefix, arch, "transductive", seed, "test")
                if stored:
                    max_sanity_diff = max(max_sanity_diff, abs(stored["auprc"] - rec["T_official"]["auprc"]))
            pi1 = predict(prefix, arch, "inductive", seed, s1)
            pi2 = predict(prefix, arch, "inductive", seed, s2)
            if pi1 is not None and pi2 is not None:
                (p1, y1), (p2, y2) = pi1, pi2
                rec["S1_official"] = macro(y1, p1, evaluable["inductive_s1"])
                rec["S2_official"] = macro(y2, p2, evaluable["inductive_s2"])
                rec["S1_common"] = macro(y1, p1, common)
                rec["S2_common"] = macro(y2, p2, common)
                for split, r in (("s1", rec["S1_official"]), ("s2", rec["S2_official"])):
                    stored = stored_metrics(prefix, arch, "inductive", seed, split)
                    if stored:
                        max_sanity_diff = max(max_sanity_diff, abs(stored["auprc"] - r["auprc"]))
                for part, mask in s1_masks.items():
                    rec[f"S1_{part}"] = partition_metrics(y1, p1, mask, sorted(evaluable["inductive_s1"]))
                for split, (y_s, p_s) in (("s1", (y1, p1)), ("s2", (y2, p2))):
                    for b, mask in sim_masks[split].items():
                        rec[f"{split.upper()}_sim_{b}"] = partition_metrics(
                            y_s, p_s, mask, sorted(evaluable[f"inductive_{split}"]))
            if rec:
                results[key] = rec
                log.info(f"{key}: " + ", ".join(f"{k}={v['auprc']:.3f}" for k, v in rec.items() if v))
    log.info(f"sanity: max |stored - recomputed| AUPRC over evaluated runs = {max_sanity_diff:.2e}")
    assert max_sanity_diff < 1e-3, "recomputed metrics disagree with stored ones -- do not trust this re-analysis"

    summary = {}
    for label, _, _ in MODELS:
        recs = [results[f"{label}|{s}"] for s in SEEDS if f"{label}|{s}" in results]
        if not recs:
            continue
        keys = sorted({k for r in recs for k, v in r.items() if v})
        summary[label] = {}
        for k in keys:
            vals = [r[k] for r in recs if r.get(k)]
            summary[label][k] = {m: summarize([v[m] for v in vals]) for m in ("auroc", "auprc")}
            summary[label][k]["n_classes"] = vals[0]["n_classes"]
            if "n_pairs" in vals[0]:
                summary[label][k]["n_pairs"] = vals[0]["n_pairs"]
        if "T_common" in summary[label] and "S2_common" in summary[label]:
            for split in ("S1", "S2"):
                gaps = [r["T_common"]["auprc"] - r[f"{split}_common"]["auprc"]
                        for r in recs if r.get("T_common") and r.get(f"{split}_common")]
                summary[label][f"gap_T_minus_{split}_common_auprc"] = summarize(gaps)

    with open(out_path, "w") as f:
        json.dump({"composition": comp, "common_classes": common, "similarity": sim_info,
                   "min_partition_positives": MIN_PARTITION_POSITIVES, "sanity_max_abs_diff_auprc": max_sanity_diff,
                   "summary": summary, "per_run": results}, f, indent=2, default=str)

    def fmt(s, k):
        v = s.get(k)
        return "--" if not v else f"{v['auprc']['mean']:.3f}±{v['auprc']['std']:.3f}"

    lines = ["# Camera-ready re-analysis (inference only)", "",
             f"Common evaluable class set: {len(common)} types. Sanity max |stored-recomputed| AUPRC: {max_sanity_diff:.1e}.",
             "", "| model | T common | S1 common | S2 common | gap T-S2 common | S1 train-partner | S1 val-partner |",
             "|---|---|---|---|---|---|---|"]
    for label, s in summary.items():
        g = s.get("gap_T_minus_S2_common_auprc")
        gtxt = "--" if not g else f"{g['mean']:.3f}±{g['std']:.3f}"
        lines.append(f"| {label} | {fmt(s, 'T_common')} | {fmt(s, 'S1_common')} | {fmt(s, 'S2_common')} | {gtxt} | "
                     f"{fmt(s, 'S1_train_partner')} | {fmt(s, 'S1_val_partner')} |")
    lines += ["", "| model | S1 sim low | S1 sim mid | S1 sim high | S2 sim low | S2 sim high |", "|---|---|---|---|---|---|"]
    for label, s in summary.items():
        lines.append(f"| {label} | {fmt(s, 'S1_sim_low')} | {fmt(s, 'S1_sim_mid')} | {fmt(s, 'S1_sim_high')} | "
                     f"{fmt(s, 'S2_sim_low')} | {fmt(s, 'S2_sim_high')} |")
    lines += ["", "```", json.dumps(comp, indent=1), json.dumps(sim_info, indent=1), "```"]
    with open(os.path.join(ROOT, "results", "camera_ready_reanalysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"wrote {out_path} and results/camera_ready_reanalysis.md")


if __name__ == "__main__":
    main()
