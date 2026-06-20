"""Multi-seed full-scale sweep on the real 1,706-drug dataset: fusion=concat,
graph-only, text-only, fingerprint-mlp baseline, each trained transductively
and inductively, repeated over --seeds (default 0,1,2).

The data SPLIT is fixed (always the seed-0 split saved by 04_build_full_data.py
-- "every model sees identical splits" per the brief) across all training
seeds; only the training seed (model init + batch shuffling) varies, which is
standard practice for estimating optimization variance without conflating it
with split-lottery variance.

Supports --max-epochs / --batch-size overrides for a quick timing probe
before committing to a long run.
"""
import argparse
import json
import logging
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import baselines as bl
import data_pipeline as dp
import evaluate as ev
import experiment_utils as eu
import splits as sp
import train as tr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("full_scale_calibration")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_full_artifacts():
    graphs = torch.load(os.path.join(ROOT, "data", "graph_cache.pt"), weights_only=False)
    with open(os.path.join(ROOT, "data", "text_cache.json")) as f:
        text_cache = json.load(f)
    has_text = {k: bool(v) for k, v in text_cache.items()}
    text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings.pt"), weights_only=False)
    fp_cache = torch.load(os.path.join(ROOT, "data", "fingerprint_cache.pt"), weights_only=False)
    return graphs, has_text, text_blob["embeddings"], text_blob["dim"], fp_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated subset of architectures to run, e.g. 'fusion=concat'")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "train_full.yaml")) as f:
        train_cfg_base = yaml.safe_load(f)
    if args.max_epochs is not None:
        train_cfg_base["epochs"] = args.max_epochs
    if args.batch_size is not None:
        train_cfg_base["batch_size"] = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device} train_cfg_base={train_cfg_base} seeds={seeds}")

    log.info("=== evaluate.py sanity checks ===")
    _, sanity_ok = ev.run_sanity_checks()
    if not sanity_ok:
        raise SystemExit("evaluate.py sanity checks failed -- refusing to run models against broken metrics")

    log.info("=== loading full-scale cached artifacts + splits (fixed across all training seeds) ===")
    graphs, has_text, text_emb_cache, lm_dim, fp_cache = load_full_artifacts()
    split_seed = model_cfg["seed"]  # the split itself never changes across training seeds
    splits_dir = os.path.join(ROOT, "data", "splits", f"drugbank_seed{split_seed}")
    transductive, drug_split, inductive = sp.load_splits(split_seed, tag="drugbank")
    evaluable = sp.load_evaluable_classes(splits_dir)
    log.info(f"evaluable-class counts at full scale: { {k: len(v) for k, v in evaluable.items()} }")

    common = dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=lm_dim,
                  hidden_dim=model_cfg["hidden_dim"], graph_layers=model_cfg["graph_layers"],
                  graph_conv=model_cfg["graph_conv"], graph_readout=model_cfg["graph_readout"],
                  head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"])

    trans_train = transductive[transductive["split"] == "train"]
    trans_val = transductive[transductive["split"] == "val"]
    trans_test = transductive[transductive["split"] == "test"]

    architectures = ["fusion=concat", "graph-only", "text-only", "fingerprint-mlp", "ssiddi"]
    if args.only:
        architectures = [a for a in architectures if a in set(args.only.split(","))]
    log.info(f"architectures to run: {architectures}")

    def components_for(name):
        if name == "fingerprint-mlp":
            return eu.make_fingerprint_components(
                fp_cache, fp_dim=bl.FINGERPRINT_BITS, hidden_dim=model_cfg["hidden_dim"],
                head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"],
                num_classes=dp.NUM_INTERACTION_TYPES, binary=False)
        if name == "ssiddi":
            return eu.make_ssiddi_components(graphs, dp.ATOM_FEATURE_DIM, dp.NUM_INTERACTION_TYPES)
        return eu.make_ddi_components(name, common, dp.NUM_INTERACTION_TYPES, False,
                                       graphs, text_emb_cache, has_text, lm_dim, device)

    all_results = []  # (seed, regime, name, result)
    gap_by_model_seed = {}
    for seed in seeds:
        train_cfg = {**train_cfg_base, "seed": seed}
        for name in architectures:
            log.info(f"=== seed={seed} TRANSDUCTIVE (full scale): {name} ===")
            model_factory, batch_fn = components_for(name)
            result_t = tr.train_one_run(
                run_name=f"fullscale_{name}_transductive_seed{seed}", model_factory=model_factory, batch_fn=batch_fn,
                pairs_splits={"train": trans_train, "val": trans_val, "test": trans_test},
                id1_col="ID1", id2_col="ID2", label_col="Y", binary=False, train_cfg=train_cfg,
                evaluable_classes_by_split={"val": evaluable["transductive_val"], "test": evaluable["transductive_test"]},
                device=device)
            tr.log_run(result_t, {"name": name, "split_type": "transductive", "scale": "full", "seed": seed})
            all_results.append((seed, "transductive", name, result_t))

            log.info(f"=== seed={seed} INDUCTIVE (full scale): {name} ===")
            model_factory, batch_fn = components_for(name)
            result_i = tr.train_one_run(
                run_name=f"fullscale_{name}_inductive_seed{seed}", model_factory=model_factory, batch_fn=batch_fn,
                pairs_splits={"train": inductive["train"], "val": inductive["val"],
                              "s1": inductive["s1"], "s2": inductive["s2"]},
                id1_col="ID1", id2_col="ID2", label_col="Y", binary=False, train_cfg=train_cfg,
                evaluable_classes_by_split={"val": evaluable["inductive_val"], "s1": evaluable["inductive_s1"],
                                            "s2": evaluable["inductive_s2"]},
                device=device)
            tr.log_run(result_i, {"name": name, "split_type": "inductive", "scale": "full", "seed": seed})
            all_results.append((seed, "inductive", name, result_i))

            combined = {"transductive": result_t["metrics_by_split"]["test"],
                        "s1": result_i["metrics_by_split"]["s1"], "s2": result_i["metrics_by_split"]["s2"]}
            gap_by_model_seed[(name, seed)] = ev.transductive_inductive_gap(combined)
            log.info(f"[{name} seed={seed}] transductive-minus-inductive gap: {gap_by_model_seed[(name, seed)]}")

    log.info("=== building summary table ===")
    rows = []
    for seed, regime, name, result in all_results:
        for split_name, m in result["metrics_by_split"].items():
            rows.append({"seed": seed, "regime": regime, "model": name, "eval_split": split_name,
                         "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"], "accuracy": m["accuracy"],
                         "n_evaluable_classes_used": m.get("n_evaluable_classes_used"),
                         "wall_clock_sec": result["wall_clock_sec"], "epochs_run": result["epochs_run"],
                         "stopped_early": result["stopped_early"], "hit_epoch_cap": result["hit_epoch_cap"],
                         "peak_gpu_mem_mb": result["peak_gpu_mem_mb"]})
    summary_df = pd.DataFrame(rows)

    gap_rows = []
    for (name, seed), gap in gap_by_model_seed.items():
        for gap_name, metrics in gap.items():
            gap_rows.append({"model": name, "seed": seed, "gap": gap_name, **metrics})
    gap_df = pd.DataFrame(gap_rows)

    out = {
        "seeds": seeds, "train_cfg_base": train_cfg_base,
        "evaluable_class_counts_full_scale": {k: len(v) for k, v in evaluable.items()},
        "summary_table": summary_df.to_dict(orient="records"),
        "transductive_inductive_gap": gap_df.to_dict(orient="records"),
        "total_wall_clock_sec": sum(r["wall_clock_sec"] for _, _, _, r in all_results),
        "n_runs": len(all_results),
    }
    out_path = os.path.join(ROOT, "results", "full_scale_calibration_stats.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"wrote {out_path}")

    print("\n=== FULL-SCALE SUMMARY TABLE (all seeds) ===")
    print(summary_df.to_string(index=False))
    print("\n=== TRANSDUCTIVE-MINUS-INDUCTIVE GAP (all seeds) ===")
    print(gap_df.to_string(index=False) if not gap_df.empty else "(empty)")
    print(f"\ntotal wall clock: {out['total_wall_clock_sec']:.1f}s across {out['n_runs']} runs, seeds={seeds}")


if __name__ == "__main__":
    main()
