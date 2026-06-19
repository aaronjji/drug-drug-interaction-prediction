"""End-to-end train+eval smoke test on the 150-drug subset: full fusion
(concat), graph-only, text-only -- each trained transductively and
inductively -- plus the binary head on BioSNAP, once. Gate before the full
sweep: proves train.py/evaluate.py are correct, not the scientific result.
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import data_pipeline as dp
import evaluate as ev
import splits as sp
import train as tr
from models import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_eval_smoke_test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_smoke_artifacts():
    graphs = torch.load(os.path.join(ROOT, "data", "graph_cache_smoke.pt"), weights_only=False)
    with open(os.path.join(ROOT, "data", "text_cache_smoke.json")) as f:
        text_cache = json.load(f)
    has_text = {k: bool(v) for k, v in text_cache.items()}
    text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings_smoke.pt"), weights_only=False)
    return graphs, has_text, text_blob["embeddings"], text_blob["dim"]


def build_binary_splits(data_cfg, train_cfg, graphs, seed):
    smiles_map_full = dp.drug_smiles_table(dp.load_drugbank_pairs(data_cfg["drugbank_repo_raw"]))
    chch = dp.load_chch_miner_binary(data_cfg["chch_miner_url"], smiles_map_full)
    subset_drug_ids = list(graphs.keys())
    subset_set = set(subset_drug_ids)
    chch_subset = chch[chch["d1"].isin(subset_set) & chch["d2"].isin(subset_set)].reset_index(drop=True)
    log.info(f"BioSNAP positives induced on smoke subset: {len(chch_subset)}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(chch_subset))
    n_test = int(round(len(perm) * train_cfg["binary_test_frac"]))
    n_val = int(round(len(perm) * train_cfg["binary_val_frac"]))
    idx = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}

    all_positives = list(zip(chch_subset["d1"], chch_subset["d2"]))
    sampler = NegativeSampler(subset_drug_ids, all_positives, seed=seed, ratio=train_cfg["negative_sample_ratio"])

    splits = {}
    for name in ["train", "val", "test"]:
        pos_rows = chch_subset.iloc[idx[name]]
        pos_pairs = list(zip(pos_rows["d1"], pos_rows["d2"]))
        neg_pairs = sampler.sample(len(pos_pairs))
        df = pd.DataFrame(
            {"d1": [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs],
             "d2": [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs],
             "label": [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)})
        splits[name] = df
        log.info(f"binary {name}: {len(pos_pairs)} positives + {len(neg_pairs)} negatives "
                 f"(ratio={train_cfg['negative_sample_ratio']}:1)")
    return splits


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "train.yaml")) as f:
        train_cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    log.info("=== evaluate.py sanity checks (must pass before trusting any model metric) ===")
    sanity_results, sanity_ok = ev.run_sanity_checks()
    if not sanity_ok:
        raise SystemExit("evaluate.py sanity checks failed -- refusing to run models against broken metrics")

    log.info("=== loading cached artifacts + splits (built in steps 3-4) ===")
    graphs, has_text, text_emb_cache, lm_dim = load_smoke_artifacts()
    splits_dir = os.path.join(ROOT, "data", "splits", "smoke_drugbank_seed0")
    transductive, drug_split, inductive = sp.load_splits(model_cfg["seed"], tag="smoke_drugbank")
    evaluable = sp.load_evaluable_classes(splits_dir)
    log.info(f"loaded evaluable-class sets from disk: { {k: len(v) for k, v in evaluable.items()} }")

    common = dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=lm_dim,
                  hidden_dim=model_cfg["hidden_dim"], graph_layers=model_cfg["graph_layers"],
                  graph_conv=model_cfg["graph_conv"], graph_readout=model_cfg["graph_readout"],
                  head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"])

    all_results = []

    trans_train = transductive[transductive["split"] == "train"]
    trans_val = transductive[transductive["split"] == "val"]
    trans_test = transductive[transductive["split"] == "test"]

    multiclass_configs = [
        ("fusion=concat", dict(fusion_type="concat", use_graph=True, use_text=True)),
        ("graph-only", dict(fusion_type="concat", use_graph=True, use_text=False)),
        ("text-only", dict(fusion_type="concat", use_graph=False, use_text=True)),
    ]

    gap_by_model = {}
    for name, flags in multiclass_configs:
        model_kwargs = {**common, **flags}

        log.info(f"=== TRANSDUCTIVE: {name} ===")
        result_t = tr.train_one_run(
            run_name=f"{name}_transductive", model_kwargs=model_kwargs,
            pairs_splits={"train": trans_train, "val": trans_val, "test": trans_test},
            id1_col="ID1", id2_col="ID2", label_col="Y", binary=False,
            graphs=graphs, text_emb_cache=text_emb_cache, has_text=has_text, lm_dim=lm_dim,
            train_cfg=train_cfg,
            evaluable_classes_by_split={"val": evaluable["transductive_val"], "test": evaluable["transductive_test"]},
            device=device, num_classes=dp.NUM_INTERACTION_TYPES)
        tr.log_run(result_t, {"name": name, "split_type": "transductive", **flags})
        all_results.append(("transductive", name, result_t))

        log.info(f"=== INDUCTIVE: {name} ===")
        result_i = tr.train_one_run(
            run_name=f"{name}_inductive", model_kwargs=model_kwargs,
            pairs_splits={"train": inductive["train"], "val": inductive["val"],
                          "s1": inductive["s1"], "s2": inductive["s2"]},
            id1_col="ID1", id2_col="ID2", label_col="Y", binary=False,
            graphs=graphs, text_emb_cache=text_emb_cache, has_text=has_text, lm_dim=lm_dim,
            train_cfg=train_cfg,
            evaluable_classes_by_split={"val": evaluable["inductive_val"], "s1": evaluable["inductive_s1"],
                                        "s2": evaluable["inductive_s2"]},
            device=device, num_classes=dp.NUM_INTERACTION_TYPES)
        tr.log_run(result_i, {"name": name, "split_type": "inductive", **flags})
        all_results.append(("inductive", name, result_i))

        # transductive and inductive are separate trained models (different
        # train sets), so the headline gap is computed here, once both exist,
        # by combining each run's own metrics_by_split into one lookup.
        combined = {"transductive": result_t["metrics_by_split"]["test"],
                    "s1": result_i["metrics_by_split"]["s1"], "s2": result_i["metrics_by_split"]["s2"]}
        gap_by_model[name] = ev.transductive_inductive_gap(combined)
        log.info(f"[{name}] transductive-minus-inductive gap: {gap_by_model[name]}")

    log.info("=== BINARY: fusion=concat on BioSNAP ChCh-Miner ===")
    binary_splits = build_binary_splits(data_cfg, train_cfg, graphs, model_cfg["seed"])
    result_b = tr.train_one_run(
        run_name="fusion=concat_binary_biosnap",
        model_kwargs={**common, "fusion_type": "concat", "use_graph": True, "use_text": True},
        pairs_splits={"train": binary_splits["train"], "val": binary_splits["val"], "test": binary_splits["test"]},
        id1_col="d1", id2_col="d2", label_col="label", binary=True,
        graphs=graphs, text_emb_cache=text_emb_cache, has_text=has_text, lm_dim=lm_dim,
        train_cfg=train_cfg, evaluable_classes_by_split={}, device=device)
    tr.log_run(result_b, {"name": "fusion=concat", "split_type": "binary_biosnap"})
    all_results.append(("binary_biosnap", "fusion=concat", result_b))

    log.info("=== building summary table ===")
    rows = []
    for regime, name, result in all_results:
        for split_name, m in result["metrics_by_split"].items():
            row = {"regime": regime, "model": name, "eval_split": split_name,
                   "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"], "accuracy": m["accuracy"],
                   "n_evaluable_classes_used": m.get("n_evaluable_classes_used"),
                   "wall_clock_sec": result["wall_clock_sec"], "epochs_run": result["epochs_run"]}
            rows.append(row)
    summary_df = pd.DataFrame(rows)

    gap_rows = []
    for name, gap in gap_by_model.items():
        for gap_name, metrics in gap.items():
            gap_rows.append({"model": name, "gap": gap_name, **metrics})
    gap_df = pd.DataFrame(gap_rows)

    eval_consistency = {}
    for split_key in ["transductive_test", "inductive_s1", "inductive_s2"]:
        eval_consistency[split_key] = {"n_classes_on_disk": len(evaluable[split_key])}

    out = {
        "sanity_checks": {k: v["pass"] for k, v in sanity_results.items()},
        "summary_table": summary_df.to_dict(orient="records"),
        "transductive_inductive_gap": gap_df.to_dict(orient="records"),
        "evaluable_class_consistency_check": eval_consistency,
        "total_wall_clock_sec": sum(r["wall_clock_sec"] for _, _, r in all_results),
        "n_runs": len(all_results),
    }
    out_path = os.path.join(ROOT, "results", "train_eval_smoke_test_stats.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"wrote {out_path}")

    print("\n=== SUMMARY TABLE ===")
    print(summary_df.to_string(index=False))
    print("\n=== TRANSDUCTIVE-MINUS-INDUCTIVE GAP ===")
    print(gap_df.to_string(index=False) if not gap_df.empty else "(empty)")
    print(f"\nsanity checks: {out['sanity_checks']}")
    print(f"total wall clock: {out['total_wall_clock_sec']:.1f}s across {out['n_runs']} runs")


if __name__ == "__main__":
    main()
