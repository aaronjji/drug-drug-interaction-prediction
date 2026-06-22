"""Full-scale BioSNAP ChCh-Miner binary robustness check: same 4 architectures,
same transductive+inductive(S1/S2) regimes, same 3-seed protocol as the
DrugBank sweep, reusing the identical split/evaluable-class/negative-sampling
harness. All drug-level artifacts (graphs, text, fingerprints) are already
cached from 04_build_full_data.py since ChCh-Miner's drugs are a subset of
DrugBank's 1,706.

Framing: this is NOT a "which model wins overall" comparison. The claim under
test is whether fusion degrades LEAST and wins cold-start (S1/S2) here too,
replicating the DrugBank pattern -- not whether it's the transductive SOTA.
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

import baselines as bl
import data_pipeline as dp
import evaluate as ev
import experiment_utils as eu
import splits as sp
import train as tr
from models import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("full_scale_binary_sweep")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_SEED = 0  # split is fixed across training seeds, same convention as DrugBank


def load_full_artifacts():
    graphs = torch.load(os.path.join(ROOT, "data", "graph_cache.pt"), weights_only=False)
    with open(os.path.join(ROOT, "data", "text_cache.json")) as f:
        text_cache = json.load(f)
    has_text = {k: bool(v) for k, v in text_cache.items()}
    text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings.pt"), weights_only=False)
    fp_cache = torch.load(os.path.join(ROOT, "data", "fingerprint_cache.pt"), weights_only=False)
    return graphs, has_text, text_blob["embeddings"], text_blob["dim"], fp_cache


def build_negative_augmented_splits(chch_df, drug_ids, drug_split, train_cfg, ratio):
    """Returns (transductive_splits, inductive_splits) dicts of {name: DataFrame[d1,d2,label]}
    for one training seed. Negatives are bucket-aware for inductive (respect the same
    drug-membership rule as the positives in that bucket) and globally-excluded-positive
    for transductive."""
    seed = train_cfg["seed"]
    all_positive_pairs = list(zip(chch_df["d1"], chch_df["d2"]))
    all_positive_set = {frozenset(p) for p in all_positive_pairs}

    # transductive: single seeded sampler, called in a fixed order so its dedup
    # ("seen") accumulator guarantees no negative is reused across train/val/test
    trans_sampler = NegativeSampler(drug_ids, all_positive_pairs, seed=seed, ratio=ratio)
    transductive = {}
    for name in ["train", "val", "test"]:
        pos_rows = chch_df[chch_df["split"] == name]
        pos_pairs = list(zip(pos_rows["d1"], pos_rows["d2"]))
        neg_pairs = trans_sampler.sample(len(pos_pairs))
        transductive[name] = pd.DataFrame(
            {"d1": [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs],
             "d2": [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs],
             "label": [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)})

    # inductive: bucket-aware negatives, independently seeded per bucket (safe --
    # buckets draw from disjoint drug-pool combinations by construction)
    inductive_pos = sp.derive_inductive_pairs(chch_df, "d1", "d2", drug_split)
    inductive = {}
    for name, pos_df in inductive_pos.items():
        pos_pairs = list(zip(pos_df["d1"], pos_df["d2"]))
        neg_pairs = sp.sample_bucket_negatives(name, drug_split, all_positive_set,
                                                n_needed=len(pos_pairs), seed=seed, ratio=ratio)
        assert not any(frozenset(p) in all_positive_set for p in neg_pairs), \
            f"a sampled '{name}' negative coincides with a known positive"
        inductive[name] = pd.DataFrame(
            {"d1": [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs],
             "d2": [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs],
             "label": [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)})
    return transductive, inductive


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "train_full.yaml")) as f:
        train_cfg_base = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0, 1, 2]
    ratio = train_cfg_base["negative_sample_ratio"]
    log.info(f"device={device} train_cfg_base={train_cfg_base} seeds={seeds} negative_ratio={ratio}:1")

    log.info("=== evaluate.py sanity checks ===")
    _, sanity_ok = ev.run_sanity_checks()
    if not sanity_ok:
        raise SystemExit("evaluate.py sanity checks failed")

    log.info("=== loading shared artifacts (already cached from the DrugBank build) ===")
    graphs, has_text, text_emb_cache, lm_dim, fp_cache = load_full_artifacts()
    smiles_map_full = dp.drug_smiles_table(dp.load_drugbank_pairs(data_cfg["drugbank_repo_raw"]))
    chch_full = dp.load_chch_miner_binary(data_cfg["chch_miner_url"], smiles_map_full)
    drug_ids = sorted(set(chch_full["d1"]) | set(chch_full["d2"]))
    log.info(f"ChCh-Miner: {len(chch_full):,} positive pairs, {len(drug_ids):,} drugs")

    log.info("=== building fixed split (seed=0) ===")
    sp.set_seed(SPLIT_SEED)
    chch_split = sp.build_transductive_split(
        chch_full, label_col=None, val_frac=data_cfg["transductive_val_frac"],
        test_frac=data_cfg["transductive_test_frac"], seed=SPLIT_SEED)
    drug_split, coverage_report = sp.build_inductive_drug_split(
        drug_ids, has_text, train_frac=data_cfg["inductive_train_frac"],
        val_frac=data_cfg["inductive_val_frac"], test_frac=data_cfg["inductive_test_frac"],
        seed=SPLIT_SEED, stratify_by_text=True)
    inductive_pos = sp.derive_inductive_pairs(chch_full, "d1", "d2", drug_split)
    out_dir = sp.save_splits(SPLIT_SEED, chch_split, drug_split, inductive_pos, tag="chch")

    train_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["train"]]))
    test_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["test"]]))
    log.info(f"=== text-coverage confound check (train vs S1/S2-pool test drugs) ===")
    log.info(f"before rebalance: {coverage_report['before_rebalance_coverage']}")
    log.info(f"after rebalance:  {coverage_report['after_rebalance_coverage']}")
    log.info(f"train drugs text coverage: {train_drug_cov:.3f} ({len(drug_split['train'])} drugs)")
    log.info(f"test (S1/S2) drugs text coverage: {test_drug_cov:.3f} ({len(drug_split['test'])} drugs)")

    log.info(f"transductive pair counts: "
             f"{ {n: int((chch_split['split']==n).sum()) for n in ['train','val','test']} }")
    log.info(f"inductive positive pair counts: { {k: len(v) for k, v in inductive_pos.items()} }")

    common = dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=lm_dim,
                  hidden_dim=model_cfg["hidden_dim"], graph_layers=model_cfg["graph_layers"],
                  graph_conv=model_cfg["graph_conv"], graph_readout=model_cfg["graph_readout"],
                  head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"])
    architectures = ["fusion=concat", "graph-only", "text-only", "fingerprint-mlp"]

    def components_for(name):
        if name == "fingerprint-mlp":
            return eu.make_fingerprint_components(
                fp_cache, fp_dim=bl.FINGERPRINT_BITS, hidden_dim=model_cfg["hidden_dim"],
                head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"],
                num_classes=None, binary=True)
        return eu.make_ddi_components(name, common, None, True, graphs, text_emb_cache, has_text, lm_dim, device)

    all_results = []
    gap_by_model_seed = {}
    neg_sample_log = []
    for seed in seeds:
        train_cfg = {**train_cfg_base, "seed": seed}
        log.info(f"=== seed={seed}: sampling negatives (ratio={ratio}:1, reseeded per training seed) ===")
        transductive, inductive = build_negative_augmented_splits(chch_split, drug_ids, drug_split, train_cfg, ratio)
        neg_sample_log.append({
            "seed": seed, "ratio": ratio,
            "transductive_counts": {n: int((df["label"] == 0).sum()) for n, df in transductive.items()},
            "inductive_counts": {n: int((df["label"] == 0).sum()) for n, df in inductive.items()},
        })

        for name in architectures:
            log.info(f"=== seed={seed} TRANSDUCTIVE (BioSNAP): {name} ===")
            model_factory, batch_fn = components_for(name)
            result_t = tr.train_one_run(
                run_name=f"binary_{name}_transductive_seed{seed}", model_factory=model_factory, batch_fn=batch_fn,
                pairs_splits={"train": transductive["train"], "val": transductive["val"], "test": transductive["test"]},
                id1_col="d1", id2_col="d2", label_col="label", binary=True, train_cfg=train_cfg,
                evaluable_classes_by_split={}, device=device)
            tr.log_run(result_t, {"name": name, "split_type": "transductive", "dataset": "chch_biosnap", "seed": seed})
            all_results.append((seed, "transductive", name, result_t))

            log.info(f"=== seed={seed} INDUCTIVE (BioSNAP): {name} ===")
            model_factory, batch_fn = components_for(name)
            result_i = tr.train_one_run(
                run_name=f"binary_{name}_inductive_seed{seed}", model_factory=model_factory, batch_fn=batch_fn,
                pairs_splits={"train": inductive["train"], "val": inductive["val"],
                              "s1": inductive["s1"], "s2": inductive["s2"]},
                id1_col="d1", id2_col="d2", label_col="label", binary=True, train_cfg=train_cfg,
                evaluable_classes_by_split={}, device=device)
            tr.log_run(result_i, {"name": name, "split_type": "inductive", "dataset": "chch_biosnap", "seed": seed})
            all_results.append((seed, "inductive", name, result_i))

            combined = {"transductive": result_t["metrics_by_split"]["test"],
                        "s1": result_i["metrics_by_split"]["s1"], "s2": result_i["metrics_by_split"]["s2"]}
            gap_by_model_seed[(name, seed)] = ev.transductive_inductive_gap(combined)
            log.info(f"[{name} seed={seed}] gap: {gap_by_model_seed[(name, seed)]}")

    log.info("=== building summary table ===")
    rows = []
    for seed, regime, name, result in all_results:
        for split_name, m in result["metrics_by_split"].items():
            rows.append({"seed": seed, "regime": regime, "model": name, "eval_split": split_name,
                         "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"], "accuracy": m["accuracy"],
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
        "dataset": "BioSNAP ChCh-Miner", "n_positive_pairs": len(chch_full), "n_drugs": len(drug_ids),
        "negative_sampling": neg_sample_log,
        "text_coverage_confound": {
            "before_rebalance": coverage_report["before_rebalance_coverage"],
            "after_rebalance": coverage_report["after_rebalance_coverage"],
            "train_drugs_coverage": train_drug_cov, "test_drugs_coverage": test_drug_cov,
        },
        "transductive_pair_counts": {n: int((chch_split["split"] == n).sum()) for n in ["train", "val", "test"]},
        "inductive_positive_pair_counts": {k: len(v) for k, v in inductive_pos.items()},
        "summary_table": summary_df.to_dict(orient="records"),
        "transductive_inductive_gap": gap_df.to_dict(orient="records"),
        "total_wall_clock_sec": sum(r["wall_clock_sec"] for _, _, _, r in all_results),
        "n_runs": len(all_results),
        "splits_saved_to": out_dir,
    }
    out_path = os.path.join(ROOT, "results", "full_scale_binary_sweep_stats.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"wrote {out_path}")

    print("\n=== BIOSNAP FULL-SCALE SUMMARY TABLE (all seeds) ===")
    print(summary_df.to_string(index=False))
    print("\n=== TRANSDUCTIVE-MINUS-INDUCTIVE GAP (all seeds) ===")
    print(gap_df.to_string(index=False) if not gap_df.empty else "(empty)")
    print(f"\ntotal wall clock: {out['total_wall_clock_sec']:.1f}s across {out['n_runs']} runs, seeds={seeds}")


if __name__ == "__main__":
    main()
