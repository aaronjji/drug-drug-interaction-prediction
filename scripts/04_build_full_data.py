"""Build the real (non-smoke) caches and seed-0 splits for the full
1,706-drug / 191,808-pair DrugBank dataset: graphs, PubChem text (slow,
~15 min one-time, cached), frozen-LM text embeddings, Morgan fingerprints,
transductive + inductive splits, and the persisted evaluable-class sets.
Gate before the full-scale calibration sweep.
"""
import json
import logging
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import baselines as bl
import data_pipeline as dp
import splits as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_full_data")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sp.set_seed(cfg["seed"])

    log.info("=== loading full DrugBank multi-relational pairs ===")
    pairs = dp.load_drugbank_pairs(cfg["drugbank_repo_raw"])
    smiles_map = dp.drug_smiles_table(pairs)
    log.info(f"full dataset: {len(pairs):,} pairs, {len(smiles_map):,} drugs, "
             f"{pairs['Y'].nunique()} interaction types (0-indexed)")

    log.info("=== molecular graph construction (RDKit -> PyG), all drugs ===")
    graph_cache = dp.GraphCache()  # default (non-smoke) path
    graphs = graph_cache.build_or_load(smiles_map)
    n_parsed = sum(1 for i in smiles_map["id"] if i in graphs)
    log.info(f"graphs built: {n_parsed}/{len(smiles_map)}")

    log.info("=== Morgan/ECFP fingerprints, all drugs ===")
    fp_cache_obj = bl.FingerprintCache()  # default (non-smoke) path
    fp_cache = fp_cache_obj.build_or_load(smiles_map)
    log.info(f"fingerprints built: {len(fp_cache)}/{len(smiles_map)}")

    log.info("=== drug text fetch + cache (PubChem), all drugs -- slow, one-time ===")
    text_cache_obj = dp.TextCache()  # default (non-smoke) path
    text_cache = text_cache_obj.build_or_load(smiles_map, rate_limit_sec=cfg["pubchem_rate_limit_sec"],
                                                max_retries=cfg["pubchem_max_retries"])
    has_text = text_cache_obj.has_text()
    n_with_text = sum(has_text.values())
    log.info(f"text coverage on full dataset: {n_with_text}/{len(smiles_map)} ({100*n_with_text/len(smiles_map):.1f}%)")

    log.info("=== frozen LM text embeddings, all drugs with text ===")
    embeddings, dim = dp.build_or_load_text_embeddings(text_cache, cfg["text_model_name"])
    log.info(f"text embeddings: {len(embeddings)} drugs, dim={dim}")

    log.info("=== transductive split (pairs), seed=0 ===")
    transductive = sp.build_transductive_split(
        pairs, label_col="Y", val_frac=cfg["transductive_val_frac"],
        test_frac=cfg["transductive_test_frac"], seed=cfg["seed"],
        min_stratify_count=cfg["transductive_min_stratify_count"])

    log.info("=== inductive drug split (cold start), seed=0 ===")
    drug_split, coverage_report = sp.build_inductive_drug_split(
        list(smiles_map["id"]), has_text,
        train_frac=cfg["inductive_train_frac"], val_frac=cfg["inductive_val_frac"],
        test_frac=cfg["inductive_test_frac"], seed=cfg["seed"], stratify_by_text=True)
    inductive = sp.derive_inductive_pairs(pairs, "ID1", "ID2", drug_split)

    log.info("=== evaluable-class policy (computed once, persisted, reused at eval time) ===")
    evaluable_all = sp.compute_evaluable_classes_for_splits(transductive, inductive, "Y", k=cfg["evaluable_class_min_k"])
    for name, v in evaluable_all.items():
        log.info(f"  {name}: {v['info']['n_evaluable']}/{v['info']['n_total_classes']} evaluable (k={cfg['evaluable_class_min_k']})")

    train_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["train"]]))
    test_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["test"]]))
    log.info(f"train drugs text coverage: {train_drug_cov:.3f} ({len(drug_split['train'])} drugs)")
    log.info(f"test (S1/S2) drugs text coverage: {test_drug_cov:.3f} ({len(drug_split['test'])} drugs)")

    out_dir = sp.save_splits(cfg["seed"], transductive, drug_split, inductive, tag="drugbank")
    sp.save_evaluable_classes(out_dir, evaluable_all)

    stats = {
        "full_dataset": {"drugs": len(smiles_map), "pairs": len(pairs), "interaction_types": int(pairs["Y"].nunique())},
        "graphs_parsed": n_parsed,
        "fingerprints_built": len(fp_cache),
        "text_coverage": {"with_text": n_with_text, "total": len(smiles_map), "fraction": n_with_text / len(smiles_map)},
        "text_embeddings": {"n": len(embeddings), "dim": dim},
        "transductive_split": {name: int((transductive["split"] == name).sum()) for name in ["train", "val", "test"]},
        "inductive_drug_split": {name: len(drug_split[name]) for name in ["train", "val", "test"]},
        "inductive_pairs": {name: len(inductive[name]) for name in ["train", "val", "s1", "s2"]},
        "text_coverage_rebalancing": coverage_report,
        "train_vs_test_drug_text_coverage": {"train": train_drug_cov, "test": test_drug_cov},
        "evaluable_classes": {name: v["info"] for name, v in evaluable_all.items()},
        "splits_saved_to": out_dir,
    }
    out_path = os.path.join(ROOT, "results", "full_data_build_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info(f"wrote {out_path}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
