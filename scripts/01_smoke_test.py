"""Tiny-subset smoke test for data_pipeline.py + splits.py. Proves the whole
data/split pipeline runs end to end on a small sample before any full run.
Writes results/smoke_test_stats.json and prints a human-readable summary.
Trains no model -- that is gated separately after this passes review.
"""
import json
import logging
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import data_pipeline as dp
import splits as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoke_test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    sp.set_seed(cfg["seed"])

    log.info("=== loading full DrugBank multi-relational pairs ===")
    pairs_full = dp.load_drugbank_pairs(cfg["drugbank_repo_raw"])
    smiles_map_full = dp.drug_smiles_table(pairs_full)
    log.info(f"full dataset: {len(pairs_full):,} pairs, {len(smiles_map_full):,} drugs, "
              f"{pairs_full['Y'].nunique()} interaction types")

    log.info("=== sampling tiny subset for smoke test ===")
    n_sample = cfg["smoke_test_n_drugs"]
    rng = np.random.RandomState(cfg["seed"])
    sample_drugs = set(rng.choice(smiles_map_full["id"].values, size=n_sample, replace=False))
    smiles_map = smiles_map_full[smiles_map_full["id"].isin(sample_drugs)].reset_index(drop=True)
    pairs = pairs_full[pairs_full["ID1"].isin(sample_drugs) & pairs_full["ID2"].isin(sample_drugs)].reset_index(drop=True)
    log.info(f"subset: {len(smiles_map)} drugs, {len(pairs)} induced pairs, "
              f"{pairs['Y'].nunique()} distinct interaction types present")

    log.info("=== molecular graph construction (RDKit -> PyG) ===")
    graph_cache = dp.GraphCache(cache_path=os.path.join(ROOT, "data", "graph_cache_smoke.pt"))
    graphs = graph_cache.build_or_load(smiles_map)
    n_parsed = sum(1 for i in smiles_map["id"] if i in graphs)
    log.info(f"graphs built: {n_parsed}/{len(smiles_map)}; "
              f"atom_feature_dim={dp.ATOM_FEATURE_DIM} bond_feature_dim={dp.BOND_FEATURE_DIM}")

    log.info("=== drug text fetch + cache (PubChem) ===")
    text_cache_obj = dp.TextCache(cache_path=os.path.join(ROOT, "data", "text_cache_smoke.json"))
    text_cache = text_cache_obj.build_or_load(smiles_map, rate_limit_sec=cfg["pubchem_rate_limit_sec"],
                                                max_retries=cfg["pubchem_max_retries"])
    has_text = text_cache_obj.has_text()
    n_with_text = sum(has_text.values())
    log.info(f"text coverage on subset: {n_with_text}/{len(smiles_map)} ({100*n_with_text/len(smiles_map):.1f}%)")

    log.info("=== frozen LM text embeddings ===")
    embeddings, dim = dp.build_or_load_text_embeddings(
        text_cache, cfg["text_model_name"],
        cache_path=os.path.join(ROOT, "data", "text_embeddings_smoke.pt"))
    log.info(f"text embeddings: {len(embeddings)} drugs, dim={dim}")

    log.info("=== transductive split (pairs) ===")
    transductive = sp.build_transductive_split(
        pairs, label_col="Y", val_frac=cfg["transductive_val_frac"],
        test_frac=cfg["transductive_test_frac"], seed=cfg["seed"],
        min_stratify_count=cfg["transductive_min_stratify_count"])

    log.info("=== inductive drug split (cold start) ===")
    drug_split, coverage_report = sp.build_inductive_drug_split(
        list(smiles_map["id"]), has_text,
        train_frac=cfg["inductive_train_frac"], val_frac=cfg["inductive_val_frac"],
        test_frac=cfg["inductive_test_frac"], seed=cfg["seed"], stratify_by_text=True)
    inductive = sp.derive_inductive_pairs(pairs, "ID1", "ID2", drug_split)

    log.info("=== evaluable-class policy (computed once, persisted, reused at eval time) ===")
    evaluable_all = sp.compute_evaluable_classes_for_splits(transductive, inductive, "Y", k=cfg["evaluable_class_min_k"])
    trans_info = evaluable_all["transductive_test"]["info"]
    s1_info = evaluable_all["inductive_s1"]["info"]
    s2_info = evaluable_all["inductive_s2"]["info"]

    log.info("=== text coverage: train drugs vs inductive-test drugs (S1/S2 pool) ===")
    train_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["train"]])) if drug_split["train"] else float("nan")
    test_drug_cov = float(np.mean([has_text.get(i, False) for i in drug_split["test"]])) if drug_split["test"] else float("nan")
    log.info(f"train drugs text coverage: {train_drug_cov:.3f} ({len(drug_split['train'])} drugs)")
    log.info(f"test (S1/S2) drugs text coverage: {test_drug_cov:.3f} ({len(drug_split['test'])} drugs)")

    out_dir = sp.save_splits(cfg["seed"], transductive, drug_split, inductive, tag="smoke_drugbank")
    sp.save_evaluable_classes(out_dir, evaluable_all)

    log.info("=== ChCh-Miner binary dataset (already resolved decision, re-verified) ===")
    chch = dp.load_chch_miner_binary(cfg["chch_miner_url"], smiles_map_full)
    chch_drugs = set(chch["d1"]) | set(chch["d2"])

    stats = {
        "full_dataset": {
            "drugs": len(smiles_map_full), "pairs": len(pairs_full),
            "interaction_types": int(pairs_full["Y"].nunique()),
        },
        "smoke_subset": {
            "n_sample_drugs": n_sample, "drugs": len(smiles_map), "pairs": len(pairs),
            "interaction_types_present": int(pairs["Y"].nunique()),
            "graphs_parsed": n_parsed, "atom_feature_dim": dp.ATOM_FEATURE_DIM,
            "bond_feature_dim": dp.BOND_FEATURE_DIM,
        },
        "text": {
            "subset_coverage": n_with_text / len(smiles_map),
            "embeddings_built": len(embeddings), "embedding_dim": dim,
        },
        "transductive_split": {
            "train_pairs": int((transductive["split"] == "train").sum()),
            "val_pairs": int((transductive["split"] == "val").sum()),
            "test_pairs": int((transductive["split"] == "test").sum()),
            "evaluable_classes_vs_test": trans_info,
        },
        "inductive_drug_split": {
            "n_train_drugs": len(drug_split["train"]), "n_val_drugs": len(drug_split["val"]),
            "n_test_drugs": len(drug_split["test"]),
            "text_coverage_rebalancing": coverage_report,
            "train_drugs_text_coverage": train_drug_cov,
            "test_drugs_text_coverage": test_drug_cov,
        },
        "inductive_pairs": {
            "train": len(inductive["train"]), "val": len(inductive["val"]),
            "s1": len(inductive["s1"]), "s2": len(inductive["s2"]),
            "s1_evaluable_classes": s1_info, "s2_evaluable_classes": s2_info,
        },
        "binary_dataset_decision": {
            "twosides_unique_Y": 963, "twosides_unique_pairs": 63472, "twosides_total_rows": 4576287,
            "twosides_pairs_with_multiple_labels": 63398,
            "conclusion": "TWOSIDES is positives-only multi-relational (avg 72 labels/pair); "
                           "switched binary robustness dataset to BioSNAP ChCh-Miner.",
            "chch_miner_total_pairs": len(chch), "chch_miner_drugs": len(chch_drugs),
        },
        "splits_saved_to": out_dir,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "smoke_test_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info(f"wrote results/smoke_test_stats.json")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
