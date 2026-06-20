"""Cheap follow-up on the BioSNAP F1 exception (graph-only edges out fusion
on S2-F1 despite fusion winning AUROC/AUPRC): F1 at a fixed 0.5 threshold is
sensitive to probability calibration, unlike the ranking metrics. Tunes the
decision threshold on the inductive VAL split (never on S2 itself -- that
would be leakage) and re-evaluates S2 F1 at that threshold, for both models,
all 3 seeds. Reuses the exact saved checkpoints and the exact same
deterministic negative sampling as the original sweep -- no retraining.
"""
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import data_pipeline as dp
import experiment_utils as eu
import splits as sp
import train as tr
from models import NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("f1_threshold_check")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_val_s2_negatives(chch_df, drug_ids, drug_split, seed, ratio=1):
    """Reconstructs the exact same val/s2 negative-augmented sets the original
    sweep used for this seed -- sample_bucket_negatives is deterministic given
    the same seed, so this reproduces the identical evaluation data without
    needing to re-run training."""
    all_positive_pairs = list(zip(chch_df["d1"], chch_df["d2"]))
    all_positive_set = {frozenset(p) for p in all_positive_pairs}
    inductive_pos = sp.derive_inductive_pairs(chch_df, "d1", "d2", drug_split)
    out = {}
    for name in ["val", "s2"]:
        pos_df = inductive_pos[name]
        pos_pairs = list(zip(pos_df["d1"], pos_df["d2"]))
        neg_pairs = sp.sample_bucket_negatives(name, drug_split, all_positive_set,
                                                n_needed=len(pos_pairs), seed=seed, ratio=ratio)
        import pandas as pd
        out[name] = pd.DataFrame(
            {"d1": [p[0] for p in pos_pairs] + [p[0] for p in neg_pairs],
             "d2": [p[1] for p in pos_pairs] + [p[1] for p in neg_pairs],
             "label": [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)})
    return out


def best_f1_threshold(labels, probs):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs = torch.load(os.path.join(ROOT, "data", "graph_cache.pt"), weights_only=False)
    with open(os.path.join(ROOT, "data", "text_cache.json")) as f:
        text_cache = json.load(f)
    has_text = {k: bool(v) for k, v in text_cache.items()}
    text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings.pt"), weights_only=False)
    text_emb_cache, lm_dim = text_blob["embeddings"], text_blob["dim"]

    smiles_map_full = dp.drug_smiles_table(dp.load_drugbank_pairs(data_cfg["drugbank_repo_raw"]))
    chch_full = dp.load_chch_miner_binary(data_cfg["chch_miner_url"], smiles_map_full)
    drug_ids = sorted(set(chch_full["d1"]) | set(chch_full["d2"]))

    chch_split, drug_split, inductive_pos = sp.load_splits(0, tag="chch")

    common = dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=lm_dim,
                  hidden_dim=model_cfg["hidden_dim"], graph_layers=model_cfg["graph_layers"],
                  graph_conv=model_cfg["graph_conv"], graph_readout=model_cfg["graph_readout"],
                  head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"])

    results = []
    for name in ["fusion=concat", "graph-only"]:
        for seed in [0, 1, 2]:
            val_s2 = build_val_s2_negatives(chch_split, drug_ids, drug_split, seed=seed, ratio=1)
            val_df, s2_df = val_s2["val"], val_s2["s2"]

            model_factory, batch_fn = eu.make_ddi_components(
                name, common, None, True, graphs, text_emb_cache, has_text, lm_dim, device)
            model = model_factory().to(device)
            ckpt_path = os.path.join(ROOT, "results", "checkpoints", f"binary_{name}_inductive_seed{seed}.pt")
            model.load_state_dict(torch.load(ckpt_path, weights_only=False))

            val_probs, val_labels = tr._predict_split(model, val_df, "d1", "d2", "label", batch_fn,
                                                        binary=True, device=device, batch_size=128)
            s2_probs, s2_labels = tr._predict_split(model, s2_df, "d1", "d2", "label", batch_fn,
                                                      binary=True, device=device, batch_size=128)

            f1_fixed = f1_score(s2_labels, (s2_probs >= 0.5).astype(int))
            tuned_t = best_f1_threshold(val_labels, val_probs)
            f1_tuned = f1_score(s2_labels, (s2_probs >= tuned_t).astype(int))

            results.append({"model": name, "seed": seed, "tuned_threshold": tuned_t,
                             "s2_f1_at_0.5": f1_fixed, "s2_f1_at_tuned": f1_tuned})
            log.info(f"{name} seed={seed}: tuned_threshold={tuned_t:.3f} "
                     f"S2_F1(0.5)={f1_fixed:.4f} S2_F1(tuned)={f1_tuned:.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    print("\n=== F1 THRESHOLD-TUNING CHECK (BioSNAP S2) ===")
    print(df.to_string(index=False))
    agg = df.groupby("model")[["s2_f1_at_0.5", "s2_f1_at_tuned"]].agg(["mean", "std"])
    print("\n=== mean +- std across seeds ===")
    print(agg.round(4))

    out_path = os.path.join(ROOT, "results", "biosnap_f1_threshold_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
