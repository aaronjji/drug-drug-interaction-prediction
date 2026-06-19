"""Forward + backward smoke test for models.py on the 150-drug subset, before
any real training. For every config (each fusion type, graph-only, text-only,
binary head + negative sampling): one forward pass, one backward pass, report
param counts, output shape, and which branches actually received gradients.
Also verifies the frozen LM contributes zero trainable parameters.
No training happens here -- losses are computed only to get a gradient signal.
"""
import json
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import data_pipeline as dp
import splits as sp
from models import DDIModel, NegativeSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model_smoke_test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def param_counts(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_pair_batch(pairs, graphs, text_emb_cache, has_text, lm_dim, device):
    """pairs: list of (id_a, id_b). Returns the two graph batches and the two
    (text_embedding, has_text) tensor pairs, aligned by position."""
    data_a = [graphs[a] for a, _ in pairs]
    data_b = [graphs[b] for _, b in pairs]
    batch_a = Batch.from_data_list(data_a).to(device)
    batch_b = Batch.from_data_list(data_b).to(device)

    def text_tensors(ids):
        emb = torch.stack([
            torch.tensor(text_emb_cache[i], dtype=torch.float) if has_text.get(i, False)
            else torch.zeros(lm_dim, dtype=torch.float)
            for i in ids
        ]).to(device)
        flags = torch.tensor([1.0 if has_text.get(i, False) else 0.0 for i in ids], device=device)
        return emb, flags

    text_a, ht_a = text_tensors([a for a, _ in pairs])
    text_b, ht_b = text_tensors([b for _, b in pairs])
    return batch_a, text_a, ht_a, batch_b, text_b, ht_b


def check_grad_flow(model: DDIModel):
    def has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in module.parameters())
    report = {}
    if model.use_graph:
        report["graph_encoder_received_grad"] = has_grad(model.graph_encoder)
    if model.use_text:
        report["text_encoder_received_grad"] = has_grad(model.text_encoder)
    report["head_received_grad"] = has_grad(model.head)
    return report


def run_config(name, model, batch_args, labels, binary, device):
    model.to(device)
    out = model(*batch_args)
    if binary:
        loss = nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), labels.float())
    else:
        loss = nn.functional.cross_entropy(out, labels)
    model.zero_grad()
    loss.backward()

    total, trainable = param_counts(model)
    grad_report = check_grad_flow(model)
    result = {
        "config": name,
        "output_shape": list(out.shape),
        "loss": loss.item(),
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params_in_model": total - trainable,
        "grad_flow": grad_report,
    }
    log.info(f"[{name}] out={result['output_shape']} loss={result['loss']:.4f} "
              f"params(total/trainable)={total}/{trainable} grad_flow={grad_report}")
    return result


def main():
    with open(os.path.join(ROOT, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    sp.set_seed(model_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    log.info("=== loading cached smoke-test artifacts (built in step 3) ===")
    graphs = torch.load(os.path.join(ROOT, "data", "graph_cache_smoke.pt"), weights_only=False)
    with open(os.path.join(ROOT, "data", "text_cache_smoke.json")) as f:
        text_cache = json.load(f)
    has_text = {k: bool(v) for k, v in text_cache.items()}
    text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings_smoke.pt"), weights_only=False)
    text_emb_cache, lm_dim = text_blob["embeddings"], text_blob["dim"]
    log.info(f"{len(graphs)} cached graphs, {sum(has_text.values())}/{len(has_text)} drugs with text, lm_dim={lm_dim}")

    transductive = pd.read_csv(os.path.join(ROOT, "data", "splits", "smoke_drugbank_seed0", "transductive.csv"))
    train_rows = transductive[transductive["split"] == "train"].sample(16, random_state=0)
    multiclass_pairs = list(zip(train_rows["ID1"], train_rows["ID2"]))
    multiclass_labels = torch.tensor([dp.zero_indexed_label(y) for y in train_rows["Y"]])

    log.info("=== building binary (BioSNAP-style) smoke batch with negative sampling ===")
    chch = dp.load_chch_miner_binary(data_cfg["chch_miner_url"],
                                       dp.drug_smiles_table(dp.load_drugbank_pairs(data_cfg["drugbank_repo_raw"])))
    subset_drug_ids = list(graphs.keys())
    subset_drug_set = set(subset_drug_ids)
    chch_subset = chch[chch["d1"].isin(subset_drug_set) & chch["d2"].isin(subset_drug_set)]
    pos_pairs = list(zip(chch_subset["d1"], chch_subset["d2"]))[:8]
    sampler = NegativeSampler(subset_drug_ids, pos_pairs, seed=model_cfg["seed"],
                                ratio=model_cfg["negative_sample_ratio"])
    neg_pairs = sampler.sample(len(pos_pairs))
    binary_pairs = pos_pairs + neg_pairs
    binary_labels = torch.tensor([1.0] * len(pos_pairs) + [0.0] * len(neg_pairs))
    log.info(f"binary smoke batch: {len(pos_pairs)} positives + {len(neg_pairs)} negatives "
              f"(ratio={model_cfg['negative_sample_ratio']}:1, as configured)")

    mc_batch = build_pair_batch(multiclass_pairs, graphs, text_emb_cache, has_text, lm_dim, device)
    bin_batch = build_pair_batch(binary_pairs, graphs, text_emb_cache, has_text, lm_dim, device)
    multiclass_labels, binary_labels = multiclass_labels.to(device), binary_labels.to(device)

    common = dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=lm_dim,
                  hidden_dim=model_cfg["hidden_dim"], graph_layers=model_cfg["graph_layers"],
                  graph_conv=model_cfg["graph_conv"], graph_readout=model_cfg["graph_readout"],
                  head_hidden=model_cfg["head_hidden"], dropout=model_cfg["dropout"])

    results = []
    for fusion_type in ["concat", "gated", "cross_attention"]:
        model = DDIModel(fusion_type=fusion_type, use_graph=True, use_text=True,
                          num_classes=dp.NUM_INTERACTION_TYPES, binary=False, **common)
        results.append(run_config(f"fusion={fusion_type} (multiclass)", model, mc_batch,
                                    multiclass_labels, binary=False, device=device))

    model = DDIModel(fusion_type="concat", use_graph=True, use_text=False,
                      num_classes=dp.NUM_INTERACTION_TYPES, binary=False, **common)
    results.append(run_config("graph-only (multiclass)", model, mc_batch, multiclass_labels,
                                binary=False, device=device))

    model = DDIModel(fusion_type="concat", use_graph=False, use_text=True,
                      num_classes=dp.NUM_INTERACTION_TYPES, binary=False, **common)
    results.append(run_config("text-only (multiclass)", model, mc_batch, multiclass_labels,
                                binary=False, device=device))

    model = DDIModel(fusion_type="concat", use_graph=True, use_text=True, binary=True, **common)
    results.append(run_config("fusion=concat (binary, BioSNAP)", model, bin_batch, binary_labels,
                                binary=True, device=device))

    log.info("=== frozen LM verification ===")
    embedder_check = dp.TextEmbedder(data_cfg["text_model_name"], device=device)
    lm_total = sum(p.numel() for p in embedder_check.model.parameters())
    lm_trainable = sum(p.numel() for p in embedder_check.model.parameters() if p.requires_grad)
    fusion_model_param_ids = {id(p) for p in model.parameters()}
    lm_param_ids = {id(p) for p in embedder_check.model.parameters()}
    overlap = fusion_model_param_ids & lm_param_ids
    frozen_report = {
        "lm_total_params": lm_total, "lm_trainable_params": lm_trainable,
        "lm_params_with_requires_grad_true": sum(1 for p in embedder_check.model.parameters() if p.requires_grad),
        "lm_params_overlapping_with_ddimodel_optimizer": len(overlap),
    }
    log.info(f"frozen LM check: {frozen_report}")
    assert lm_trainable == 0, "frozen LM must have zero trainable params"
    assert len(overlap) == 0, "LM params must never appear in the DDIModel optimizer"

    out_path = os.path.join(ROOT, "results", "model_smoke_test_stats.json")
    with open(out_path, "w") as f:
        json.dump({"configs": results, "frozen_lm_check": frozen_report}, f, indent=2)
    log.info(f"wrote {out_path}")
    print(json.dumps({"configs": results, "frozen_lm_check": frozen_report}, indent=2))


if __name__ == "__main__":
    main()
