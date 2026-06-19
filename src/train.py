"""Training loop shared by every model/ablation/dataset combination. Config-
driven, seeded, early-stopped on validation AUPRC, every run appended to
results/run_log.txt and results/runs/*.json (never overwritten).
"""
import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import data_pipeline as dp
import evaluate as ev
import splits as sp
from models import DDIModel

log = logging.getLogger("train")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
RUN_LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# Relaxed k used ONLY for the internal val-AUPRC early-stopping signal when the
# official (config) k leaves zero evaluable classes in val on a tiny subset.
# Never used for transductive_test/s1/s2 -- those always use the persisted,
# official evaluable_classes.json verbatim.
VAL_FALLBACK_K = 2


def iterate_minibatches(pairs_df: pd.DataFrame, id1_col: str, id2_col: str, label_col: str,
                         graphs, text_emb_cache, has_text, lm_dim: int, batch_size: int,
                         device: str, shuffle: bool, seed: int):
    idx = np.arange(len(pairs_df))
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sel = idx[start:start + batch_size]
        rows = pairs_df.iloc[sel]
        pairs = list(zip(rows[id1_col], rows[id2_col]))
        batch = dp.build_pair_batch(pairs, graphs, text_emb_cache, has_text, lm_dim, device)
        labels = rows[label_col].values
        yield batch, labels


def _predict_split(model, df, id1_col, id2_col, label_col, graphs, text_emb_cache, has_text,
                    lm_dim, binary, device, batch_size):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch, labels in iterate_minibatches(df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                                                   has_text, lm_dim, batch_size, device, shuffle=False, seed=0):
            out = model(*batch)
            probs = torch.sigmoid(out.squeeze(-1)) if binary else torch.softmax(out, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels


def evaluate_split(model, df, id1_col, id2_col, label_col, graphs, text_emb_cache, has_text,
                    lm_dim, binary, device, batch_size, evaluable_classes=None):
    probs, labels = _predict_split(model, df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                                    has_text, lm_dim, binary, device, batch_size)
    if binary:
        return ev.compute_binary_metrics(labels, probs)
    return ev.compute_multiclass_metrics(np.asarray(labels), probs, evaluable_classes)


def _val_metrics_for_monitoring(model, val_df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                                 has_text, lm_dim, binary, device, batch_size, val_evaluable, train_labels):
    """Validation AUPRC used only for early stopping. Falls back to a relaxed k
    if the official val evaluable-class set is empty on a tiny subset (logged,
    never silently used for a reported metric)."""
    if binary:
        return evaluate_split(model, val_df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                               has_text, lm_dim, binary, device, batch_size)
    if val_evaluable:
        return evaluate_split(model, val_df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                               has_text, lm_dim, binary, device, batch_size, val_evaluable)
    relaxed, _ = sp.evaluable_classes(train_labels, val_df[label_col], k=VAL_FALLBACK_K)
    if relaxed:
        log.warning(f"val evaluable-class set empty at official k; using relaxed k={VAL_FALLBACK_K} "
                    f"for early-stopping monitoring only ({len(relaxed)} classes)")
        return evaluate_split(model, val_df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                               has_text, lm_dim, binary, device, batch_size, relaxed)
    log.warning("no evaluable val classes even at relaxed k -- falling back to train loss for early stopping")
    return None


def train_one_run(run_name: str, model_kwargs: dict, pairs_splits: Dict[str, pd.DataFrame],
                   id1_col: str, id2_col: str, label_col: str, binary: bool,
                   graphs, text_emb_cache, has_text, lm_dim: int, train_cfg: dict,
                   evaluable_classes_by_split: Dict[str, set], device: str,
                   num_classes: Optional[int] = None) -> dict:
    """pairs_splits must contain 'train' and 'val', plus one or more eval splits
    (e.g. {'test': df} for transductive, or {'s1': df, 's2': df} for inductive)."""
    sp.set_seed(train_cfg["seed"])
    model = DDIModel(num_classes=num_classes, binary=binary, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    opt_param_count = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert opt_param_count == trainable_param_count, (
        f"optimizer wraps {opt_param_count} params but model has {trainable_param_count} trainable -- "
        f"frozen params leaking into the optimizer (or vice versa)")
    log.info(f"[{run_name}] optimizer verified: exactly {opt_param_count} trainable params, no more/less")

    train_df, val_df = pairs_splits["train"], pairs_splits["val"]
    best_score, best_state, epochs_without_improve = -1e18, None, 0
    history = []
    t0 = time.time()
    using_loss_fallback = False

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch, labels in iterate_minibatches(train_df, id1_col, id2_col, label_col, graphs, text_emb_cache,
                                                   has_text, lm_dim, train_cfg["batch_size"], device,
                                                   shuffle=True, seed=train_cfg["seed"] + epoch):
            optimizer.zero_grad()
            out = model(*batch)
            if binary:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out.squeeze(-1), torch.tensor(labels, dtype=torch.float, device=device))
            else:
                target = torch.tensor(labels, dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        val_metrics = _val_metrics_for_monitoring(
            model, val_df, id1_col, id2_col, label_col, graphs, text_emb_cache, has_text, lm_dim,
            binary, device, train_cfg["batch_size"], evaluable_classes_by_split.get("val"), train_df[label_col])
        if val_metrics is not None:
            score = val_metrics["auprc"]
            using_loss_fallback = False
        else:
            score = -train_loss
            using_loss_fallback = True

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_score": score, "using_loss_fallback": using_loss_fallback})
        log.info(f"[{run_name}] epoch {epoch}: train_loss={train_loss:.4f} "
                 f"val_score={score:.4f}{' (loss fallback)' if using_loss_fallback else ' (val auprc)'}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= train_cfg["patience"]:
                log.info(f"[{run_name}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    wall_clock = time.time() - t0

    metrics_by_split = {}
    for split_name, df in pairs_splits.items():
        if split_name in ("train", "val"):
            continue
        metrics_by_split[split_name] = evaluate_split(
            model, df, id1_col, id2_col, label_col, graphs, text_emb_cache, has_text, lm_dim,
            binary, device, train_cfg["batch_size"], evaluable_classes_by_split.get(split_name))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.pt")
    torch.save(best_state, ckpt_path)

    result = {
        "run_name": run_name, "best_val_score": best_score, "used_loss_fallback": using_loss_fallback,
        "wall_clock_sec": wall_clock, "epochs_run": len(history), "history": history,
        "metrics_by_split": metrics_by_split, "checkpoint": ckpt_path,
    }
    # NOTE: transductive and inductive are necessarily separate training runs
    # (different train sets), so a single run's metrics_by_split never holds
    # both a 'transductive' and an 's1'/'s2' key -- the gap must be computed
    # by the caller once both runs for a given model exist (see
    # evaluate.transductive_inductive_gap, called from the orchestrating script).
    log.info(f"[{run_name}] done in {wall_clock:.1f}s, {len(history)} epochs, "
              f"metrics={ {k: {m: round(v, 4) for m, v in mv.items() if m in ev.METRIC_NAMES} for k, mv in metrics_by_split.items()} }")
    return result


def log_run(run_result: dict, config: dict) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = f"{run_result['run_name']}_{int(time.time() * 1000)}"
    with open(os.path.join(RUNS_DIR, f"{run_id}.json"), "w") as f:
        json.dump({"config": config, "result": run_result}, f, indent=2, default=str)
    with open(RUN_LOG_PATH, "a") as f:
        f.write(f"=== {run_id} ===\n")
        f.write(f"config: {json.dumps(config, default=str)}\n")
        for split, m in run_result["metrics_by_split"].items():
            f.write(f"  {split}: {json.dumps(m, default=str)}\n")
        if "transductive_inductive_gap" in run_result:
            f.write(f"  gap: {json.dumps(run_result['transductive_inductive_gap'], default=str)}\n")
        f.write(f"wall_clock_sec={run_result['wall_clock_sec']:.1f} epochs_run={run_result['epochs_run']}\n\n")
    log.info(f"logged run {run_id} to {RUN_LOG_PATH} and {RUNS_DIR}/{run_id}.json")
    return run_id
