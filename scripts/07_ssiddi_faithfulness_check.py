"""Stage A faithfulness check for the SSI-DDI literature baseline (brief
section 7). Downloads the original kanz76/SSI-DDI repo + data (gitignored
cache under external/, not redistributed), then trains/evaluates THEIR
verbatim model, loss, data loader, and pos-vs-corrupted-negative metric
definition on THEIR OWN train/val/test split -- no adaptation to our harness
yet. This is purely a correctness gate: does my reproduction land in a
plausible range of the published numbers?

If this does not reproduce convincingly, the brief says STOP before running
the inductive sweep -- a broken reimplementation makes a misleading baseline.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import optim
from torch.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ssiddi_faithfulness")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT, "external", "ssiddi_orig")
REPO_RAW = "https://raw.githubusercontent.com/kanz76/SSI-DDI/master"
CODE_FILES = ["models.py", "layers.py", "custom_loss.py", "data_preprocessing.py"]
DATA_FILES = ["ddi_training.csv", "ddi_validation.csv", "ddi_test.csv", "drug_smiles.csv",
              "ddis.csv", "Interaction_information.csv"]

MAX_EPOCHS = 300  # their published default (extended from the 100-epoch Stage A run, which was still climbing)
PATIENCE = 25  # plateau guardrail: stop if no real improvement for this many epochs
MIN_IMPROVEMENT = 0.001  # an epoch only counts as "improved" if it beats best by more than this
BATCH_SIZE = 1024
LR = 1e-2
WEIGHT_DECAY = 5e-4
CHECKPOINT_PATH = os.path.join(ROOT, "results", "checkpoints", "ssiddi_faithfulness_latest.pt")


def ensure_repo_cached():
    import requests
    os.makedirs(os.path.join(EXT_DIR, "data"), exist_ok=True)
    for f in CODE_FILES:
        path = os.path.join(EXT_DIR, f)
        if not os.path.exists(path):
            log.info(f"downloading {f}")
            open(path, "wb").write(requests.get(f"{REPO_RAW}/{f}", timeout=60).content)
    for f in DATA_FILES:
        path = os.path.join(EXT_DIR, "data", f)
        if not os.path.exists(path):
            log.info(f"downloading data/{f}")
            open(path, "wb").write(requests.get(f"{REPO_RAW}/data/{f}", timeout=120).content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision (matches original FP32 training exactly)")
    parser.add_argument("--checkpoint-tag", type=str, default="latest", help="separate checkpoint file, for side-by-side diagnostic runs")
    args = parser.parse_args()
    checkpoint_path = os.path.join(ROOT, "results", "checkpoints", f"ssiddi_faithfulness_{args.checkpoint_tag}.pt")

    ensure_repo_cached()
    cwd = os.getcwd()
    os.chdir(EXT_DIR)  # their data_preprocessing.py uses relative paths ('data/...')
    sys.path.insert(0, EXT_DIR)
    try:
        import custom_loss
        import data_preprocessing as dpp
        import models
        import pandas as pd

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"device={device} TOTAL_ATOM_FEATS={dpp.TOTAL_ATOM_FEATS} "
                 f"drugs_with_graph={len(dpp.MOL_EDGE_LIST_FEAT_MTX)}")

        df_train = pd.read_csv("data/ddi_training.csv")
        df_val = pd.read_csv("data/ddi_validation.csv")
        df_test = pd.read_csv("data/ddi_test.csv")
        log.info(f"rows: train={len(df_train)} val={len(df_val)} test={len(df_test)}")

        train_tup = list(zip(df_train["d1"], df_train["d2"], df_train["type"]))
        val_tup = list(zip(df_val["d1"], df_val["d2"], df_val["type"]))
        test_tup = list(zip(df_test["d1"], df_test["d2"], df_test["type"]))

        train_data = dpp.DrugDataset(train_tup, neg_ent=1)
        val_data = dpp.DrugDataset(val_tup, disjoint_split=False)
        test_data = dpp.DrugDataset(test_tup, disjoint_split=False)
        log.info(f"DrugDataset sizes: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

        train_loader = dpp.DrugDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = dpp.DrugDataLoader(val_data, batch_size=BATCH_SIZE * 3)
        test_loader = dpp.DrugDataLoader(test_data, batch_size=BATCH_SIZE * 3)

        model = models.SSI_DDI(dpp.TOTAL_ATOM_FEATS, 64, 64, 86,
                                heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2]).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"SSI_DDI instantiated, {n_params:,} params (faithful to published hyperparameters)")

        loss_fn = custom_loss.SigmoidLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)
        use_amp = (device == "cuda") and not args.no_amp
        scaler = GradScaler(device="cuda", enabled=use_amp)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def do_compute(batch):
            pos_tri, neg_tri = batch
            pos_tri = [t.to(device) for t in pos_tri]
            neg_tri = [t.to(device) for t in neg_tri]
            with autocast(device_type="cuda", enabled=use_amp):
                p_score = model(pos_tri)
                n_score = model(neg_tri)
            probas = np.concatenate([torch.sigmoid(p_score.detach()).float().cpu().numpy(),
                                      torch.sigmoid(n_score.detach()).float().cpu().numpy()])
            truth = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])
            return p_score, n_score, probas, truth

        def compute_metrics(probas, truth):
            pred = (probas >= 0.5).astype(int)
            acc = metrics.accuracy_score(truth, pred)
            auc_roc = metrics.roc_auc_score(truth, probas)
            f1 = metrics.f1_score(truth, pred)
            p, r, _ = metrics.precision_recall_curve(truth, probas)
            auc_prc = metrics.auc(r, p)
            return acc, auc_roc, auc_prc, f1

        best_val_auc, best_state, no_improve = -1.0, None, 0
        history = []
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_auc, best_state = ckpt["best_val_auc"], ckpt["best_state"]
            no_improve, history = ckpt["no_improve"], ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            log.info(f"resumed from checkpoint at epoch {start_epoch}, best_val_auc so far={best_val_auc:.4f}")

        t0 = time.time()
        stopped_early = False
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                p_score, n_score, _, _ = do_compute(batch)
                loss, _, _ = loss_fn(p_score, n_score)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(p_score)
            train_loss /= len(train_data)
            scheduler.step()

            model.eval()
            val_probas, val_truth = [], []
            with torch.no_grad():
                for batch in val_loader:
                    _, _, probas, truth = do_compute(batch)
                    val_probas.append(probas)
                    val_truth.append(truth)
            val_probas, val_truth = np.concatenate(val_probas), np.concatenate(val_truth)
            val_acc, val_auc, val_auprc, val_f1 = compute_metrics(val_probas, val_truth)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                             "val_auroc": val_auc, "val_auprc": val_auprc})
            log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                     f"val_auroc={val_auc:.4f} val_auprc={val_auprc:.4f}")

            if val_auc > best_val_auc + MIN_IMPROVEMENT:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(), "epoch": epoch,
                        "best_val_auc": best_val_auc, "best_state": best_state,
                        "no_improve": no_improve, "history": history}, checkpoint_path)

            if no_improve >= PATIENCE:
                log.info(f"plateau guardrail: no improvement > {MIN_IMPROVEMENT} for {PATIENCE} epochs, "
                         f"stopping at epoch {epoch}")
                stopped_early = True
                break

        wall_clock = time.time() - t0
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
        model.load_state_dict(best_state)

        model.eval()
        test_probas, test_truth = [], []
        with torch.no_grad():
            for batch in test_loader:
                _, _, probas, truth = do_compute(batch)
                test_probas.append(probas)
                test_truth.append(truth)
        test_probas, test_truth = np.concatenate(test_probas), np.concatenate(test_truth)
        test_acc, test_auc, test_auprc, test_f1 = compute_metrics(test_probas, test_truth)

        log.info(f"=== FAITHFULNESS CHECK RESULT (extended run) ===")
        log.info(f"reproduced (this run):  acc={test_acc:.4f} auroc={test_auc:.4f} auprc={test_auprc:.4f} f1={test_f1:.4f}")
        log.info(f"Stage A (100-epoch run): acc=0.8802 auroc=0.9438 auprc=0.9324 f1=0.8824")
        log.info(f"published range found via web search (unverified exact decimal, paywalled paper): "
                 f"acc ~0.94-0.96, auroc ~0.98-0.99 (multiple inconsistent sources, see report)")
        log.info(f"wall_clock_sec={wall_clock:.1f} epochs_run={epoch+1} stopped_early={stopped_early} "
                 f"peak_gpu_mem_mb={peak_mem_mb}")

        import json
        out = {"test_acc": test_acc, "test_auroc": test_auc, "test_auprc": test_auprc, "test_f1": test_f1,
               "wall_clock_sec": wall_clock, "epochs_run": epoch + 1, "stopped_early": stopped_early,
               "peak_gpu_mem_mb": peak_mem_mb, "n_params": n_params, "history": history,
               "stage_a_100_epoch": {"test_acc": 0.8802, "test_auroc": 0.9438, "test_auprc": 0.9324, "test_f1": 0.8824},
               "repo_commit": "23e958f65f59f689edea8d68a38f944c6d18dc96"}
        out_name = f"ssiddi_faithfulness_check_{args.checkpoint_tag}.json"
        with open(os.path.join(ROOT, "results", out_name), "w") as fh:
            json.dump(out, fh, indent=2)
        log.info(f"wrote results/{out_name}")
    finally:
        os.chdir(cwd)
        sys.path.remove(EXT_DIR)


if __name__ == "__main__":
    main()
