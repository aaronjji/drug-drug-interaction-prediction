"""Seed-controlled transductive + inductive (cold-start) splits, saved to disk
so every model/ablation sees identical splits. Also implements the
evaluable-class policy used by evaluate.py for macro-averaging.

Inductive drug partition and what each pair-bucket means:
  D_train, D_val, D_test partition the drug set.
  - train : both endpoints in D_train
  - val   : at least one endpoint in D_val, no endpoint in D_test
            (used for inductive early stopping; mirrors S1/S2 loosely)
  - s1    : exactly one endpoint in D_test, the other in D_train u D_val
  - s2    : both endpoints in D_test
Every pair falls into exactly one bucket; none are dropped.
"""
import json
import logging
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger("splits")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(ROOT, "data", "splits")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Transductive: split pairs, drugs may appear anywhere
# ---------------------------------------------------------------------------

def build_transductive_split(pairs_df: pd.DataFrame, label_col: Optional[str] = None,
                              val_frac: float = 0.1, test_frac: float = 0.2,
                              seed: int = 0, min_stratify_count: int = 3) -> pd.DataFrame:
    """Returns pairs_df with an added 'split' column in {train, val, test}.
    Stratifies by label_col via per-class proportional allocation (not sklearn's
    StratifiedShuffleSplit, which errors on classes with very few members once a
    long-tailed label distribution is split twice). Classes with fewer than
    min_stratify_count examples are forced into train and logged."""
    df = pairs_df.reset_index(drop=True).copy()
    n = len(df)
    rng = np.random.RandomState(seed)
    split = np.empty(n, dtype=object)

    if label_col is None:
        perm = rng.permutation(n)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        split[perm[:n_test]] = "test"
        split[perm[n_test:n_test + n_val]] = "val"
        split[perm[n_test + n_val:]] = "train"
    else:
        groups = df.groupby(label_col).indices
        rare_classes = []
        for label, idxs in groups.items():
            idxs = np.array(idxs)
            if len(idxs) < min_stratify_count:
                split[idxs] = "train"
                rare_classes.append(label)
                continue
            perm = rng.permutation(idxs)
            n_test = min(int(round(len(perm) * test_frac)), len(perm) - 1)
            n_val = min(int(round(len(perm) * val_frac)), len(perm) - 1 - n_test)
            split[perm[:n_test]] = "test"
            split[perm[n_test:n_test + n_val]] = "val"
            split[perm[n_test + n_val:]] = "train"
        if rare_classes:
            log.info(f"{len(rare_classes)} classes have < {min_stratify_count} examples; "
                     f"forcing all their pairs into train (cannot stratify): {rare_classes}")

    df["split"] = split
    log.info(f"transductive split: train={sum(split=='train')} val={sum(split=='val')} test={sum(split=='test')}")
    return df


# ---------------------------------------------------------------------------
# Inductive: split drugs, derive S1/S2 pair buckets
# ---------------------------------------------------------------------------

def build_inductive_drug_split(drug_ids: List[str], has_text: Dict[str, bool],
                                train_frac: float = 0.7, val_frac: float = 0.1, test_frac: float = 0.2,
                                seed: int = 0, stratify_by_text: bool = True
                                ) -> Tuple[Dict[str, Set[str]], Dict[str, float]]:
    """Splits drugs into train/val/test sets. If stratify_by_text, stratifies on
    has_text so text coverage is comparable across groups. Returns the split and
    a coverage report comparing naive-random vs the split actually used."""
    ids = list(drug_ids)
    text_flags = np.array([1 if has_text.get(i, False) else 0 for i in ids])

    def _split(stratify_arr):
        tr, temp = train_test_split(ids, test_size=(val_frac + test_frac), random_state=seed,
                                     stratify=stratify_arr)
        if stratify_arr is not None:
            temp_flags = np.array([1 if has_text.get(i, False) else 0 for i in temp])
            strat_temp = temp_flags
        else:
            strat_temp = None
        val_size = val_frac / (val_frac + test_frac)
        va, te = train_test_split(temp, test_size=(1 - val_size), random_state=seed, stratify=strat_temp)
        return set(tr), set(va), set(te)

    def _coverage(tr, va, te):
        def cov(group):
            if not group:
                return float("nan")
            return float(np.mean([has_text.get(i, False) for i in group]))
        return {"train": cov(tr), "val": cov(va), "test": cov(te)}

    naive_tr, naive_va, naive_te = _split(None)
    naive_cov = _coverage(naive_tr, naive_va, naive_te)

    if stratify_by_text:
        tr, va, te = _split(text_flags)
    else:
        tr, va, te = naive_tr, naive_va, naive_te
    final_cov = _coverage(tr, va, te)

    report = {"before_rebalance_coverage": naive_cov, "after_rebalance_coverage": final_cov,
              "stratified": stratify_by_text}
    log.info(f"inductive drug split coverage before rebalancing: {naive_cov}")
    log.info(f"inductive drug split coverage after rebalancing:  {final_cov}")
    return {"train": tr, "val": va, "test": te}, report


def derive_inductive_pairs(pairs_df: pd.DataFrame, id1_col: str, id2_col: str,
                            drug_split: Dict[str, Set[str]]) -> Dict[str, pd.DataFrame]:
    d_train, d_val, d_test = drug_split["train"], drug_split["val"], drug_split["test"]

    def bucket(row):
        a, b = row[id1_col], row[id2_col]
        a_test, b_test = a in d_test, b in d_test
        if a_test and b_test:
            return "s2"
        if a_test or b_test:
            return "s1"
        a_val, b_val = a in d_val, b in d_val
        if a_val or b_val:
            return "val"
        return "train"

    buckets = pairs_df.apply(bucket, axis=1)
    out = {}
    for name in ["train", "val", "s1", "s2"]:
        out[name] = pairs_df[buckets == name].reset_index(drop=True)
    for name, df in out.items():
        log.info(f"inductive bucket '{name}': {len(df):,} pairs")
    return out


# ---------------------------------------------------------------------------
# Evaluable-class policy
# ---------------------------------------------------------------------------

def evaluable_classes(train_labels: pd.Series, eval_labels: pd.Series, k: int = 10
                       ) -> Tuple[Set, Dict]:
    """Classes with >= k examples in BOTH train_labels and eval_labels.
    Never silently produces NaN/zero metrics for empty classes -- callers must
    exclude classes not in the returned set, and the returned info dict logs why."""
    train_counts = train_labels.value_counts()
    eval_counts = eval_labels.value_counts()
    all_classes = set(train_counts.index) | set(eval_counts.index)
    evaluable = {c for c in all_classes if train_counts.get(c, 0) >= k and eval_counts.get(c, 0) >= k}
    excluded = all_classes - evaluable
    info = {
        "n_total_classes": len(all_classes),
        "n_evaluable": len(evaluable),
        "n_excluded": len(excluded),
        "excluded_classes": sorted(excluded, key=lambda c: -train_counts.get(c, 0))[:20],
        "min_k": k,
    }
    log.info(f"evaluable classes: {info['n_evaluable']}/{info['n_total_classes']} "
             f"(excluded {info['n_excluded']} with <{k} examples in train or eval split)")
    return evaluable, info


def compute_evaluable_classes_for_splits(transductive_df: pd.DataFrame, inductive: Dict[str, pd.DataFrame],
                                          label_col: str, k: int) -> Dict[str, dict]:
    """Computes the evaluable-class set for every non-train eval bucket against
    its own training labels, exactly once. This must be saved (save_evaluable_classes)
    and reused verbatim at eval time -- never recomputed ad hoc per call site,
    or different code paths could silently disagree on which classes count."""
    out = {}
    train_labels_trans = transductive_df.loc[transductive_df["split"] == "train", label_col]
    for name in ("val", "test"):
        eval_labels = transductive_df.loc[transductive_df["split"] == name, label_col]
        classes, info = evaluable_classes(train_labels_trans, eval_labels, k=k)
        out[f"transductive_{name}"] = {"classes": sorted(classes), "info": info}

    train_labels_ind = inductive["train"][label_col]
    for name in ("val", "s1", "s2"):
        classes, info = evaluable_classes(train_labels_ind, inductive[name][label_col], k=k)
        out[f"inductive_{name}"] = {"classes": sorted(classes), "info": info}
    return out


def save_evaluable_classes(out_dir: str, evaluable: Dict[str, dict]) -> None:
    with open(os.path.join(out_dir, "evaluable_classes.json"), "w") as f:
        json.dump(evaluable, f, indent=2)
    log.info(f"saved evaluable-class sets to {out_dir}/evaluable_classes.json")


def load_evaluable_classes(out_dir: str) -> Dict[str, Set]:
    with open(os.path.join(out_dir, "evaluable_classes.json")) as f:
        raw = json.load(f)
    return {k: set(v["classes"]) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_splits(seed: int, transductive: pd.DataFrame, drug_split: Dict[str, Set[str]],
                 inductive: Dict[str, pd.DataFrame], tag: str = "drugbank") -> str:
    out_dir = os.path.join(SPLITS_DIR, f"{tag}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    transductive.to_csv(os.path.join(out_dir, "transductive.csv"), index=False)
    with open(os.path.join(out_dir, "inductive_drugs.json"), "w") as f:
        json.dump({k: sorted(v) for k, v in drug_split.items()}, f)
    for name, df in inductive.items():
        df.to_csv(os.path.join(out_dir, f"inductive_{name}.csv"), index=False)
    log.info(f"saved splits to {out_dir}")
    return out_dir


def load_splits(seed: int, tag: str = "drugbank") -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, pd.DataFrame]]:
    in_dir = os.path.join(SPLITS_DIR, f"{tag}_seed{seed}")
    transductive = pd.read_csv(os.path.join(in_dir, "transductive.csv"))
    with open(os.path.join(in_dir, "inductive_drugs.json")) as f:
        drug_split = {k: set(v) for k, v in json.load(f).items()}
    inductive = {name: pd.read_csv(os.path.join(in_dir, f"inductive_{name}.csv"))
                 for name in ["train", "val", "s1", "s2"]}
    return transductive, drug_split, inductive
