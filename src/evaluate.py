"""Metrics for binary and multi-class DDI prediction, restricted to the
evaluable-class policy from splits.py, plus the transductive-vs-inductive
gap computation and standalone sanity checks for this module itself.

These numbers become the paper's results table -- correctness here matters
more than anywhere else in the pipeline. No metric is ever silently NaN or
zero-filled: degenerate or empty cases raise, not fabricate.
"""
import logging
import math
from typing import Dict, Set

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

log = logging.getLogger("evaluate")

METRIC_NAMES = ("auroc", "auprc", "f1", "accuracy")


def _assert_no_nan(metrics: dict) -> None:
    bad = {k: v for k, v in metrics.items() if isinstance(v, float) and math.isnan(v)}
    if bad:
        raise ValueError(f"NaN in reported metrics, refusing to report: {bad}")


def compute_binary_metrics(y_true, y_score) -> Dict[str, float]:
    """Standard binary metrics. y_score is a probability (post-sigmoid)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    _assert_no_nan(metrics)
    return metrics


def compute_multiclass_metrics(y_true, y_prob: np.ndarray, evaluable_classes: Set[int]) -> Dict[str, float]:
    """Macro-averages AUROC/AUPRC/recall over evaluable_classes ONLY (the
    >=k-in-both-train-and-eval policy persisted by splits.py -- pass in the
    exact set loaded from evaluable_classes.json, never a set recomputed here).

    F1 uses sklearn's label-restricted macro F1 directly (labels=evaluable,
    average='macro'), which is the standard definition of macro-F1 over a
    class subset.

    'accuracy' is the macro-averaged per-class recall (balanced accuracy)
    over evaluable classes -- plain global accuracy is not well-defined once
    classes are excluded from evaluation, so this is the explicit, documented
    choice rather than an implicit one.

    Classes in `evaluable_classes` that turn out degenerate in THIS eval set
    (zero positives or zero negatives, despite passing the >=k policy at
    split-build time -- can't happen by construction but checked anyway) are
    skipped and logged, never zero-filled."""
    y_true = np.asarray(y_true)
    y_pred = y_prob.argmax(axis=1)
    evaluable = sorted(evaluable_classes)
    if not evaluable:
        raise ValueError("no evaluable classes -- cannot compute multiclass metrics")

    aurocs, auprcs, recalls = [], [], []
    skipped = []
    for c in evaluable:
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            skipped.append(c)
            continue
        aurocs.append(roc_auc_score(y_bin, y_prob[:, c]))
        auprcs.append(average_precision_score(y_bin, y_prob[:, c]))
        recalls.append(float((y_pred[y_true == c] == c).mean()))
    if skipped:
        log.warning(f"skipped {len(skipped)} evaluable classes degenerate in this eval set: {skipped}")
    if not aurocs:
        raise ValueError("all evaluable classes were degenerate in this eval set -- cannot compute metrics")

    f1 = float(f1_score(y_true, y_pred, labels=evaluable, average="macro", zero_division=0))
    metrics = {
        "auroc": float(np.mean(aurocs)),
        "auprc": float(np.mean(auprcs)),
        "f1": f1,
        "accuracy": float(np.mean(recalls)),
        "n_evaluable_classes_used": len(aurocs),
        "n_evaluable_classes_skipped": len(skipped),
    }
    _assert_no_nan({k: v for k, v in metrics.items() if k in METRIC_NAMES})
    return metrics


def transductive_inductive_gap(metrics_by_split: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """transductive-minus-inductive gap per metric -- the paper's headline
    number. Computed once here, as a first-class field of every run's output,
    not derived by hand later from a table."""
    gap = {}
    trans = metrics_by_split.get("transductive")
    if trans is None:
        return gap
    for split_name in ("s1", "s2"):
        other = metrics_by_split.get(split_name)
        if other is None:
            continue
        gap[f"transductive_minus_{split_name}"] = {
            m: trans[m] - other[m] for m in METRIC_NAMES if m in trans and m in other
        }
    return gap


# ---------------------------------------------------------------------------
# Sanity checks for this module (not for any trained model)
# ---------------------------------------------------------------------------

def _sanity_binary() -> Dict[str, dict]:
    rng = np.random.RandomState(0)
    n = 1000
    y_true = rng.randint(0, 2, n)

    y_score_perfect = y_true.astype(float)
    perfect = compute_binary_metrics(y_true, y_score_perfect)
    perfect_pass = perfect["auroc"] > 0.999 and perfect["auprc"] > 0.999

    y_score_random = rng.rand(n)
    random_metrics = compute_binary_metrics(y_true, y_score_random)
    random_pass = abs(random_metrics["auroc"] - 0.5) < 0.05

    return {
        "binary_perfect": {"metrics": perfect, "pass": bool(perfect_pass)},
        "binary_random": {"metrics": random_metrics, "pass": bool(random_pass)},
    }


def _sanity_multiclass(num_classes: int = 10, n_per_class: int = 50, seed: int = 0) -> Dict[str, dict]:
    rng = np.random.RandomState(seed)
    classes = list(range(num_classes))
    y_true = np.repeat(classes, n_per_class)
    rng.shuffle(y_true)
    n = len(y_true)
    evaluable = set(classes)

    y_prob_perfect = np.full((n, num_classes), 1e-6)
    y_prob_perfect[np.arange(n), y_true] = 1.0
    y_prob_perfect = y_prob_perfect / y_prob_perfect.sum(axis=1, keepdims=True)
    perfect = compute_multiclass_metrics(y_true, y_prob_perfect, evaluable)
    perfect_pass = all(perfect[m] > 0.999 for m in METRIC_NAMES)

    raw = rng.rand(n, num_classes)
    y_prob_random = raw / raw.sum(axis=1, keepdims=True)
    random_metrics = compute_multiclass_metrics(y_true, y_prob_random, evaluable)
    chance_acc = 1.0 / num_classes
    random_pass = (abs(random_metrics["accuracy"] - chance_acc) < 0.05
                   and abs(random_metrics["auroc"] - 0.5) < 0.05)

    return {
        "multiclass_perfect": {"metrics": perfect, "pass": bool(perfect_pass)},
        "multiclass_random": {"metrics": random_metrics, "pass": bool(random_pass)},
    }


def run_sanity_checks() -> (Dict[str, dict], bool):
    results = {}
    results.update(_sanity_binary())
    results.update(_sanity_multiclass())
    all_pass = all(v["pass"] for v in results.values())
    log.info(f"evaluate.py sanity checks: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    for name, v in results.items():
        log.info(f"  {name}: pass={v['pass']}")
    return results, all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, ok = run_sanity_checks()
    raise SystemExit(0 if ok else 1)
