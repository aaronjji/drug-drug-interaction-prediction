"""Camera-ready experiments requested in the AI4DD (NeurIPS 2026 workshop)
reviews, run through the unchanged train.py/evaluate.py harness on the
persisted seed-0 splits -- the same splits and evaluable-class sets as
scripts 05/06, training seeds 0/1/2:

  fp-text          ECFP-MLP + frozen-text fusion (concat), DrugBank and BioSNAP,
                   transductive and inductive (S1/S2)          [reviewers SM1A, yfbP]
  gcn              GCN instead of GIN, graph-only and fusion=concat,
                   DrugBank inductive (S1/S2)                  [SM1A]
  fusion-variants  fusion=gated and fusion=cross_attention,
                   DrugBank inductive (S1/S2)                  [tFTz]
  filtered-text    fusion=concat and text-only on interaction-filtered
                   descriptions (scripts/12_text_leakage_screen.py),
                   DrugBank inductive (S1/S2)                  [SM1A, tFTz]
  filtered-generic same, but removing only explicit interaction wording
                   (co-administration, CYP450, P-gp, ...; --level generic),
                   which keeps pharmacological roles   [SM1A, tFTz]

Every run name starts with "cr_", so no run here can resume from, or
overwrite, a checkpoint of the original sweeps, and the original
results/*_stats.json files are never touched; 15_camera_ready_tables.py
aggregates results/runs/cr_*.json afterwards.

Jobs are claimed through exclusive lock files in results/cr_queue/, so several
worker processes can share one GPU (peak memory per run is tens of MB). A
finished job leaves a .done marker and is skipped on relaunch; a failed one
leaves a .failed traceback. To retry an interrupted job, delete its .claim
file -- train.py then resumes it from its *_latest.pt checkpoint.

BioSNAP inductive negatives come from scripts/06's own sampler, but that
sampler draws from Python sets of drug ids, whose iteration order depends on
the per-process string-hash seed; launch workers with PYTHONHASHSEED=0 so the
new draws are reproducible (the original sweep's exact inductive draw cannot
be recreated -- transductive negatives are unaffected, they use a sorted list).

Usage:
  python scripts/13_camera_ready_experiments.py --smoke --exps fp-text,gcn,fusion-variants
  PYTHONHASHSEED=0 python scripts/13_camera_ready_experiments.py --exps fp-text,gcn,fusion-variants --worker 0
"""
import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import baselines as bl
import data_pipeline as dp
import evaluate as ev
import experiment_utils as eu
import splits as sp
import train as tr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera_ready_experiments")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE_DIR = os.path.join(ROOT, "results", "cr_queue")
# filter level -> file suffix written by 12_text_leakage_screen.py --level
FILTER_SUFFIX = {"filtered": "", "generic": "_generic"}
SEEDS = (0, 1, 2)
SPLIT_SEED = 0

# (dataset, arch, regime) per experiment; arch names are what appears in run names
EXPERIMENTS = {
    "fp-text": [("drugbank", "fp-text", "transductive"), ("drugbank", "fp-text", "inductive"),
                ("biosnap", "fp-text", "transductive"), ("biosnap", "fp-text", "inductive")],
    "gcn": [("drugbank", "gcn-graph-only", "inductive"), ("drugbank", "gcn-fusion=concat", "inductive")],
    "fusion-variants": [("drugbank", "fusion=gated", "inductive"),
                        ("drugbank", "fusion=cross_attention", "inductive")],
    "filtered-text": [("drugbank", "filtered-fusion=concat", "inductive"),
                      ("drugbank", "filtered-text-only", "inductive")],
    "filtered-generic": [("drugbank", "generic-fusion=concat", "inductive"),
                         ("drugbank", "generic-text-only", "inductive")],
}

# rough single-run minutes on the original 4GB T550, from results/run_log.txt;
# only used to schedule the longest jobs first
EXPECTED_MINUTES = {
    ("drugbank", "transductive"): 40, ("drugbank", "inductive"): 22,
    ("biosnap", "transductive"): 6, ("biosnap", "inductive"): 4,
}


def run_name_for(dataset, arch, regime, seed, prefix="cr"):
    return f"{prefix}_{dataset}_{arch}_{regime}_seed{seed}"


def build_jobs(exps):
    jobs = [(dataset, arch, regime, seed)
            for exp in exps for dataset, arch, regime in EXPERIMENTS[exp] for seed in SEEDS]
    # longest first, then by seed so the three seeds of one config spread across workers
    return sorted(jobs, key=lambda j: (-EXPECTED_MINUTES[(j[0], j[2])], j[3], j[1]))


@contextmanager
def file_lock(path, timeout_sec=120):
    """Cross-process mutex via O_EXCL lock file; guards the shared append-only
    results/run_log.txt so concurrent workers never interleave its lines."""
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_sec:
                log.warning(f"breaking stale lock {path}")
                os.remove(path)
            time.sleep(0.1)
    try:
        yield
    finally:
        os.remove(path)


def try_claim(run_name, worker):
    if os.path.exists(os.path.join(QUEUE_DIR, f"{run_name}.done")):
        return False
    try:
        fd = os.open(os.path.join(QUEUE_DIR, f"{run_name}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"worker={worker} pid={os.getpid()} claimed={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return True


def keep_system_awake():
    """Windows only: ask the OS not to idle-sleep while this process runs (the
    request lapses automatically when the process exits). Lid-close or
    battery-critical sleep still apply."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("requested no idle sleep for the lifetime of this worker")


def load_module_from_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(ROOT, "scripts", filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    """Lazily loaded, per-worker shared artifacts (identical files to 05/06)."""

    def __init__(self, device, smoke):
        self.device = device
        self.smoke = smoke
        with open(os.path.join(ROOT, "configs", "model.yaml")) as f:
            self.model_cfg = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs", "train_full.yaml")) as f:
            self.train_cfg_base = yaml.safe_load(f)
        assert self.model_cfg["seed"] == SPLIT_SEED
        self.graphs = torch.load(os.path.join(ROOT, "data", "graph_cache.pt"), weights_only=False)
        with open(os.path.join(ROOT, "data", "text_cache.json"), encoding="utf-8") as f:
            text_cache = json.load(f)
        self.has_text = {k: bool(v) for k, v in text_cache.items()}
        text_blob = torch.load(os.path.join(ROOT, "data", "text_embeddings.pt"), weights_only=False)
        self.text_emb, self.lm_dim = text_blob["embeddings"], text_blob["dim"]
        self.fp_cache = torch.load(os.path.join(ROOT, "data", "fingerprint_cache.pt"), weights_only=False)
        self._drugbank = None
        self._biosnap = None
        self._biosnap_negatives = {}
        self._filtered = {}

    def common(self, graph_conv=None):
        m = self.model_cfg
        return dict(atom_dim=dp.ATOM_FEATURE_DIM, edge_dim=dp.BOND_FEATURE_DIM, lm_dim=self.lm_dim,
                    hidden_dim=m["hidden_dim"], graph_layers=m["graph_layers"],
                    graph_conv=graph_conv or m["graph_conv"], graph_readout=m["graph_readout"],
                    head_hidden=m["head_hidden"], dropout=m["dropout"])

    def drugbank(self):
        if self._drugbank is None:
            transductive, drug_split, inductive = sp.load_splits(SPLIT_SEED, tag="drugbank")
            evaluable = sp.load_evaluable_classes(os.path.join(ROOT, "data", "splits", f"drugbank_seed{SPLIT_SEED}"))
            self._drugbank = (transductive, drug_split, inductive, evaluable)
        return self._drugbank

    def biosnap(self, seed):
        """Fixed seed-0 ChCh-Miner split as saved by 06, with that script's own
        seeded, bucket-aware negative sampling for this training seed."""
        if self._biosnap is None:
            transductive, drug_split, _ = sp.load_splits(SPLIT_SEED, tag="chch")
            drug_ids = sorted(set(transductive["d1"]) | set(transductive["d2"]))
            sweep06 = load_module_from_script("06_full_scale_binary_sweep.py", "sweep06")
            self._biosnap = (transductive, drug_split, drug_ids, sweep06)
        if seed not in self._biosnap_negatives:
            transductive, drug_split, drug_ids, sweep06 = self._biosnap
            train_cfg = {**self.train_cfg_base, "seed": seed}
            self._biosnap_negatives[seed] = sweep06.build_negative_augmented_splits(
                transductive, drug_ids, drug_split, train_cfg, self.train_cfg_base["negative_sample_ratio"])
        return self._biosnap_negatives[seed]

    def filtered_text(self, kind="filtered"):
        if kind not in self._filtered:
            suffix = FILTER_SUFFIX[kind]
            cache_path = os.path.join(ROOT, "data", f"text_cache_filtered{suffix}.json")
            embed_path = os.path.join(ROOT, "data", f"text_embeddings_filtered{suffix}.pt")
            if not (os.path.exists(cache_path) and os.path.exists(embed_path)):
                raise FileNotFoundError(f"{kind} text artifacts missing -- run scripts/12_text_leakage_screen.py first")
            with open(cache_path, encoding="utf-8") as f:
                filtered_cache = json.load(f)
            blob = torch.load(embed_path, weights_only=False)
            assert blob["dim"] == self.lm_dim
            has_text_filtered = {k: bool(v) for k, v in filtered_cache.items()}
            assert all(k in blob["embeddings"] for k, v in has_text_filtered.items() if v)
            self._filtered[kind] = (blob["embeddings"], has_text_filtered)
        return self._filtered[kind]

    def components(self, dataset, arch):
        m = self.model_cfg
        num_classes, binary = (dp.NUM_INTERACTION_TYPES, False) if dataset == "drugbank" else (None, True)
        if arch == "fp-text":
            return eu.make_fingerprint_text_components(
                self.fp_cache, self.text_emb, self.has_text, self.lm_dim, fp_dim=bl.FINGERPRINT_BITS,
                hidden_dim=m["hidden_dim"], head_hidden=m["head_hidden"], dropout=m["dropout"],
                num_classes=num_classes, binary=binary)
        if arch.startswith("gcn-"):
            return eu.make_ddi_components(arch[len("gcn-"):], self.common(graph_conv="gcn"), num_classes, binary,
                                           self.graphs, self.text_emb, self.has_text, self.lm_dim, self.device)
        for kind in FILTER_SUFFIX:
            if arch.startswith(f"{kind}-"):
                emb, has_text = self.filtered_text(kind)
                return eu.make_ddi_components(arch[len(kind) + 1:], self.common(), num_classes, binary,
                                               self.graphs, emb, has_text, self.lm_dim, self.device)
        return eu.make_ddi_components(arch, self.common(), num_classes, binary,
                                       self.graphs, self.text_emb, self.has_text, self.lm_dim, self.device)

    def run_args(self, dataset, regime, seed):
        """(pairs_splits, id/label columns, binary, evaluable sets) for one job --
        exactly the arguments 05 (DrugBank) / 06 (BioSNAP) pass to train_one_run."""
        if dataset == "drugbank":
            transductive, _, inductive, evaluable = self.drugbank()
            cols = dict(id1_col="ID1", id2_col="ID2", label_col="Y", binary=False)
            if regime == "transductive":
                pairs = {n: transductive[transductive["split"] == n] for n in ("train", "val", "test")}
                ev_sets = {"val": evaluable["transductive_val"], "test": evaluable["transductive_test"]}
            else:
                pairs = {n: inductive[n] for n in ("train", "val", "s1", "s2")}
                ev_sets = {"val": evaluable["inductive_val"], "s1": evaluable["inductive_s1"],
                           "s2": evaluable["inductive_s2"]}
        else:
            transductive, inductive = self.biosnap(seed)
            cols = dict(id1_col="d1", id2_col="d2", label_col="label", binary=True)
            pairs = dict(transductive) if regime == "transductive" else dict(inductive)
            ev_sets = {}
        if self.smoke:
            rng = random.Random(0)
            pairs = {n: df.sample(n=min(len(df), 3000 if n == "train" else 1500), random_state=rng.randint(0, 10**6))
                     for n, df in pairs.items()}
        return pairs, cols, ev_sets


def run_job(ctx, job, prefix, smoke):
    dataset, arch, regime, seed = job
    run_name = run_name_for(dataset, arch, regime, seed, prefix)
    train_cfg = {**ctx.train_cfg_base, "seed": seed}
    if smoke:
        train_cfg["epochs"] = 1
    pairs, cols, ev_sets = ctx.run_args(dataset, regime, seed)
    model_factory, batch_fn = ctx.components(dataset, arch)
    result = tr.train_one_run(run_name=run_name, model_factory=model_factory, batch_fn=batch_fn,
                              pairs_splits=pairs, train_cfg=train_cfg, evaluable_classes_by_split=ev_sets,
                              device=ctx.device, **cols)
    config = {"name": arch, "split_type": regime, "dataset": dataset, "seed": seed, "scale": "full",
              "camera_ready_experiment": True}
    return run_name, result, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", type=str, required=True, help=f"comma-separated subset of {sorted(EXPERIMENTS)}")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads for this worker")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on small seeded subsets, one job per (dataset, arch, regime); "
                             "nothing logged, checkpoints deleted afterwards")
    args = parser.parse_args()
    exps = [e for e in args.exps.split(",") if e]
    unknown = set(exps) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"unknown experiments: {sorted(unknown)}")

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"worker={args.worker} device={device} exps={exps} smoke={args.smoke} "
             f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
    if not args.smoke and os.environ.get("PYTHONHASHSEED") != "0" and any(
            d == "biosnap" for e in exps for d, _, _ in EXPERIMENTS[e]):
        raise SystemExit("set PYTHONHASHSEED=0 for BioSNAP jobs (reproducible inductive negatives)")

    _, sanity_ok = ev.run_sanity_checks()
    if not sanity_ok:
        raise SystemExit("evaluate.py sanity checks failed -- refusing to run models against broken metrics")

    ctx = Context(device, args.smoke)

    if args.smoke:
        seen = set()
        for dataset, arch, regime, seed in build_jobs(exps):
            if (dataset, arch, regime) in seen:
                continue
            seen.add((dataset, arch, regime))
            run_name, result, _ = run_job(ctx, (dataset, arch, regime, seed), "crsmoke", smoke=True)
            log.info(f"SMOKE OK {run_name}: params={result['trainable_params']} "
                     f"metrics={ {k: round(v['auprc'], 4) for k, v in result['metrics_by_split'].items()} }")
            for suffix in (".pt", "_latest.pt"):
                path = os.path.join(tr.CHECKPOINT_DIR, f"{run_name}{suffix}")
                if os.path.exists(path):
                    os.remove(path)
        log.info("all smoke jobs passed")
        return

    keep_system_awake()
    os.makedirs(QUEUE_DIR, exist_ok=True)
    jobs = build_jobs(exps)
    log.info(f"{len(jobs)} jobs in this worker's view: {[run_name_for(*j) for j in jobs]}")
    n_done = 0
    for job in jobs:
        run_name = run_name_for(*job)
        if not try_claim(run_name, args.worker):
            continue
        log.info(f"=== worker {args.worker} claimed {run_name} ===")
        try:
            run_name, result, config = run_job(ctx, job, "cr", smoke=False)
            with file_lock(os.path.join(QUEUE_DIR, "run_log.lock")):
                run_id = tr.log_run(result, config)
            with open(os.path.join(QUEUE_DIR, f"{run_name}.done"), "w") as f:
                f.write(f"{run_id}\n")
            n_done += 1
        except Exception:
            with open(os.path.join(QUEUE_DIR, f"{run_name}.failed"), "w") as f:
                f.write(traceback.format_exc())
            log.exception(f"job {run_name} failed; continuing with the next job")
    log.info(f"worker {args.worker} finished: {n_done} jobs completed by this worker")


if __name__ == "__main__":
    main()
