# DDI-Fusion — Language-Grounded Drug–Drug Interaction Prediction

Code for: *"Language-Grounded Drug Representations for Cold-Start Drug–Drug Interaction Prediction"*

Preprint: Research Square — https://doi.org/10.21203/rs.3.rs-10334947/v1

## Overview

DDI-Fusion is a GIN-based drug–drug interaction classifier that fuses molecular-graph embeddings with frozen PubMedBERT text embeddings at training time. The goal is to reduce the performance gap between transductive (both drugs seen during training) and cold-start (one or both drugs unseen) settings — the hardest, most clinically realistic regime for DDI prediction.

Key results (3-seed mean ± std):

| Model | DrugBank Trans | DrugBank S1 | DrugBank S2 | BioSNAP Trans | BioSNAP S1 | BioSNAP S2 |
|---|---|---|---|---|---|---|
| **Fusion (ours)** | 0.947 ± 0.003 | **0.627 ± 0.008** | **0.376 ± 0.025** | 0.927 ± 0.001 | **0.831 ± 0.002** | **0.726 ± 0.012** |
| Graph-only | 0.950 ± 0.002 | 0.583 ± 0.010 | 0.344 ± 0.018 | 0.932 ± 0.005 | 0.785 ± 0.038 | 0.688 ± 0.032 |
| Text-only | 0.882 ± 0.003 | 0.497 ± 0.013 | 0.251 ± 0.010 | 0.906 ± 0.005 | 0.792 ± 0.003 | 0.671 ± 0.018 |
| Fingerprint-MLP | **0.973 ± 0.003** | 0.558 ± 0.011 | 0.296 ± 0.012 | **0.962 ± 0.003** | 0.812 ± 0.009 | 0.676 ± 0.009 |

Metric: AUPRC (macro-averaged over evaluable interaction types, ≥10 examples in both train and eval split). S1 = one drug unseen; S2 = both drugs unseen. Fingerprint-MLP wins transductive on both datasets; Fusion wins all four cold-start columns and has the smallest transductive→cold-start gap on both datasets.

## Repository Structure

```
src/
  train.py              # Training loop with early stopping (val AUPRC)
  evaluate.py           # AUROC, AUPRC, F1, accuracy; evaluable-class logic
  data_pipeline.py      # SMILES→graph, PubChem text fetch+cache, PubMedBERT embeddings
  models.py             # GIN encoder, text/fingerprint heads, Fusion/ablation models
  splits.py             # Transductive + inductive (S1/S2) split construction
  experiment_utils.py   # Model factory helpers
  baselines.py          # Fingerprint-MLP baseline
  report.py             # Tables and figures from saved JSON (no retraining)

scripts/
  00_check_data.py              # Confirm raw data downloads + dataset stats
  01_smoke_test.py              # Data pipeline + split sanity on 150-drug subset
  02_model_smoke_test.py        # Model forward/backward sanity
  03_train_eval_smoke_test.py   # End-to-end train+eval on subset
  04_build_full_data.py         # Graph/fingerprint/text caches + splits (run once)
  05_full_scale_calibration.py  # Primary sweep: DrugBank, 4 models × 2 regimes × 3 seeds
  06_full_scale_binary_sweep.py # Robustness sweep: BioSNAP, same protocol
  07_ssiddi_faithfulness_check.py  # SSI-DDI reproduction attempt (exploratory)
  08_biosnap_f1_threshold_check.py # F1 threshold calibration diagnostic
  09_per_type_drugbank_breakdown.py # Per-interaction-type fusion vs. graph-only analysis
  10_per_type_figures.py           # Per-type figures
  11_per_metric_tables.py          # Separate AUROC/F1/accuracy tables

configs/
  model.yaml       # GIN depth/width, head dims, dropout
  train_full.yaml  # Epochs, batch size, lr, patience, negative sampling ratio
  data.yaml        # Dataset URLs, split fractions, text model name

results/           # Saved JSON stats, per-seed CSVs, LaTeX tables, checkpoints
figures/           # Paper figures (PDF)
run_all.sh         # One-command end-to-end reproduction (~14 hours)
REPRODUCE.md       # Step-by-step commands with observed wall-clocks
```

## Trained Checkpoints

The 24 trained checkpoints (4 models × 2 regimes × 3 seeds) from the DrugBank sweep and 24 from the BioSNAP sweep are saved under `results/checkpoints/` after running the sweeps. Naming convention:

```
results/checkpoints/fullscale_{model}_{transductive|inductive}_seed{0,1,2}.pt   # DrugBank
results/checkpoints/binary_{model}_{transductive|inductive}_seed{0,1,2}.pt      # BioSNAP
```

where `{model}` is one of `fusion=concat`, `graph-only`, `text-only`, `fingerprint-mlp`.

## Installation

```
pip install -r requirements.txt
```

Python 3.9+ recommended. Tested on PyTorch 2.6, torch-geometric 2.8, numpy 2.x. GPU is optional — peak memory on the main sweeps is 21–47 MB, so even a 4 GB GPU has enormous headroom.

## Dataset Setup

**DrugBank** (multi-relational, 191,808 pairs, 1,706 drugs, 86 interaction types) is downloaded automatically by the scripts via the GMPNN-CS public mirror:

```
python scripts/00_check_data.py   # downloads and validates
python scripts/04_build_full_data.py  # builds caches (run once, ~15-40 min)
```

Drug text descriptions are fetched from PubChem (~1.3 s/drug, rate-limited, cached to `data/text_cache.json` so the fetch only runs once). 1,501 of 1,706 drugs (88%) receive a description.

**BioSNAP ChCh-Miner** (binary, 39,474 positive pairs, 1,268 drugs) is also downloaded automatically from the Stanford SNAP repository. Its 1,268 drugs are a subset of the DrugBank 1,706, so the graph/text/fingerprint caches from step 3 are reused — no extra fetching needed.

Expected layout after setup:

```
data/
  drugbank.tab
  graph_cache.pt
  text_cache.json
  text_embeddings.pt
  fingerprint_cache.pt
  splits/
    drugbank_seed0/
    chch_seed0/
```

## Training

### DrugBank primary sweep (4 architectures × 2 regimes × 3 seeds, ~11.8h)

```bash
python scripts/05_full_scale_calibration.py
```

Flags for partial runs:

```bash
# Run only specific architectures
python scripts/05_full_scale_calibration.py --only fusion=concat,graph-only

# Run only specific seeds
python scripts/05_full_scale_calibration.py --seeds 0,1

# Quick timing probe (short epochs)
python scripts/05_full_scale_calibration.py --max-epochs 5 --batch-size 256
```

### BioSNAP robustness sweep (same protocol, ~1.7h)

```bash
python scripts/06_full_scale_binary_sweep.py
```

### Architectures

| Name (CLI) | Description |
|---|---|
| `fusion=concat` | GIN + PubMedBERT text → concat → MLP head **(proposed)** |
| `graph-only` | GIN only → MLP head |
| `text-only` | PubMedBERT only → MLP head |
| `fingerprint-mlp` | ECFP4 fingerprint → MLP head (no graph, no text) |

### Model hyperparameters (`configs/model.yaml`)

- GIN: 3 layers, hidden dim 64, mean readout
- Atom features: 35-dim; bond features: 6-dim
- PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`): frozen, 768-dim mean-pooled
- MLP head: 1 × 128, dropout 0.1
- Adam, lr 1e-3, batch 128, ≤40 epochs, early stop on val AUPRC (patience 5)
- Negative sampling: 1:1 ratio

## Smoke Tests (optional but recommended before a full run)

```bash
python scripts/01_smoke_test.py           # ~5 min
python scripts/02_model_smoke_test.py     # ~2 min
python scripts/03_train_eval_smoke_test.py  # ~10-20 min
```

These run the full pipeline on a 150-drug subset to confirm the harness is correct before spending hours on the real sweep.

## Tables and Figures

```bash
python src/report.py
```

Reads the saved sweep JSON outputs — never retrains. Writes:

- `results/main_results_drugbank.{tex,csv}`, `results/main_results_biosnap.{tex,csv}`
- `results/gap_table.{tex,csv}`
- `figures/inductive_gap.pdf`, `figures/ablation.pdf`

Per-interaction-type breakdown (DrugBank fusion vs. graph-only):

```bash
python scripts/09_per_type_drugbank_breakdown.py  # ~30 min, reloads checkpoints
python scripts/10_per_type_figures.py
python scripts/11_per_metric_tables.py            # AUROC/F1/accuracy sub-tables
```

## Full Reproduction

```bash
bash run_all.sh
```

Runs all steps in order. Expect roughly **14 hours** end-to-end on a single GPU, dominated by the DrugBank sweep (~12h). See `REPRODUCE.md` for step-by-step commands, observed wall-clocks, and notes on each script.

## Citation

Please contact the author's email if you need to cite this work. Preprint available at:
https://doi.org/10.21203/rs.3.rs-10334947/v1

## License

MIT License. See `LICENSE` for details.
