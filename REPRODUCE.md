# Reproducing these results

Exact commands, in order, from a fresh clone to the final tables and figures.
All seeds used throughout: **0, 1, 2**. Wall-clocks below are observed on a
single NVIDIA T550 (4GB), not estimates — your hardware will vary, but GPU
memory headroom is enormous everywhere (21–47MB peak on the main sweeps), so
the bottleneck is wall-clock, not memory.

## 0. Environment

```
pip install -r requirements.txt
```

Verifies: `torch` (CUDA build recommended; falls back to CPU automatically),
`torch-geometric`, `rdkit`, `transformers`, `scikit-learn`, `pandas`,
`matplotlib`. See `requirements.txt` for the exact pinned versions used.

## 1. Confirm data sources download and parse (~1 min)

```
python scripts/00_check_data.py
```

Downloads `drugbank.tab` (primary multi-relational dataset) and
`twosides_ge_500.zip`, prints dataset stats, and runs a 60-drug PubChem
text-coverage sample. No model training. Confirms: 1,706 drugs, 191,808
pairs, 86 interaction types.

## 2. Pipeline smoke tests on a 150-drug subset (~10–20 min total)

```
python scripts/01_smoke_test.py            # data_pipeline.py + splits.py
python scripts/02_model_smoke_test.py      # models.py forward/backward
python scripts/03_train_eval_smoke_test.py # train.py + evaluate.py end-to-end
```

These prove the harness is correct on a cheap subset before spending hours of
compute on the real sweep. Not required to reproduce the final numbers, but
recommended before re-running anything at full scale after a code change.

## 3. Build full-scale data (~15–40 min, dominated by the PubChem fetch)

```
python scripts/04_build_full_data.py
```

Builds graph/fingerprint caches for all 1,706 drugs (seconds) and fetches
PubChem text descriptions for all of them (slow: ~1.3s/drug, rate-limited,
cached to disk afterward so this only runs once). Builds and saves the
seed-0 transductive + inductive (S1/S2) splits and the persisted
evaluable-class sets that every model below reuses verbatim.

Real output: 1,501/1,706 drugs (88.0%) get a text description; 63/86, 62/86,
28/86 evaluable interaction types for transductive-test/S1/S2 respectively.

## 4. Primary sweep: DrugBank, 4 architectures × 2 regimes × 3 seeds (~11.8h)

```
python scripts/05_full_scale_calibration.py
```

Trains fusion=concat, graph-only, text-only, and the fingerprint-MLP
baseline, each transductively and inductively, for seeds 0/1/2 (24 runs
total). Every run is logged to `results/run_log.txt` and
`results/runs/*.json`, and checkpointed every epoch
(`results/checkpoints/`) so an interrupted run resumes automatically.

`--only <arch1,arch2>` restricts to a subset of architectures;
`--seeds <csv>` restricts seeds; `--max-epochs`/`--batch-size` override
`configs/train_full.yaml` (epochs=40, batch_size=128) for a quick timing
probe before committing to the full run.

## 5. Robustness sweep: BioSNAP ChCh-Miner, same protocol (~1.7h)

```
python scripts/06_full_scale_binary_sweep.py
```

Same 4 architectures, same transductive+inductive(S1/S2) protocol, same
3 seeds, on the binary ChCh-Miner dataset — reuses the graph/text/
fingerprint caches from step 3 (ChCh-Miner's 1,268 drugs are a subset of
DrugBank's 1,706). Builds its own seed-0 splits
(`data/splits/chch_seed0/`) and 1:1 seeded negative sampling, bucket-aware
for the inductive splits.

## 6. BioSNAP F1 calibration check (~1 min, reuses saved checkpoints)

```
python scripts/08_biosnap_f1_threshold_check.py
```

No retraining — reloads the fusion=concat and graph-only checkpoints from
step 5 and re-evaluates S2 at a validation-tuned decision threshold instead
of the fixed 0.5 cutoff. Confirms the apparent S2-F1 advantage for
graph-only at the default threshold is a calibration artifact, not a
ranking deficiency.

## 7. Generate tables and figures (~5 sec, reads saved JSON only)

```
python src/report.py
```

Reads `results/full_scale_calibration_stats.json`,
`results/full_scale_binary_sweep_stats.json`, and
`results/biosnap_f1_threshold_check.json` — never retrains, never calls
`train.py`. Writes:

- `results/main_results_drugbank.{tex,csv}`, `results/main_results_biosnap.{tex,csv}`
- `results/gap_table.{tex,csv}`
- `figures/inductive_gap.pdf`, `figures/ablation.pdf`
- `results/report_missing_metrics.txt` (metrics a paper table might want that
  the saved outputs don't contain)

If a required input JSON is missing, this prints which one and exits rather
than silently regenerating it by retraining.

## Optional / exploratory: SSI-DDI literature baseline (not in the final results)

```
python scripts/07_ssiddi_faithfulness_check.py                              # ~4.7h, 100 epochs
python scripts/07_ssiddi_faithfulness_check.py --max-epochs 30 --no-amp \
    --checkpoint-tag noamp30                                                # ~1h, fidelity diagnostic
```

We attempted to add SSI-DDI (Nyamabo et al. 2021) as a published-method
comparison. Our faithful reproduction of their architecture/data/training
setup plateaus at AUROC≈0.93–0.95 on their own evaluation, short of the
(unverified) published range, and a time-boxed fidelity check found no
implementation discrepancy. **This is not included in the final comparison
tables** — see `results/results_summary.md` for the full account. These
commands are included for transparency, not because they're needed to
reproduce the reported results.

## All of the above in one script

```
bash run_all.sh
```

Runs steps 0–7 in order. Expect roughly **14 hours** end to end, dominated by
step 4 (~12h). Does not run the optional SSI-DDI commands.

## Camera-ready additions (AI4DD 2026 reviews)

Controls and analyses requested in review. None of these scripts touches the
original sweep's outputs: new training runs are named `cr_*`, and the
original `results/*_stats.json` files are only read.

```
python scripts/12_text_leakage_screen.py                 # ~3 min: screen descriptions, filter, re-embed
python scripts/12_text_leakage_screen.py --level generic # milder filter: explicit interaction wording only
python scripts/13_camera_ready_experiments.py --smoke --exps fp-text,gcn,fusion-variants,filtered-text
PYTHONHASHSEED=0 python scripts/13_camera_ready_experiments.py \
    --exps fp-text,gcn,fusion-variants,filtered-text,filtered-generic --worker 0   # start several workers to share the GPU
python scripts/14_camera_ready_reanalysis.py             # inference only (--only "Label A|Label B" to limit): common classes, S1 partner, similarity
PYTHONHASHSEED=0 python scripts/14_camera_ready_reanalysis.py --biosnap-redraw
python scripts/15_camera_ready_tables.py                 # reads saved outputs only
```

- `12_text_leakage_screen.py` matches every PubChem description sentence
  against the effect terms of all 86 DrugBank interaction types (plus
  morphological variants and defining class names) and generic interaction
  wording, removes matching sentences, and re-embeds the rest with the same
  frozen PubMedBERT (`data/text_cache_filtered.json`,
  `data/text_embeddings_filtered.pt`, `results/text_leakage_screen.{json,md}`).
  It reuses an existing embedding file built from identical text.
- `13_camera_ready_experiments.py` trains fingerprint+text (DrugBank and
  BioSNAP, both regimes), GCN graph-only/fusion, gated and cross-attention
  fusion, and fusion/text-only on filtered text (DrugBank inductive), 3 seeds
  each, through the unchanged `train.py`. Workers claim jobs via lock files in
  `results/cr_queue/`, so several can run at once (peak GPU memory per run is
  tens of MB); delete a `.claim` file to retry an interrupted job (it resumes
  from its `_latest.pt` checkpoint).
- `PYTHONHASHSEED=0` matters for BioSNAP only: the bucket-aware negative
  sampler (`splits.sample_bucket_negatives`, used by script 06) iterates
  Python sets of drug ids, whose order depends on the per-process hash seed.
  The original sweep did not fix it, so its exact inductive negatives cannot
  be recreated; `14 ... --biosnap-redraw` re-scores the original BioSNAP
  checkpoints on the `PYTHONHASHSEED=0` draw (max per-seed AUPRC shift 0.021,
  rankings unchanged). Script 08's threshold check likewise uses its own
  re-draw for both models.
- `14_camera_ready_reanalysis.py` re-scores saved DrugBank checkpoints on the
  28 types evaluable in T, S1 and S2, on S1 split by partner (training vs.
  validation drug), and on structural-similarity strata. It asserts that the
  recomputed official metrics match the stored ones (max diff 3e-5).
- Wall clock: the 36 camera-ready training runs (12 configurations x 3 seeds)
  sum to 10.3 GPU-run hours; with six to eight workers sharing the 4GB T550
  they finished in about 2.2 hours.
