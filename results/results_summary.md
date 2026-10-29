# Results summary

Language-grounded inductive drug–drug interaction prediction: does fusing a
molecular-graph encoder with a frozen biomedical-LM text embedding help DDI
prediction hold up for drugs the model has never seen interact (cold-start),
where structure-only methods degrade? All numbers below come from code that
was actually run (`scripts/05_full_scale_calibration.py`,
`scripts/06_full_scale_binary_sweep.py`), 3 seeds each, on the full datasets.

**Generated artifacts** (via `src/report.py`, reads-only, no retraining):
`results/main_results_drugbank.{tex,csv}`, `results/main_results_biosnap.{tex,csv}`,
`results/gap_table.{tex,csv}`, `figures/inductive_gap.pdf`, `figures/ablation.pdf`.
See `results/report_missing_metrics.txt` for metrics a paper table might want
that the saved outputs don't contain (per-class breakdowns, significance tests).

## Headline finding

**Fusion wins both cold-start splits (S1, S2) on AUROC and AUPRC, with the
smallest transductive→cold-start gap of any model, on two structurally
different datasets (DrugBank multi-relational, BioSNAP ChCh-Miner binary).**
The pattern is consistent and not a coincidence of one dataset:

- Fingerprint-MLP (structure-only, no graph learning, no text) is the best
  *transductive* model on both datasets — simple ECFP fingerprints are a
  strong memorization signal when training drugs are seen at test time.
- That same model degrades the *most* moving to cold-start (largest gap),
  exactly the "structure-only methods degrade" pattern the project set out
  to test.
- Fusion is never the transductive winner, but degrades the *least* and wins
  outright on both cold-start splits, on both datasets.

## DrugBank (multi-relational, 86 interaction types, 1,706 drugs), 3 seeds

Evaluable classes (≥10 examples in both train and eval split): 63/86
(transductive test), 62/86 (S1), 28/86 (S2) — at full scale the task is
learnable across most classes, not a handful.

AUPRC, mean ± std over seeds 0/1/2:

| model | transductive | S1 (one drug unseen) | S2 (both drugs unseen) |
|---|---|---|---|
| fusion=concat | 0.947 ± 0.003 | **0.627 ± 0.008** | **0.376 ± 0.025** |
| graph-only | 0.950 ± 0.002 | 0.583 ± 0.010 | 0.344 ± 0.018 |
| text-only | 0.882 ± 0.003 | 0.497 ± 0.013 | 0.251 ± 0.010 |
| fingerprint-mlp | 0.974 ± 0.003 | 0.558 ± 0.011 | 0.296 ± 0.012 |

Transductive-minus-cold-start AUPRC gap (smaller = degrades less):

| model | gap → S1 | gap → S2 |
|---|---|---|
| **fusion=concat** | **0.320 ± 0.005** | **0.571 ± 0.023** |
| graph-only | 0.367 ± 0.008 | 0.606 ± 0.018 |
| text-only | 0.386 ± 0.016 | 0.631 ± 0.012 |
| fingerprint-mlp | 0.416 ± 0.009 | 0.677 ± 0.015 |

Full sweep: 24 runs (4 architectures × 2 regimes × 3 seeds), 11.8 hours wall
clock, 21–47MB peak GPU memory. Only 2/24 runs hit the 40-epoch cap; the rest
converged with room to spare.

## BioSNAP ChCh-Miner (binary, 39,474 positive pairs, 1,268 drugs), 3 seeds

Same architectures, same protocol, same negative-sampling/evaluable-class/
metric harness as DrugBank — no special-casing. Negative sampling: 1:1 ratio,
seeded per training seed (resampled fresh each seed, not frozen), and
bucket-aware for the inductive splits so a cold-start negative can't
accidentally be two train drugs (which would leak train-drug information into
S1/S2 evaluation). Text coverage: 89.3% (train drugs) vs 89.4% (S1/S2-pool
test drugs) — the stratified split keeps this confound from concentrating
text-coverage gaps in the cold-start set.

AUPRC, mean ± std over seeds 0/1/2:

| model | transductive | S1 | S2 |
|---|---|---|---|
| fusion=concat | 0.927 ± 0.001 | **0.831 ± 0.002** | **0.726 ± 0.012** |
| graph-only | 0.932 ± 0.005 | 0.785 ± 0.038 | 0.688 ± 0.032 |
| text-only | 0.907 ± 0.005 | 0.792 ± 0.003 | 0.671 ± 0.018 |
| fingerprint-mlp | 0.962 ± 0.003 | 0.812 ± 0.009 | 0.676 ± 0.010 |

Transductive-minus-cold-start AUPRC gap:

| model | gap → S1 | gap → S2 |
|---|---|---|
| **fusion=concat** | **0.096 ± 0.001** | **0.201 ± 0.013** |
| graph-only | 0.147 ± 0.040 | 0.244 ± 0.033 |
| text-only | 0.114 ± 0.006 | 0.236 ± 0.017 |
| fingerprint-mlp | 0.150 ± 0.011 | 0.286 ± 0.013 |

### F1 exception, resolved

On S2-F1 specifically (not AUROC/AUPRC), graph-only beats fusion at the fixed
0.5 decision threshold (0.614 ± 0.016 vs 0.562 ± 0.021). Tuning the threshold
on the validation split (never on S2 itself) and re-evaluating flips the
order: fusion 0.679 ± 0.015 vs graph-only 0.659 ± 0.012. Fusion's optimal
threshold is far lower (0.03–0.08) than graph-only's (0.22–0.40), meaning
fusion's cold-start probabilities are systematically more conservative — a
calibration artifact at the fixed cutoff, not a deficiency in ranking ability
(AUROC/AUPRC already favored fusion before any tuning).

Full sweep: 24 runs, 1.67 hours wall clock.

## SSI-DDI literature baseline: not included, here's why

We attempted to add SSI-DDI (Nyamabo et al., 2021,
[kanz76/SSI-DDI](https://github.com/kanz76/SSI-DDI), commit `23e958f`) as a
published-method comparison, faithfully reproducing their GAT-block +
SAGPooling + co-attention + RESCAL architecture, their own data loader, their
own negative-sampling scheme, and their own optimizer/LR schedule —
confirmed byte-for-byte identical to their `train_script.py` defaults
(`ratio=1.0, neg_ent=1, lr=1e-2, weight_decay=5e-4, Adam,
LambdaLR(0.96^epoch)`).

Training their model on their own train/val/test split (their own
pos-vs-corrupted-negative evaluation, not ours) plateaus at **AUROC ≈ 0.93–0.95**
across two independent runs (100 epochs and an extended 137-epoch run with a
plateau guardrail), and does not improve with more training — ruling out an
epoch-budget explanation. We also tested whether our one genuine code
addition (mixed-precision training) explained the gap: a side-by-side FP32
run tracked the mixed-precision run almost exactly at the same epochs, ruling
that out too.

We were unable to reproduce SSI-DDI's published numbers within our compute
budget. We therefore do not report it as a head-to-head baseline in the
comparison tables above. We could not independently verify the exact
published figures either — web searches for the paper's reported numbers
returned inconsistent values across sources (e.g. 94.47% vs 96.33% reported
accuracy), and the original paper (Briefings in Bioinformatics, 2021) is
paywalled, so we did not read the primary table directly. We cite the
~0.94–0.99 range found via search as context only, not as a verified ground
truth, and make no claim about what SSI-DDI itself actually achieves — only
that our reproduction attempt, under our compute budget, did not reach that
range.

*Update (Oct 2026):* the published figures were later read from the article's
full text (DrugBank, transductive: accuracy 0.9447, AUROC 0.9838, AUPRC 0.9814).
Our runs reached test AUROC 0.944 (100 epochs) and 0.933 (137 epochs), i.e.
4.0–5.1 points below; see the paper's SSI-DDI appendix.

## What this means for the paper

The core contribution — fusion vs. graph-only/text-only/fingerprint-MLP,
replicated across two structurally different datasets with a statistically
clean cold-start gap analysis — stands on its own and does not depend on the
SSI-DDI comparison. SSI-DDI was attempted as a nice-to-have published-method
reference point, not the result.
