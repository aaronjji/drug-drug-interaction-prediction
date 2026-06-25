# Per-interaction-type cold-start breakdown: DrugBank, fusion=concat vs. graph-only

No retraining. Evaluated the exact saved checkpoints from the 3-seed full-scale sweep (`results/checkpoints/fullscale_{fusion=concat,graph-only}_inductive_seed{0,1,2}.pt`) on the persisted seed-0 S1/S2 inductive splits and evaluable-class sets every other script in this repo reuses verbatim (`data/splits/drugbank_seed0/`).

## Sanity check: does this reconcile with the main table?

Recomputing the macro average of the per-type AUPRC/AUROC over each split's evaluable classes reproduces the `full_scale_calibration_stats.json` summary table to within max|diff| = 3.2e-06 (AUPRC) and 4.8e-08 (AUROC) across all 12 (model, seed, split) combinations -- floating-point noise, not a discrepancy. Evaluable-class counts also match exactly (62/86 for S1, 28/86 for S2). **This analysis is consistent with the headline numbers already in the paper.**

## Verdict: is the cold-start advantage concentrated, or diffuse?

**Mostly diffuse, with a broad but inconsistent tilt toward fusion, not a clean interpretable subset.** Fusion has a positive AUPRC delta on 71% of S1's 62 evaluable types and 68% of S2's 28 evaluable types -- a majority, consistent with fusion winning on the macro-averaged metric -- but the per-type delta is highly heterogeneous (S1: mean 0.044, std 0.112; S2: mean 0.032, std 0.130 -- the spread across types is larger than the average gain itself), ranging from a roughly -0.28 to +0.43 swing in S1 alone. A substantial minority of types (18/62 in S1, 9/28 in S2) actually favor graph-only.

### Rule-based mechanistic bucketing (derived from the literal `Map` template text in `data/drugbank.tab`, not invented)

Every evaluable type's template was assigned to exactly one bucket by substring match: **PK (ADME)** if it mentions bioavailability/metabolism/absorption/serum concentration/excretion rate; **Risk/severity (unspecified mechanism)** if it starts with "The risk or severity of..."; **Therapeutic efficacy (unspecified mechanism)** if it mentions "therapeutic efficacy"; **PD (named activity)** for the remaining "...activities of..." templates. All evaluable types in both splits matched exactly one bucket (no leftovers).

Bucket means (mean ± std of the per-type AUPRC delta, NOT a macro-average -- small buckets are noted explicitly):

| split | bucket | n types | mean delta | std delta |
|---|---|---|---|---|
| S1 | PD (named activity) | 47 | 0.049 | 0.126 |
| S1 | PK (ADME) | 10 | 0.021 | 0.028 |
| S1 | Risk/severity (unspecified mechanism) | 3 | 0.083 | 0.056 |
| S1 | Therapeutic efficacy (unspecified mechanism) | 2 | -0.008 | 0.055 |
| S2 | PD (named activity) | 19 | 0.047 | 0.154 |
| S2 | PK (ADME) | 7 | -0.010 | 0.047 |
| S2 | Risk/severity (unspecified mechanism) | 1 | 0.053 | n/a (n=1) |
| S2 | Therapeutic efficacy (unspecified mechanism) | 1 | 0.022 | n/a (n=1) |

PK (ADME) types show a smaller average delta than PD (named-activity) types on both splits (S1: 0.021 vs. 0.049; S2: -0.010 vs. 0.047), a directionally consistent hint that language grounding helps less for the ADME-process types and more for the named-pharmacological-activity types. **This is a weak signal, not a clean finding**: the within-bucket standard deviation (0.13-0.15 for PD) is several times larger than the between-bucket mean gap (0.03-0.06), the PK bucket itself is small (10 types in S1, 7 in S2), and a Mann-Whitney test does not reject equality of the two bucket distributions at conventional significance:

- S1: PK (ADME) n=10 mean=0.021 vs. PD (named activity) n=47 mean=0.049 -- Mann-Whitney p=0.40 (not significant at this sample size)
- S2: PK (ADME) n=7 mean=-0.010 vs. PD (named activity) n=19 mean=0.047 -- Mann-Whitney p=0.28 (not significant at this sample size)

The Risk/severity and Therapeutic-efficacy buckets have only 1-3 evaluable types per split each -- far too few to support any claim about them and reported here only for completeness.

## Top-5 / bottom-5 types by AUPRC delta (fusion − graph-only)

### S1

**Top 5 (fusion helps most):**

| type_id | label | bucket | n_train | n_test | low support? | delta AUPRC (mean±std) |
|---|---|---|---|---|---|---|
| 16 | #Drug1 may decrease the neuromuscular blocking activities of #Drug2. | PD (named activity) | 55 | 19 | YES | 0.434±0.117 |
| 83 | #Drug1 may increase the vasopressor activities of #Drug2. | PD (named activity) | 50 | 10 | YES | 0.361±0.205 |
| 22 | #Drug1 may increase the stimulatory activities of #Drug2. | PD (named activity) | 24 | 12 | YES | 0.248±0.297 |
| 56 | #Drug1 may increase the nephrotoxic activities of #Drug2. | PD (named activity) | 254 | 223 | no | 0.240±0.132 |
| 5 | #Drug1 may increase the anticoagulant activities of #Drug2. | PD (named activity) | 1197 | 1255 | no | 0.205±0.076 |

**Bottom 5 (graph-only does relatively better):**

| type_id | label | bucket | n_train | n_test | low support? | delta AUPRC (mean±std) |
|---|---|---|---|---|---|---|
| 18 | #Drug1 may increase the vasoconstricting activities of #Drug2. | PD (named activity) | 50 | 41 | no | -0.283±0.184 |
| 35 | #Drug1 may increase the antipsychotic activities of #Drug2. | PD (named activity) | 36 | 12 | YES | -0.154±0.134 |
| 13 | #Drug1 may decrease the bronchodilatory activities of #Drug2. | PD (named activity) | 199 | 89 | no | -0.143±0.122 |
| 67 | #Drug1 may increase the hyperkalemic activities of #Drug2. | PD (named activity) | 170 | 89 | no | -0.136±0.058 |
| 82 | #Drug1 may increase the hypokalemic activities of #Drug2. | PD (named activity) | 736 | 313 | no | -0.078±0.060 |

### S2

**Top 5 (fusion helps most):**

| type_id | label | bucket | n_train | n_test | low support? | delta AUPRC (mean±std) |
|---|---|---|---|---|---|---|
| 5 | #Drug1 may increase the anticoagulant activities of #Drug2. | PD (named activity) | 1197 | 256 | no | 0.324±0.154 |
| 56 | #Drug1 may increase the nephrotoxic activities of #Drug2. | PD (named activity) | 254 | 33 | no | 0.307±0.217 |
| 39 | #Drug1 may increase the respiratory depressant activities of #Drug2. | PD (named activity) | 90 | 21 | no | 0.281±0.324 |
| 20 | #Drug1 may increase the neuromuscular blocking activities of #Drug2. | PD (named activity) | 209 | 10 | YES | 0.214±0.069 |
| 26 | #Drug1 may increase the neuroexcitatory activities of #Drug2. | PD (named activity) | 468 | 20 | no | 0.136±0.105 |

**Bottom 5 (graph-only does relatively better):**

| type_id | label | bucket | n_train | n_test | low support? | delta AUPRC (mean±std) |
|---|---|---|---|---|---|---|
| 84 | #Drug1 may increase the tachycardic activities of #Drug2. | PD (named activity) | 219 | 11 | YES | -0.241±0.133 |
| 70 | #Drug1 may increase the hypertensive activities of #Drug2. | PD (named activity) | 296 | 38 | no | -0.146±0.103 |
| 82 | #Drug1 may increase the hypokalemic activities of #Drug2. | PD (named activity) | 736 | 17 | YES | -0.140±0.346 |
| 15 | #Drug1 may increase the central nervous system depressant (CNS depressant) activities of #Drug2. | PD (named activity) | 3417 | 72 | no | -0.091±0.121 |
| 3 | The metabolism of #Drug2 can be increased when combined with #Drug1. | PK (ADME) | 2758 | 151 | no | -0.088±0.172 |

Of the 10 combined top-5 entries above (S1+S2), 4 are flagged `low_support` (n_test < 20, i.e. close to the evaluable-class floor of 10). Restricting to types with n_test >= 20 (see `results/per_type_drugbank.csv`, `low_support` column) changes which exact types sit at the extreme top/bottom but does not change the overall verdict: the ranking is still dominated by the numerically larger PD (named-activity) bucket simply because it contains most of the evaluable types (47/62 in S1, 19/28 in S2), and the spread within that bucket alone (both strongly positive and strongly negative deltas) is as wide as the spread across the whole type set.

