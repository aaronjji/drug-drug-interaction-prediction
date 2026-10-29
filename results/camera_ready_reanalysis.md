# Camera-ready re-analysis (inference only)

Common evaluable class set: 28 types. Sanity max |stored-recomputed| AUPRC: 2.9e-05.

| model | T common | S1 common | S2 common | gap T-S2 common | S1 train-partner | S1 val-partner |
|---|---|---|---|---|---|---|
| Fusion (GIN+text) | 0.967±0.002 | 0.615±0.008 | 0.376±0.025 | 0.591±0.023 | 0.657±0.006 | 0.384±0.017 |
| Graph-only (GIN) | 0.968±0.002 | 0.567±0.016 | 0.344±0.018 | 0.624±0.020 | 0.614±0.009 | 0.359±0.024 |
| Text-only | 0.875±0.005 | 0.476±0.016 | 0.251±0.010 | 0.624±0.012 | 0.528±0.013 | 0.293±0.010 |
| Fingerprint-MLP | 0.983±0.001 | 0.544±0.003 | 0.296±0.012 | 0.687±0.013 | 0.599±0.011 | 0.287±0.018 |
| Fingerprint+text | 0.981±0.002 | 0.590±0.003 | 0.335±0.016 | 0.646±0.017 | 0.634±0.006 | 0.361±0.003 |
| GCN graph-only | -- | 0.554±0.016 | 0.344±0.017 | -- | 0.611±0.014 | 0.371±0.014 |
| GCN fusion | -- | 0.595±0.001 | 0.363±0.030 | -- | 0.640±0.007 | 0.366±0.010 |
| Fusion (gated) | -- | 0.595±0.014 | 0.346±0.035 | -- | 0.631±0.011 | 0.387±0.016 |
| Fusion (cross-attention) | -- | 0.572±0.010 | 0.337±0.006 | -- | 0.608±0.020 | 0.361±0.007 |

| model | S1 sim low | S1 sim mid | S1 sim high | S2 sim low | S2 sim high |
|---|---|---|---|---|---|
| Fusion (GIN+text) | 0.474±0.017 | 0.707±0.004 | 0.748±0.003 | 0.313±0.020 | 0.427±0.018 |
| Graph-only (GIN) | 0.396±0.028 | 0.681±0.019 | 0.762±0.005 | 0.272±0.029 | 0.428±0.024 |
| Text-only | 0.400±0.012 | 0.566±0.014 | 0.613±0.013 | 0.211±0.018 | 0.293±0.008 |
| Fingerprint-MLP | 0.306±0.012 | 0.671±0.011 | 0.766±0.012 | 0.189±0.019 | 0.402±0.016 |
| Fingerprint+text | 0.400±0.006 | 0.693±0.008 | 0.772±0.011 | 0.226±0.025 | 0.408±0.007 |
| GCN graph-only | 0.390±0.033 | 0.679±0.006 | 0.734±0.010 | 0.263±0.031 | 0.429±0.013 |
| GCN fusion | 0.447±0.004 | 0.716±0.008 | 0.719±0.003 | 0.285±0.026 | 0.402±0.017 |
| Fusion (gated) | 0.441±0.001 | 0.726±0.013 | 0.718±0.008 | 0.280±0.040 | 0.400±0.022 |
| Fusion (cross-attention) | 0.433±0.011 | 0.659±0.013 | 0.706±0.022 | 0.282±0.013 | 0.382±0.032 |

```
{
 "drugbank": {
  "n_drugs": {
   "train": 1194,
   "val": 170,
   "test": 342
  },
  "transductive_pairs": {
   "train": 134264,
   "val": 19178,
   "test": 38366
  },
  "inductive_pairs": {
   "train": 99372,
   "val": 28027,
   "s1": 57792,
   "s2": 6617
  },
  "val_pairs_one_val_drug": 26269,
  "val_pairs_two_val_drugs": 1758,
  "s1_pairs_train_partner": 51102,
  "s1_pairs_val_partner": 6690,
  "inductive_pairs_total": 191808,
  "drug_sets_disjoint": true
 },
 "biosnap_positives": {
  "n_drugs": {
   "train": 887,
   "val": 126,
   "test": 255
  },
  "transductive_pairs": {
   "train": 27632,
   "val": 3947,
   "test": 7895
  },
  "inductive_pairs": {
   "train": 20407,
   "val": 5318,
   "s1": 12236,
   "s2": 1513
  },
  "val_pairs_one_val_drug": 5010,
  "val_pairs_two_val_drugs": 308,
  "s1_pairs_train_partner": 10841,
  "s1_pairs_val_partner": 1395,
  "inductive_pairs_total": 39474,
  "drug_sets_disjoint": true
 }
}
{
 "s1_tertile_cuts": [
  0.37931033968925476,
  0.5806451439857483
 ],
 "s2_median_cut": 0.4816638231277466,
 "test_drug_max_sim_quartiles": [
  0.3220735862851143,
  0.4747881293296814,
  0.6307397931814194
 ],
 "s1_bin_sizes": {
  "low": 19325,
  "mid": 19411,
  "high": 19056
 },
 "s2_bin_sizes": {
  "low": 3309,
  "high": 3308
 }
}
```
