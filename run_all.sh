#!/usr/bin/env bash
# End-to-end reproduction, in order. See REPRODUCE.md for what each step does
# and observed wall-clocks. Expect ~14 hours total, dominated by step 4 (~12h).
# Does not run the optional/exploratory SSI-DDI commands (see REPRODUCE.md).
set -e

echo "=== 0. dependencies ==="
pip install -r requirements.txt

echo "=== 1. confirm data sources (~1 min) ==="
python scripts/00_check_data.py

echo "=== 2. pipeline smoke tests on 150-drug subset (~10-20 min) ==="
python scripts/01_smoke_test.py
python scripts/02_model_smoke_test.py
python scripts/03_train_eval_smoke_test.py

echo "=== 3. build full-scale data (~15-40 min) ==="
python scripts/04_build_full_data.py

echo "=== 4. primary sweep: DrugBank, 3 seeds x 4 architectures x 2 regimes (~11.8h) ==="
python scripts/05_full_scale_calibration.py

echo "=== 5. robustness sweep: BioSNAP, same protocol (~1.7h) ==="
python scripts/06_full_scale_binary_sweep.py

echo "=== 6. BioSNAP F1 calibration check (~1 min) ==="
python scripts/08_biosnap_f1_threshold_check.py

echo "=== 7. tables and figures (~5 sec) ==="
python src/report.py

echo "=== done. see results/ and figures/ ==="
