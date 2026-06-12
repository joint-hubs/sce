#!/bin/bash
# SCE overnight batch on a GCP VM. Idempotent: safe to re-run after preemption.
# Results sync to GCS every 5 min (partial) and in full at the end, then the VM powers off.
set -x
exec > /var/log/sce-night.log 2>&1

BUCKET=gs://sce-night-dochubs

apt-get update
apt-get install -y git python3-venv python3-pip libgomp1

rm -rf /opt/sce
cd /opt
git clone --depth 1 https://github.com/joint-hubs/sce.git
cd sce
git rev-parse HEAD

python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -e ".[models,viz]"

mkdir -p data/parquet results/diagnostics
gsutil -m cp "$BUCKET/data/*.parquet" data/parquet/
gsutil -m cp -r "$BUCKET/diagnostics/*" results/diagnostics/

# Partial sync so a spot preemption loses at most ~5 minutes of artifacts
(while true; do sleep 300; gsutil -m rsync -r results "$BUCKET/partial/results" >/dev/null 2>&1; done) &
SYNC_PID=$!

LOG=results/night_run_full.log
echo "START $(date)" >> "$LOG"

run_step() {
  echo "=== $* :: $(date)" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
}

# 1. Cross-model categorical-mode comparison (fast GBDTs first, slow sklearn last)
for m in xgboost lightgbm catboost ridge gradient_boosting random_forest extra_trees; do
  for d in rental_poland_short m5_store_dept_daily melbourne_housing walmart_weekly rossmann_daily; do
    run_step $PY scripts/run.py --dataset "$d" --compare-categorical-modes --model-type "$m" --run-grade report-grade
  done
done

# 2. Feature-combination search (default model)
for d in rental_poland_short m5_store_dept_daily melbourne_housing walmart_weekly rossmann_daily; do
  run_step $PY scripts/run.py --dataset "$d" --search --run-grade report-grade
done

# 3. UAE full-data diagnostics
for d in experimental/sales_uae_transactions experimental/rental_uae_contracts; do
  for g in permuted_target shuffled_groups crossfit_ab; do
    run_step $PY -m scripts.diagnostics."$g" --dataset "$d" --run-grade report-grade
  done
done

# 4. Aggregate cross-model summary (fresh runs only on this clean clone)
run_step $PY scripts/generate_categorical_mode_batch_summary.py --latest

# 5. Regenerate figures
run_step $PY scripts/generate_figures.py
run_step $PY scripts/generate_summary_figures.py
run_step $PY scripts/generate_paper_appendix_figures.py

# 6. Full overnight report
run_step $PY scripts/night_report.py

echo "DONE $(date)" >> "$LOG"
kill $SYNC_PID

tar czf /tmp/night_results.tar.gz results docs/figures
gsutil cp /tmp/night_results.tar.gz "$BUCKET/done/night_results.tar.gz"
gsutil cp /var/log/sce-night.log "$BUCKET/done/sce-night.log"
echo "done $(date)" | gsutil cp - "$BUCKET/done/DONE"

shutdown -h now
