#!/bin/bash
# Targeted rerun for the ONE step that OOM-killed in the overnight batch:
# rental_uae_contracts full-data diagnostics (5.48M rows). The main batch ran on
# a 64 GB VM and exit=137'd here; this runs the same three diagnostics on a
# high-memory VM and uploads to a SEPARATE marker so it APPENDS to the existing
# results instead of overwriting them. Idempotent: safe to re-run after preemption.
set -x
exec > /var/log/sce-rental-uae.log 2>&1

BUCKET=gs://sce-night-dochubs
OUT=done_rental_uae   # separate prefix — does not touch done/ from the main batch

apt-get update
apt-get install -y git python3-venv python3-pip libgomp1 time

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
# Only the parquet we actually need for this dataset.
gsutil -m cp "$BUCKET/data/rental_uae_contracts.parquet" data/parquet/
gsutil -m cp -r "$BUCKET/diagnostics/*" results/diagnostics/ || true

# Partial sync so a spot preemption loses at most ~5 minutes of artifacts.
(while true; do sleep 300; gsutil -m rsync -r results "$BUCKET/partial_rental_uae/results" >/dev/null 2>&1; done) &
SYNC_PID=$!

LOG=results/rental_uae_run.log
echo "START $(date)" >> "$LOG"
# Report peak memory so we know whether the chosen machine had headroom.
free -h >> "$LOG"

run_step() {
  echo "=== $* :: $(date)" >> "$LOG"
  /usr/bin/time -v "$@" >> "$LOG" 2>&1
  echo "exit=$?" >> "$LOG"
}

for g in permuted_target shuffled_groups crossfit_ab; do
  run_step $PY -m scripts.diagnostics."$g" --dataset experimental/rental_uae_contracts --run-grade report-grade
done

echo "DONE $(date)" >> "$LOG"
free -h >> "$LOG"
kill $SYNC_PID

# Upload only the new diagnostics + the run log under the separate prefix.
tar czf /tmp/rental_uae_results.tar.gz results/diagnostics results/rental_uae_run.log
gsutil cp /tmp/rental_uae_results.tar.gz "$BUCKET/$OUT/rental_uae_results.tar.gz"
gsutil cp /var/log/sce-rental-uae.log "$BUCKET/$OUT/sce-rental-uae.log"
echo "done $(date)" | gsutil cp - "$BUCKET/$OUT/DONE"

shutdown -h now
