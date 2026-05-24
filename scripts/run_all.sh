#!/usr/bin/env bash
# Reproduces all Tier 1 experiments + ablations + agent task.
# Resume-safe: skips any config+model+seed that already has a results.json.
set -e

RESULTS_DIR="${RESULTS_DIR:-results}"
LOG="$RESULTS_DIR/run.log"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

CONFIGS=(
  configs/synthetic_chain.yaml
  configs/synthetic_fork.yaml
  configs/synthetic_collider.yaml
  configs/synthetic_er_k5.yaml
  configs/synthetic_er_k10.yaml
  configs/synthetic_er_k20.yaml
)

log "=== AC-LSCM Full Run ==="
log "Results dir: $RESULTS_DIR"

# --- Main experiments ---
for CFG in "${CONFIGS[@]}"; do
  log "Config: $CFG"
  python -m src.train \
    --config "$CFG" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    2>&1 | tee -a "$LOG"
done

# --- Ablations (AC-LSCM only on ER K=10) ---
log "=== Ablations ==="
python -m src.train \
  --config configs/ablations.yaml \
  --ablations-base-config configs/synthetic_er_k10.yaml \
  --results-dir "$RESULTS_DIR" \
  --resume \
  2>&1 | tee -a "$LOG"

log "=== All runs complete ==="
log "Generating tables..."
python scripts/make_tables.py --results-dir "$RESULTS_DIR" --output tables/
log "Done. Tables in tables/"
