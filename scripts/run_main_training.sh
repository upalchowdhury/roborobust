#!/usr/bin/env bash
# Main paired training: for each seed, nominal and randomized are trained CONCURRENTLY (equal machine
# contention for both conditions), 1M requested transitions each (actual = 245 whole rollouts = 1,003,520).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
# Final protocol: seeds 0-4. (Seeds 0-2 were the original plan; 3-4 were added for BOTH conditions before any test evaluation:
# validation showed large seed variance and the spec allows more seeds if runtime allows.)  Usage: SEEDS="3 4" scripts/run_main_training.sh
for s in ${SEEDS:-0 1 2 3 4}; do
  roborobust train --config configs/nominal.yaml    --seed "$s" --steps 1000000 --run-id "nominal_s$s"    > "runs/log_nominal_s$s.txt" 2>&1 &
  roborobust train --config configs/randomized.yaml --seed "$s" --steps 1000000 --run-id "randomized_s$s" > "runs/log_randomized_s$s.txt" 2>&1 &
  wait
done
echo ALL_DONE
