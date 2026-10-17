# Reproducing everything

All commands run from the repository root with the virtualenv active (`source .venv/bin/activate`). Everything is CPU-only and offline after installation.
Runtimes are what was measured on a Mac Studio (arm64, 28 cores, 96 GB) and are indicative only.

## 0. Setup and correctness gates (seconds to a minute)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                 # or: pip install -r requirements.lock && pip install -e . --no-deps
roborobust doctor
roborobust generate-scenarios --config configs/scenarios.yaml   # refuses: manifests are frozen; delete data/scenarios/manifest_hashes.json or pass --force to regenerate
roborobust validate-env                 # Gate A/B physics + sensor + manifest checks -> results/validation/physics_checks.json
pytest -q                               # ~15 s
```

## 1. Throughput benchmark (about 2 minutes)

```bash
roborobust benchmark --workers 1 4 8    # -> results/benchmark.json
```

## 2. Training ladder

```bash
# smoke (10k requested -> 12,288 transitions, ~2 s): saves, reloads, finite losses; no competence claim
roborobust train --config configs/nominal.yaml --seed 0 --steps 10000 --run-id smoke

# empty-room diagnostic (Gate C), 200k transitions, ~35 s each, then score on the 20 empty-room validation cases
roborobust train --config configs/empty_diag_nominal.yaml --seed 0 --steps 200000 --run-id diag_empty_s0
roborobust evaluate --suite configs/empty_val_suite.yaml --runs runs/diag_empty_s0 --baselines scripted random --output results/diag_empty_s0

# main study: 5 seeds x 2 conditions, 1M requested transitions (1,003,520 actual), ~165 s per run, each seed's pair runs concurrently
scripts/run_main_training.sh                      # seeds 0-4 by default; SEEDS="3 4" scripts/run_main_training.sh runs a subset
# single runs:
roborobust train --config configs/nominal.yaml    --seed 0 --steps 1000000 --run-id nominal_s0
roborobust train --config configs/randomized.yaml --seed 0 --steps 1000000 --run-id randomized_s0
```

## 3. Development (validation split only)

```bash
roborobust evaluate --suite configs/val_suite.yaml --runs runs/nominal_s0 runs/randomized_s0 --baselines scripted random --output results/dev_val
python scripts/dev_sweep.py --alpha 50 --seed 0 --tag a50                 # empty-room reward-scale sweep (Gate C debugging)
python scripts/dev_main_sweep.py --cond randomized --seed 10 --tag g995 --gamma 0.995   # main-task stability sweep
python scripts/make_dev_log.py                                              # regenerates docs/dev_log.md from raw logs
```

## 4. Final held-out evaluation (the test split is touched only here; every touch is logged to results/test_access_log.jsonl)

```bash
# primary: final checkpoints (~40 s with 8 workers)
roborobust evaluate --suite configs/test_suite.yaml \
    --runs runs/nominal_s0 runs/nominal_s1 runs/nominal_s2 runs/nominal_s3 runs/nominal_s4 \
           runs/randomized_s0 runs/randomized_s1 runs/randomized_s2 runs/randomized_s3 runs/randomized_s4 \
    --baselines scripted random --output results/weekend --workers 8

# pre-declared secondary: the ~500k-transition checkpoint of the same runs (budget sensitivity)
roborobust evaluate --suite configs/test_suite.yaml \
    --runs runs/nominal_s0 runs/nominal_s1 runs/nominal_s2 runs/nominal_s3 runs/nominal_s4 \
           runs/randomized_s0 runs/randomized_s1 runs/randomized_s2 runs/randomized_s3 runs/randomized_s4 \
    --checkpoint checkpoints/step_000503808.zip --baselines scripted --output results/weekend_500k --workers 8

roborobust report --input results/weekend_500k --output reports/weekend_500k
roborobust report --input results/weekend --output reports/weekend --secondary reports/weekend_500k
python scripts/publish_report.py weekend     # copies reports/weekend/REPORT.md to ./REPORT.md with working image links
```

`reports/weekend/` then contains `REPORT.md`, `results.csv` (per policy x seed x condition), `results_across_seeds.csv`, `episode_results.jsonl`,
`comparison.png`, `learning_curves.png`, `failure_examples.png`, `paired_examples.png`, `failure_taxonomy.csv`, `episode_tags.csv`,
`paired_bootstrap.csv`, `robustness_gap_did.csv`. Evaluation is deterministic: re-running it reproduces identical episode outcomes.

## 5. Replaying a trajectory

Saved trajectories hold the simulator integration state at every control boundary, so the viewer *restores* recorded frames (it does not re-simulate).

```bash
mjpython -m roborobust replay --episode results/weekend/trajectories/<episode_id>.npz     # interactive viewer (macOS needs mjpython)
python  -m roborobust replay --episode results/weekend/trajectories/<episode_id>.npz --png path.png    # headless top-down plot
python  -m roborobust replay --episode results/weekend/trajectories/<episode_id>.npz --gif path.gif    # headless offscreen render
python  -m roborobust replay --episode results/weekend/trajectories/<episode_id>.npz --resimulate-check   # re-runs recorded actions, reports state divergence
```
