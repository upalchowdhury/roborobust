# Implementation report

Everything below ran locally on the target machine (Mac Studio, arm64, macOS, Python 3.11.15, CPU only). Versions are frozen in `requirements.lock`.

## What ran, and how long it took

| step | result | time |
|---|---|---|
| `roborobust doctor` | all required checks pass; offscreen renderer works; `mjpython` present | seconds |
| `roborobust validate-env` | 9/9 physics, sensor and manifest checks pass (measurements in `results/validation/physics_checks.json`) | ~10 s |
| `pytest` | 85 tests pass (physics, sensors, verifier adversarial cases, rewards/shaping identities, env contract, determinism, randomization hygiene, SB3 bootstrapping, scenarios, experiment integrity, statistics, final-artifact integrity) | ~14 s |
| scenario manifests | 200/40/50 main + 200/20 empty; disjoint layouts, valid starts/goals, hashes recorded | ~10 s |
| throughput benchmark | PPO steps/s: 3.1k (1 worker), 7.1k (4), 10.1k (8); memory 0.4 / 1.3 / 2.2 GB | ~2 min |
| smoke run (12,288 transitions) | saves, reloads bit-identically, finite losses | 2 s |
| Gate C empty-room diagnostic, final protocol | 18/20, 18/20, 17/20 on validation (target >= 80%); scripted 20/20, random 0/20 | ~37 s per run |
| main training, 5 seeds x 2 conditions, 1,003,520 transitions each | all 10 complete; reload check and finite losses pass | ~165 s per run (each pair concurrent) |
| held-out evaluation: 12 policies x 6 conditions x 50 layouts = 3,600 episodes | complete, integrity checks pass | ~40 s (8 workers) |
| secondary evaluation at the ~500k checkpoints (3,300 episodes) | complete | ~35 s |
| report generation | tables, plots, REPORT.md from raw logs | ~10 s each |

## What failed along the way (and was fixed, not hidden)

1. **Sensor bug:** MuJoCo's ray `cutoff` does not truncate distances, so a wall 3.8 m away read 3.8 (observation > 1). Found by the analytic ray test; clipped explicitly.
2. **Gate C failed at the spec's reward scale** (alpha = 1): 2/20 safe arrivals. Cause: per-step progress reward smaller than the step cost. Fixed by alpha = 50 on validation data.
3. **Training stalls at the spec's discount (gamma = 0.99):** 4 of 10 main-task runs idled. Fixed by gamma = 0.999, chosen by a rule written before the sweep (0 of 10 stalled afterwards).
4. Several test and report-generator defects (wrong expectations in my own tests, NaN-as-truthy trajectory paths, a hard-coded seed count, a git-commit provenance quirk) were caught by review and fixed; see `DECISIONS.md`.

## Result

No detectable benefit of domain randomization on the primary endpoint (combined stress): 76.0% vs 76.4%, difference +0.4 pp, 95% interval [-5.2, +6.0] (scenarios) / [-10.0, +11.2] (scenarios and seeds). The comparison reverses at the ~500k checkpoints. A reactive controller scores 96-98%. See `REPORT.md`.

## Reproduction

`REPRODUCE.md` lists the exact commands in order. The experiment is deterministic given the seeds: re-evaluating the same checkpoints reproduces identical episode outcomes; training with the same seed reproduced identical diagnostic scores on re-run.

## Remaining limitations and not done

Simulation only; privileged goal/velocity observations; feedforward policy; uncalibrated stress levels; one combined intervention; five training seeds; coarse 3-value gamma sweep on four dev seeds.
Not done: motor-only/noise-only ablations (configs provided; they need a freshly generated held-out set), friction shifts, the three-obstacle family, validation-selected checkpoints, the demo video (a script and assets are in `DEMO.md` / `demo/`),
recording a git commit with the results (the repository's first commit was made by its owner after all training and test evaluation; later edits to reporting/replay code and the demo/ablation additions are not in it), and a visual confirmation of the interactive viewer window (it launched and exited cleanly under `mjpython`).
