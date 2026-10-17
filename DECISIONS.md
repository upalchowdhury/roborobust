# DECISIONS

Routine implementation decisions made autonomously, and every deviation from `docs/spec.md`, with the reason.
Tuning history lives in [docs/dev_log.md](docs/dev_log.md) (generated from raw sweep logs).

## Deviations from the spec (and why)

| # | Spec said | What was done | Why |
|---|---|---|---|
| D1 | Reward: shaping `alpha = 1` initially (tunable on dev/validation data) | **alpha = 50**, for both conditions | alpha = 1 failed the Gate C empty-room diagnostic (2/20 vs 20/20 scripted). Per-step shaping (<= 0.0027) was smaller than the 0.01 step cost. Sweep on empty-room *validation* data: alpha 1/5/10/25/50/100 -> mean successes 1/4/8/9/17/17 of 20. 50 is the smallest value meeting >= 80% on all 3 seeds. |
| D2 | "Choose workers based on measured total training throughput and memory" | Measured 1/4/8 workers: 3.1k / 7.2k / 9.95k PPO steps/s (0.4 / 1.3 / 2.2 GB). Throughput winner is 8, but **4 workers** are used | At alpha = 50, 8 workers learned worse per transition (13.3 vs 16.7 mean successes at 200k) because there are 2x fewer policy-update rounds per transition; runtime is minutes either way, so sample efficiency decided. |
| D3 | Main experiment "start with 500k, increase to 1M if runtime allows" | **1M requested transitions** (1,003,520 actual), 5 seeds per condition (the spec's preferred study is 3; more seeds were added symmetrically before any test evaluation) | A run takes ~170 s, so both the 1M option and the preferred 3-seed study are affordable. Dev runs showed both conditions still improving at 500k and the condition ordering flipped between 500k and 1M for one seed. Chosen for runtime and convergence, applied identically to both conditions. |
| D4 | "500k environment transitions" | Requested steps are **rounded up to a whole rollout** (4 workers x 1024 = 4096), so 1M -> 1,003,520 | PPO collects whole rollouts; both conditions use identical n_envs/n_steps so budgets are identical. Requested and actual counts are both in `meta.json`. |
| D5 | Validation-selected checkpoints "may be secondary" | Not implemented. Periodic checkpoints (every ~50k) are saved; the ~500k checkpoint of the same runs is evaluated as a pre-declared budget-sensitivity analysis | Optional in the spec. The final checkpoint is the primary comparison. |
| D6 | Modules listed in section 9 | Added `config.py`, `runtime.py`, `validation.py`, `benchmark.py`, `scripts/` | Shared config/provenance code, physics checks shared between `validate-env` and pytest, benchmark command, auditable dev sweeps. No listed module was renamed. |
| D7 | `robot.xml` + configuration for robot parameters | `robot.xml` is the single physical asset; `configs/env.yaml` documents the same values with units and `env.py` **asserts they match** at model build | Avoids two sources of truth that can silently diverge; tested in `test_env.py::test_robot_config_matches_model`. |
| D8 | PPO `gamma = 0.99` (and the same gamma in the shaping term) | **gamma = 0.999** for both conditions | Under the original protocol 4 of 10 main-task runs (seeds 0-4 x 2 conditions) stalled into a safe idling policy (0 collisions, 85-100% timeouts, ~0% success). Pre-registered sweep on dev seeds 10-13 (both conditions, 1M transitions): stalls base(0.99)=3 of 8, gamma .995=0, gamma .999=0, ent_coef .005=0; the pre-declared tie-break (mean nominal validation success: 33.3 vs 32.9 vs 26.4 of 40) selected 0.999 — a near-tie with 0.995, so the exact value is not claimed to be special. Reasoning: at 0.99 a +10 bonus 300 steps away is worth ~0.5 now while crashes are punished immediately, making idling an attractor (dev log section 2). |

## Design decisions (mine, not dictated)

* **Motor multiplier semantics.** The multiplier scales the motor's whole torque output: velocity-actuator gain, bias *and* torque limit together. A velocity servo hides a weak motor under light commands; only near saturation does wheel speed drop. Measured (`validate-env`): left multiplier 0.6 at full command -> left wheel 7.5 vs 9.0 rad/s, torque pinned at its 0.30 N*m limit, +0.52 m lateral deviation in 3 s; at half command only +0.05 m. The intervention is therefore real but bites mainly under demanding commands. Disclosed in README/REPORT.
* **Integrator.** `implicitfast` (timestep 0.005 s as specified). The velocity actuators' time constant (armature+wheel inertia / kv) is of the order of the timestep, which explicit Euler cannot integrate stably.
* **Contact friction.** Floor priority 0; wheel/caster geoms priority 1 so their own friction coefficients are used (MuJoCo otherwise takes the max of the pair, which would make the "low-friction" casters grip).
* **Ray casting.** Rays at 0.10 m height, group mask 0 only (walls/obstacles), robot body excluded. `mj_multiRay`'s `cutoff` does **not** truncate distances (found by the analytic test: a wall 3.8 m away read 3.8), so readings are clipped to 3.0 m explicitly. Range is measured from the robot centre.
* **Collision detection.** Contacts are checked after *every* physics substep and again at the final pose (contacts found inside `mj_step` belong to the pre-integration pose). First robot-to-wall/obstacle contact terminates immediately with substep/geom/position logged.
* **Verifier priority.** collision > tipped > out_of_bounds > success. Timeout is applied by the env only when no terminal event occurred on that step.
* **RNG streams.** One `SeedSequence` per env spawns independent geometry / domain / measurement-noise generators. Nominal and randomized runs with the same seed see the *same scenario sequence*. Evaluation pins scenario, domain parameters and noise seed (hash of eval seed + scenario id + condition, identical for every policy; Python's salted `hash()` is never used).
* **Scenarios.** One distinct layout per main scenario (so "50 held-out geometries" means 50 layouts). Disjointness is checked on the canonical 5 cm-quantised layout *and* a minimum parameter-space separation (0.25 m) between layouts of different splits, to prevent near-duplicate leakage. 70% of scenarios request a blocked straight start-goal line. Obstacles keep >= 0.7 m corridors; the connectivity check never rejected anything for that reason, so it is exercised by a constructed disconnected room in the tests. The three-obstacle / narrow-passage family and friction shifts (optional) were not built.
* **Test-split hygiene.** The test split was never evaluated during development. Every evaluation of a manifest containing test scenarios is appended to `results/test_access_log.jsonl`. `generate-scenarios` refuses to regenerate existing manifests without `--force`.
* **Trajectory logging rule.** Pinned scenarios (first three) for every policy x condition, plus first success and first two failures in scenario order per policy x seed x condition. `--save-trajectories all` saves everything.
* **Failure taxonomy.** Automatic multi-label tags from logged features with documented thresholds (`analyze.TAXONOMY`); one extra tag (`stalled_before_obstructed_route`) because most timeouts were freezes in front of blocked routes. Manual labels can be merged.

## Protocol freeze (final; written before any test evaluation)

Frozen: reward (alpha 50, bonus 10, penalty 10, step cost 0.01, **gamma 0.999**), PPO settings (spec defaults except gamma), 4 workers, 1M requested transitions (1,003,520 actual),
**seeds 0-4 per condition** (five instead of the spec's preferred three: runtime allowed it and validation had shown large seed variance; added symmetrically for both conditions, before any test score existed),
final checkpoint primary, `configs/test_suite.yaml` (conditions, eval seed, primary endpoint = safe arrival under `combined_stress`).
Secondary, pre-declared: the ~500k-transition checkpoint of the same runs (budget sensitivity) and every single condition.
The earlier gamma = 0.99 runs are archived in `runs/archive_gamma099/` and reported in `docs/dev_log.md`; they are not part of the final comparison.
If test results motivate any change, the test set becomes development data and a new one must be generated.

## Stability sweep: selection rule (written BEFORE any sweep result was seen)

Trigger: with the original protocol (alpha 50, gamma 0.99, ent_coef 0), seeds 0-2 on the *validation* split gave nominal 70/80/**0**% and randomized 75/**17.5**/60% (1M transitions).
The failed seeds had ~0 collisions, 85-100% timeouts and ~0% training success throughout, i.e. they settled into a safe idling policy.

Sweep (`scripts/dev_main_sweep.py`): configs `base` (gamma .99), `g995` (gamma .995), `g999` (gamma .999), `ent005` (ent_coef .005), each for BOTH conditions
on four dev seeds 10-13 (disjoint from the final study's seeds), 1M transitions, scored on the 40 main validation scenarios across the six conditions.

Rule: a run is a *stall* if its nominal-condition validation success is < 8/40 (20%). Choose the config with the fewest stalled runs summed over both conditions;
ties broken by higher mean nominal-condition validation success over its 8 runs. If no config has fewer stalls than `base`, keep `base`.
The chosen config applies identically to both conditions. Test data play no role. After the choice, the final study is (re)trained with seeds 0-4 per condition
and the protocol is re-frozen before the test split is evaluated; the earlier seeds 0-4 runs under the original protocol are kept and reported in the dev log as the
pre-change evidence.

## Provenance note

`runtime.git_info()` initially recorded the literal string `"HEAD"` as the git commit in repositories with no commits (`git rev-parse HEAD` prints "HEAD" and fails). This was fixed after the
final runs were trained; the `meta.json` of runs `nominal_s*`/`randomized_s*` and `results/*/eval_meta.json` therefore contain `git.commit = "HEAD"`, which means **no commit existed**.
The report validates the recorded value (40 hex characters) and prints "none" otherwise; the *source code hash* (sha256 over the package's .py/.xml files) is the code identifier for these results.
Provenance records were not edited by hand and the runs were not retrained for a metadata field.

Update: the repository's first commit (`ab4d0b4`, 2026-10-17 11:48, made by the repository owner; commit hashes in this repository changed once when author identities were rewritten to the owner's GitHub noreply address, file contents were unchanged) postdates all training (final runs started 11:08) and both test evaluations (11:24-11:25), so no
valid commit is recorded with any result; the source code hash in each `eval_meta.json` identifies the code that produced it. Verified from file modification times: the environment, scoring and training code
(`env.py`, `train.py`, `verifier.py`, `rewards.py`, `sensors.py`, `randomization.py`, `scenarios.py`, `baselines.py`, `robot.xml`; last modified 10:38 or earlier) did not change after the final training began;
only metadata, report-generation and replay code (`evaluate.py`, `runtime.py`, `analyze.py`, `cli.py`, `replay.py`) was edited afterwards, so HEAD is not byte-identical to the code that ran the experiments.
Changes after that commit (viewer timer, demo assets, ablation configs, final-artifact tests, documentation) are uncommitted at the time of writing.
