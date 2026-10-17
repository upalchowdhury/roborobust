# RoboRobust

A local MuJoCo differential-drive navigation environment, PPO training, and a paired evaluation of whether
**training-time domain randomization** (wheel-motor strength + range-sensor noise) improves *safe arrival* under held-out
actuator and sensor shifts.

**Honest scope.** Simulation only. No physical robot, no sim-to-real claim. Goal position and body velocities are
*privileged simulator-derived inputs* (the policy does not perceive them through sensors), and the policy is a feedforward MLP.
Results are in [REPORT.md](REPORT.md); the tuning history, including the failures, is in [docs/dev_log.md](docs/dev_log.md);
every deviation from the spec is in [DECISIONS.md](DECISIONS.md).

## Setup (macOS, Apple silicon, Python 3.11, CPU only)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'            # tested versions are frozen in requirements.lock
roborobust doctor                  # python/arch, imports, MuJoCo stepping, optional renderer
roborobust validate-env            # Gate A/B physics + sensor + manifest checks (writes measurements to JSON)
pytest -q
```

Internet is needed only to install packages; training and evaluation then run offline. `python -m roborobust ...` is identical to `roborobust ...`.
Training is headless (normal Python). The interactive viewer needs `mjpython` (shipped with the `mujoco` wheel); `doctor` reports whether it is on `PATH`.
Rendering problems never block training.

## The task

A 6 m x 6 m enclosed room, a cylindrical robot (radius 0.18 m, 5.1 kg) with two driven wheels (radius 0.05 m, track 0.30 m) and two low-friction ball casters.
The robot must drive to a goal at least 1 m away and **stop** there, without touching a wall or obstacle. 0-2 axis-aligned rectangular obstacles; frozen within an episode.

| quantity | value |
|---|---|
| physics step / control step | 0.005 s / 0.05 s (10 substeps per action), integrator `implicitfast` |
| episode cutoff | 600 control steps = 30 s (external cutoff: *truncation*, not termination) |
| actuators | bounded wheel-velocity servos, kv 0.4 N·m·s/rad, torque limit 0.5 N·m, target up to 10 rad/s = 0.5 m/s ideal (about 0.45 m/s delivered) |
| wheel drivetrain | damping 0.04 N·m·s/rad, armature 0.001 kg·m²; wheel friction 1.0, caster friction 0.01 |

Wheel contact with the floor produces all motion (no kinematic position updates). Parameters live in `configs/env.yaml` with units; `env.py` asserts they match `src/roborobust/robot.xml`.

### Action space
`Box(-1, 1, shape=(2,), float32)`: left/right wheel velocity targets (x 10 rad/s). Non-finite actions raise; finite out-of-range actions are clipped and counted (`action_clipped`, `clip_count`).
Commanded targets, measured wheel speeds and actuator torques are recorded in `info`.

### Observation space (29 float32 values)
| index | meaning |
|---|---|
| 0-23 | planar range rays / 3 m, 15° apart, ray 0 straight ahead, counter-clockwise; robot geometry and the goal marker are excluded; no hit -> 1.0 |
| 24 | goal distance / room diagonal (8.49 m), clipped to [0, 1] |
| 25, 26 | sin, cos of the goal bearing relative to heading |
| 27 | body-frame forward velocity / 0.5 m/s, clipped to [-1, 1] |
| 28 | yaw rate / 3.5 rad/s, clipped to [-1, 1] |

Goal bearing, goal distance and velocities are **privileged**. Range noise (Gaussian, then clipped to [0, 3] m) affects only indices 0-23.
The policy sees observations only: never scenario metadata, obstacle maps, sampled motor gains, reward internals or verifier state.
Hidden motor gains and noisy ranges make the problem partially observed; the feedforward policy is a deliberate baseline.

### Terminal semantics
* **Success** (verifier, from true simulator state): centre within 0.30 m of the goal **and** planar speed <= 0.10 m/s for **5 consecutive** control steps, with no wall/obstacle contact ever.
* **Collision**: any robot geom touching a wall/obstacle geom in *any physics substep*; stops immediately (substep, geoms and contact point are logged). Floor and wheel/caster contacts never count.
* **Tipped** (> 45°) and **out_of_bounds** are the other task failures. Priority: collision > tipped > out_of_bounds > success.
* `terminated=True` for success or failure. `truncated=True` only for the 600-step cutoff when no terminal event occurred on that step. One `terminal_reason` per episode:
  `success | collision | tipped | out_of_bounds | timeout`. Non-finite simulator state raises `SimulationError` (never silently dropped).
* Stepping a finished episode raises until `reset()`. Every episode counter, previous action and sampled parameter is reset on every `reset()`.

`verifier.py` imports nothing from `rewards.py`: reward thresholds can never determine success (enforced by a test).

### Reward (training signal only)
```
r = +10 [success]  - 10 [collision | tip | out of bounds]  - 0.01  + alpha * ( gamma * Phi(s') - Phi(s) )
Phi(s) = - goal_distance / room_diagonal          Phi(terminal) = 0          gamma = PPO discount
```
On time-limit truncation the true next potential is kept and SB3 bootstraps through it (`TimeLimit.truncated`); this is checked numerically in `tests/test_env.py`.
Assumptions of potential-based shaping and the limits of this reward are discussed in `rewards.py` and [docs/dev_log.md](docs/dev_log.md) (section 2):
**training return is not task performance**. Idling at a distance earns positive shaping per step when gamma < 1, so return is never reported as a success metric.

## Randomization and held-out shifts

Training (independent per episode, at reset): left and right motor multipliers ~ U(0.8, 1.2) and range-noise std ~ U(0, 0.03 m).
The multiplier scales the motor's whole torque output (servo gain **and** torque limit); nominal arrays are restored before every application so nothing accumulates.
Because a velocity servo compensates a weak motor under light commands, the effect is strongest under demanding commands:
with the left multiplier at 0.6 and a full-speed command, the left wheel runs 7.5 instead of 9.0 rad/s with its torque pinned at the limit, bending the path 0.5 m sideways in 3 s,
while at half command the same multiplier moves the path 0.05 m (`roborobust validate-env`).

| held-out condition | left / right motor multiplier | range noise std |
|---|---|---|
| nominal | 1.0 / 1.0 | 0 m |
| within training support | 0.85 / 1.10 | 0.02 m |
| weak left motor | 0.60 / 1.0 | 0 m |
| weak right motor | 1.0 / 0.60 | 0 m |
| noisy sensing | 1.0 / 1.0 | 0.06 m |
| combined stress | 0.60 / 1.0 | 0.06 m |

These are proposed stress levels, not calibrated hardware distributions. The scripted controller keeps >= 95% on validation under all of them, so the robot stays controllable.

## Scenarios

`roborobust generate-scenarios` writes frozen JSONL manifests with SHA-256 hashes (`data/scenarios/`): main task 200 train / 40 validation / 50 test
(one distinct layout per scenario, 1-2 obstacles, random start pose and heading, goal >= 1 m away), plus an empty-room family (200 train / 20 validation) labelled separately.
Layouts are canonicalised (5 cm grid) and verified disjoint across splits, with an extra minimum separation so near-duplicates cannot leak. Start/goal clearance and start-goal connectivity are checked on a grid inflated by robot radius + margin
(an *approximate* feasibility check: connected on the grid does not guarantee dynamic solvability). The generator refuses to overwrite existing manifests without `--force`.

## Results at a glance (simulation only; full tables in [REPORT.md](REPORT.md))

50 held-out layouts, 5 training seeds per condition, 1,003,520 transitions per run, final checkpoints, deterministic inference. Safe-arrival rate (mean over seeds; range over seeds in brackets):

| policy | nominal condition | combined stress (**primary endpoint**) |
|---|---|---|
| PPO-nominal | 85.2% [72-94] | 76.0% [64-84] |
| PPO-randomized | 83.2% [66-94] | 76.4% [70-82] |
| Scripted-reactive (same observation, no learning) | 96.0% | 98.0% |
| Random | 0% | 0% |

* **The hypothesis is not supported at this scale.** Primary-endpoint difference (randomized minus nominal) is +0.4 pp; 95% bootstrap interval [-5.2, +6.0] over scenarios and [-10.0, +11.2] when training seeds are resampled too. No difference could be detected; that does not show none exists.
* **The comparison is fragile.** At the ~500k-transition checkpoints of the same runs (pre-declared secondary analysis) PPO-nominal was ahead under combined stress (81.2% vs 75.2%, -6.0 pp, [-12.4, +0.4]).
* **A hand-coded reactive controller beats both learned policies** (96-98%). RL was not needed for this task as specified; the PPO policies are a working learning baseline, not a strong navigator.
* About 70% of PPO failures are collisions, mostly with obstacles en route (nominal 74%, randomized 68% of failures); the rest are timeouts. The two conditions' failure profiles are very similar.
* **Training reliability was the real finding.** With the spec's starting gamma = 0.99, 4 of 10 main-task runs (both conditions) converged to a safe idling policy and scored ~0%; at gamma = 0.999 none of the 10 did. This was found and fixed on validation data, with the selection rule written down before the sweep ([docs/dev_log.md](docs/dev_log.md), [DECISIONS.md](DECISIONS.md)).

What this cannot establish: sim-to-real transfer; that randomization never helps (stress levels here are mild, the motor effect bites mainly under near-saturating commands, and the budget/seed variance is large relative to any effect);
attribution to motor randomization versus noise randomization (one combined intervention).

## Replaying a trajectory

Saved trajectories are in `results/weekend/trajectories/` (319 files: pinned scenarios plus the first success and first two failures per policy, seed and condition). Example (a PPO-randomized collision under combined stress):

```bash
mjpython -m roborobust replay --episode results/weekend/trajectories/randomized_s0__combined_stress__main_test_0004.npz
python -m roborobust replay --episode results/weekend/trajectories/randomized_s0__combined_stress__main_test_0004.npz --gif out.gif   # headless
python -m roborobust replay --episode results/weekend/trajectories/randomized_s0__combined_stress__main_test_0004.npz --resimulate-check
```

The viewer restores recorded simulator states (qpos/qvel at each control boundary); it does not re-simulate. `--resimulate-check` re-runs the recorded actions separately: for this episode it reproduces the recorded states with 0.0 divergence on this machine.
Headless PNG/GIF output and the re-simulation check were exercised; the interactive `mjpython` window itself could not be opened in the session that built this repo, so run it once yourself (under plain `python` it prints this guidance instead of crashing).
Pre-rendered versions of this example are in `reports/weekend/replays/`.

## What you need to be able to explain (spec section 14)

The code does not make you an expert on its own. Where to look for each point, so you can verify instead of trust:

| # | you should be able to explain | evidence in this repo |
|---|---|---|
| 1 | how wheel torque and contact turn actions into motion, and how that was checked | `robot.xml`; `validation.py`; `roborobust validate-env` (straight / rotation sign / wall blocking / weak-motor measurements) |
| 2 | why training reward and evaluation success are separate | `verifier.py` imports nothing from `rewards.py` (test); dev log section 2: at the original gamma = 0.99, undiscounted return was *higher* for timeouts (81.5) than for successes (55.3) |
| 3 | why truncation and termination bootstrap differently | `tests/test_env.py::test_ppo_bootstraps_value_through_time_limit_but_not_through_terminal` (checks `r + gamma*V(s_T)` numerically) |
| 4 | how a policy might exploit the reward | dev log sections 1-2, 4-5: weak shaping at alpha = 1; the idling attractor at gamma = 0.99 |
| 5 | why test layouts, seeds and checkpoint choice matter | frozen manifests with hashes; `results/test_access_log.jsonl`; protocol freeze in DECISIONS.md; the 500k vs 1M reversal |
| 6 | why noise + hidden motor gains make it partially observed | README observation section; `test_motor_gains_are_not_visible_in_the_observation` |
| 7 | what the reactive controller reveals | Scripted-reactive row above; `baselines.py` |
| 8 | whether randomization helped, where it hurt | REPORT.md, including the exploratory robustness-gap table and the 500k secondary analysis |
| 9 | why simulation robustness is not hardware transfer | disclosures above; stress levels are uncalibrated |
| 10 | one follow-up experiment that would discriminate failure explanations | REPORT.md "Limits": train both conditions longer (or add memory) and re-run on a *newly generated* held-out set; cheaper first step: motor-only vs noise-only ablations |

Two things in the spec are deliberately left to you: making **one meaningful design decision yourself** (candidates: the stress levels, a larger success bonus versus a higher gamma, a recurrent policy)
and **reproducing one failure investigation** (`scripts/dev_sweep.py` and `scripts/dev_main_sweep.py` re-create the alpha and gamma investigations end to end in minutes). Disclose AI assistance if asked,
and keep your experimental decisions distinct from the generated implementation.

## Reproduce

See [REPRODUCE.md](REPRODUCE.md) for the exact command sequence, runtimes and file locations of everything below.

## Repository map

```
src/roborobust/  env.py robot.xml scenarios.py sensors.py randomization.py verifier.py rewards.py baselines.py
                 train.py evaluate.py analyze.py replay.py cli.py  (+ config.py runtime.py validation.py benchmark.py)
configs/         env.yaml scenarios.yaml base_train.yaml nominal.yaml randomized.yaml *_suite.yaml
data/scenarios/  frozen manifests + manifest_hashes.json
tests/           physics, sensors, verifier, rewards, env contract, scenarios, experiment integrity, statistics
scripts/         dev sweeps (auditable tuning), main training launcher, dev-log generator
docs/            spec.md (the task spec), dev_log.md (tuning history and failures)
```
