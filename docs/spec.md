# RoboRobust: robotics RL environment and robustness evaluation

## Purpose and honest claim

Build a local MuJoCo differential-drive navigation environment, train PPO policies, and evaluate whether training-time domain randomization improves safe arrival under held-out actuator and sensor shifts.

Target: Apple Silicon Mac Studio, native macOS, Python 3.11, CPU training, no physical robot, cloud compute, API keys, CUDA, or paid services. Dependency installation requires internet. After installation, training and evaluation must work offline.

This is a simulation research portfolio project, not evidence of real-world robot deployment. A weekend should produce a functioning environment and preliminary experiment. A convincing multi-seed study may require additional runtime. Never invent results or promise convergence by a deadline.

## Instructions to the coding agent

Implement this specification in a new repository. Work through the gates below in order. Build and run real code; do not stop at scaffolding. Make routine implementation decisions autonomously and record deviations in DECISIONS.md. Do not add unrelated product features. Do not silently replace MuJoCo physics with kinematic position updates. Do not claim tests or training ran if the execution environment prevented them.

Prioritize environment correctness, training, and honest evaluation over UI polish. If your execution machine is not the user's Mac, report which checks ran locally and provide exact Mac commands for untested operations. Pin compatible package versions after resolving and testing them; do not guess version pins from this document. Use the official mujoco package, not legacy mujoco-py.

## 1. Research question

Hypothesis: a policy trained with modest variation in wheel actuator strength and range-sensor noise has higher safe-arrival rates under those shifts than a policy trained under nominal conditions, with a possible nominal-performance tradeoff.

Main comparison:

- PPO-nominal: fixed actuators, noiseless range readings.
- PPO-randomized: independent left/right actuator multipliers and range noise sampled at episode reset.
- Scripted-reactive: hand-coded goal attraction plus obstacle avoidance using the exact same observation interface as PPO.
- Random: diagnostic sanity baseline.

One-factor training ablations (motor-only and noise-only) are stretch work. Without them, attribute an observed effect to the combined training intervention, not to a specific component. This experiment cannot establish sim-to-real transfer or general robotics superiority.

## 2. Exact task and physics

Create a 6 m by 6 m enclosed room. A cylindrical robot has approximately 0.18 m body radius, two driven wheels, and a low-friction passive support. Start with a conservative small maximum speed around 0.5 m/s. Define mass, wheel radius, track width, actuator limits, friction, and damping explicitly in configuration and document their units.

Use a free body and hinge-jointed wheels. Wheel contact with the floor must produce motion through MuJoCo physics. Use bounded wheel velocity actuators with explicit force limits. The domain-randomization motor multiplier must change effective actuator force/torque capacity; verify its behavioral effect under a sufficiently demanding command. Wheel velocity control should not make the intervention inert.

Physics timestep: start at 0.005 s. Control timestep: 0.05 s (10 physics substeps per policy action). Maximum episode length: 600 control steps / 30 simulated seconds. Treat this as an external evaluation cutoff, not a hidden finite-horizon task deadline.

Goal marker is visual and noncolliding. Start and goal must be outside inflated obstacle geometry. Randomize initial heading. Initial goal distance must be at least 1 m. Freeze obstacles during each episode.

Weekend task families:

1. Empty room: control and learning sanity check.
2. One or two rectangular obstacles: main experiment.
3. Three obstacles or a narrower passage: optional geometry stress test.

Exclude mazes, moving humans, package pickup, cameras, SLAM, manipulation, batteries, ROS, and real hardware from this build.

Validate path connectivity on an occupancy grid inflated by robot radius plus margin. Reject disconnected scenarios using a deterministic bounded rejection process. Grid connectivity is an approximate feasibility check, not a guarantee of dynamic solvability; retain this limitation in the report.

## 3. Observation and action contracts

Action: float32 Box(-1, 1, shape=(2,)), mapped to bounded left and right wheel velocity targets. Reject nonfinite actions. Clip finite out-of-range actions and log clipping. Record commanded and effective actuation.

Observation: float32 vector of exactly 29 elements:

- 24 planar range rays, spaced uniformly over 360 degrees, 3 m maximum range; readings divided by 3 m.
- Goal distance divided by room diagonal and clipped to [0, 1].
- sin(goal bearing relative to robot heading).
- cos(goal bearing relative to robot heading).
- Body-frame forward velocity, divided by a documented bound and clipped to [-1, 1].
- Yaw angular velocity, similarly scaled and clipped.

Goal localization and velocity estimates are privileged simulator-derived inputs in this first version. Explicitly disclose this simplification. Do not describe the system as learning from raw real-world sensors. Range noise affects only the range readings.

Ray casting must exclude robot geometry and the nonphysical goal marker. Confirm distances against analytic simple scenes. Set ray height so obstacles are detectable and rays do not intersect the floor accidentally.

Sample sensor noise from a separately seeded stream and clip noisy ranges to valid bounds. PPO receives observations only, never scenario metadata, true obstacle maps, sampled motor gains, reward internals, or evaluator state. Hidden randomized dynamics make the observation process partially observed; a feedforward policy is a deliberate baseline, not a fully observed MDP claim.

## 4. Independent success and failure verifier

Keep verifier.py separate from rewards.py. Reward thresholds must never determine success.

Success requires center-to-goal distance <= 0.30 m AND planar speed <= 0.10 m/s for five consecutive control steps. No wall/obstacle collision may have occurred. Verify from simulator state, not the noisy observation.

Collision: any robot geometry touching a wall/obstacle geometry during any physics substep. Exclude floor and normal wheel/support contacts. Stop immediately on the first such contact; do not lose contacts between control boundaries. Collision takes priority over success on the same step.

Other terminal failure: tip angle above a configured limit (start at 45 degrees), or leaving the permitted arena. Numerical instability/nonfinite simulation state is an environment error: record it, invalidate the affected experiment, fix it, and rerun rather than quietly dropping episodes.

terminated=True for success or task failure. truncated=True only for the external step cutoff when no terminal event already occurred. Log a single explicit terminal_reason. Reset counters, previous actions, delays if added later, and all sampled parameters on every reset.

## 5. Reward design and exploitation audit

Initial proposed reward (tunable only using development/validation scenarios):

- +10 once at verified success.
- -10 at collision/tip/out-of-bounds failure.
- -0.01 per control step.
- Progress shaping: alpha * (gamma * Phi(next_state) - Phi(state)).
- Phi(state) = -goal_distance / room_diagonal; alpha=1 initially; gamma matches PPO discount.

Set terminal next-state potential to zero for genuine terminal transitions, including failures. Preserve the true next potential on time-limit truncation and ensure PPO bootstraps correctly through the library's supported time-limit handling. Do not mark timeouts as terminal failures to simplify code. Log each component and total reward.

Potential-based shaping has assumptions; document those rather than claiming it always fixes reward hacking. Do not add arbitrary proximity bonuses repeatedly near the goal. Do not automatically normalize rewards in the initial implementation.

Adversarial scorer tests:

- Staying still does not succeed.
- Entering the goal too fast does not succeed.
- Leaving the goal before five stable steps resets the dwell counter.
- Touching an obstacle and reaching the goal in the same step is failure.
- Driving closer but never arriving is not success, even with positive training return.
- Oscillating toward/away from the goal: verify discounted shaping matches its telescoping identity over a fixed trajectory; do not assert undiscounted shaping must sum to zero.
- Success bonus is paid once; stepping a finished environment raises a clear error until reset.

## 6. Scenario splits and interventions

Generate JSONL scenario manifests before main training. Include scenario_id, split, layout_id, start pose, goal, obstacle geometry, and environment version. Store a hash of each manifest. Verify that canonicalized obstacle layouts are disjoint between train, validation, and test. Empty-room control tasks are labeled separately because their geometry cannot be disjoint.

Initial main manifests: 200 training scenarios, 40 validation scenarios, and 50 held-out test scenarios. Vary starts/headings/goals as well as layouts. Use a dedicated split-generation seed. These counts are targets, not a claim of statistical power.

Training randomization, independently sampled per episode:

- Left and right motor force multipliers: Uniform(0.8, 1.2).
- Range noise standard deviation: Uniform(0, 0.03) meters; fresh Gaussian measurement noise at each observation.
- Restore nominal model arrays before applying multipliers so changes never accumulate across resets.

Test every checkpoint on the same 50 held-out geometries under each condition:

| Condition | Left/right motor multipliers | Range noise std |
|---|---|---|
| Nominal | 1.0 / 1.0 | 0 m |
| Within training support | 0.85 / 1.10 | 0.02 m |
| Weak left motor | 0.60 / 1.0 | 0 m |
| Weak right motor | 1.0 / 0.60 | 0 m |
| Noisy sensing | 1.0 / 1.0 | 0.06 m |
| Combined stress | 0.60 / 1.0 | 0.06 m |

These are proposed stress levels, not calibrated real-hardware distributions. Confirm the robot remains controllable using development scenarios; freeze values before accessing test scores. Use separate RNG streams for geometry, domain parameters, and measurement noise. Pair evaluation scenarios and noise streams across policies; streams are common exogenous randomness, not identical sensor readings along different trajectories.

Never use held-out test scores for checkpoint selection, hyperparameter tuning, reward changes, or stopping decisions. If test results motivate changes, call that test set development data and generate a new final held-out set.

Optional friction shifts require a physics microbenchmark demonstrating that the changed contact parameters materially affect motion. Do not add a friction plot whose intervention has no effect.

## 7. Training protocol and compute budget

Dependencies: mujoco, gymnasium, stable-baselines3, torch, numpy, pandas, matplotlib, pyyaml, pytest. Add recording dependencies only when needed. Use a lightweight argparse CLI. No web app is required.

Start with PPO MlpPolicy, CPU, two hidden layers of 64 units, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10, batch_size=256, n_steps=1024, ent_coef=0.0. These are starting settings, not optimal settings. Ensure rollout size is divisible by batch size.

Benchmark headless rollout throughput at 1, 4, and 8 environments. Choose workers based on measured total training throughput and memory, not core count. Set conservative torch thread counts to prevent oversubscription. Use multiprocessing spawn and a main guard on macOS. Keep rendering out of training workers.

Training ladder:

1. 10k-step smoke run: exercises saving, reload, evaluation and finite losses; no competence claim.
2. Up to 200k-step empty-room diagnostic: require evidence of learning before main experiments. A diagnostic target is >=80% safe arrival on 20 empty-room validation cases; failure triggers debugging, not fake passing.
3. Main experiment: start with 500k environment transitions per policy; increase to 1M if measured runtime allows. Count total transitions across all workers. Both conditions get the same budget and task distribution.
4. Weekend minimum: one training seed per condition, explicitly preliminary.
5. Preferred study: three seeds per condition; report each seed, not just the best.

If using empty-room warm starts, give both conditions the same paired initialization per seed and count pretraining separately. Do not give one policy extra curriculum or tuning. Save final checkpoints as the primary comparison. Validation-selected checkpoints may be secondary, with an identical predefined selection rule and evaluation frequency across conditions.

Log learning curves, entropy, approximate KL, value loss, explained variance, and return components. These support debugging; they do not replace task success metrics. Save checkpoints, configs, package versions, platform, git commit, elapsed time and transitions. Estimate remaining runtime from measured throughput; never promise specific training hours before measurement.

If learning stalls, inspect observations, wheel signs, actuator saturation, collisions, reward magnitude, and scripted feasibility first. Simplify obstacle count for both conditions and record the protocol change. Do not build a sophisticated UI around a broken experiment.

## 8. Evaluation and analysis

Primary endpoint: safe-arrival rate under the combined-stress condition on unseen layouts. Secondary endpoints: nominal safe arrival and other individual stress conditions. Specify these before running the final evaluation.

For each policy x training seed x condition, run all 50 paired scenarios with deterministic policy inference. Random baseline actions use a fixed explicit seed. Report:

- Success numerator, denominator and rate.
- Collision rate, timeout rate, other task-failure rate; outcomes must sum to episode count.
- Episode duration across all outcomes, plus time-to-goal for successes only with its denominator.
- Path length for successful episodes, reported separately from failures.
- Return and component totals, labeled training reward rather than task performance.
- Environment steps and wall-clock training cost.

Report per-training-seed results and mean/std across seeds. For one seed, Wilson intervals on task success may describe uncertainty across sampled scenarios but do not capture training variability. For paired comparison, bootstrap scenario IDs and retain paired conditions/policies within each resample. With three training seeds, optionally add a hierarchical bootstrap over seeds and scenario IDs; label estimates exploratory because three seeds offer limited precision. Do not treat every timestep or all repeated evaluations of one layout as independent evidence.

Failure taxonomy: collision while turning, collision near a passage, stopping short, circling/oscillation, timeout despite clear route, and motor-asymmetry drift. Tag automatically where defensible and allow manual labels. Include one success and at least two failures selected by a documented rule, not only flattering examples. If no improvement occurs, report the negative result and plausible explanations without asserting untested causes.

Generate results.csv, episode_results.jsonl, comparison.png, learning_curves.png and REPORT.md from actual logs. Result tables must never contain fabricated placeholders presented as measured values.

## 9. Architecture and repository contract

Use these modules (rename only with a documented reason):

- src/roborobust/env.py: Gymnasium reset/step/render lifecycle.
- src/roborobust/robot.xml: self-contained MuJoCo robot asset.
- src/roborobust/scenarios.py: generator, connectivity checks, manifests.
- src/roborobust/sensors.py: raycasts, scaling, sensor noise.
- src/roborobust/randomization.py: parameter sampling and clean restoration.
- src/roborobust/verifier.py: state-based terminal outcomes.
- src/roborobust/rewards.py: reward terms and potential handling.
- src/roborobust/baselines.py: random and observation-only scripted policies.
- src/roborobust/train.py: PPO configuration, training and checkpoint metadata.
- src/roborobust/evaluate.py: paired deterministic evaluations and raw results.
- src/roborobust/analyze.py: aggregation, intervals, plots and report.
- src/roborobust/replay.py: trajectory visualization.
- src/roborobust/cli.py: subcommands below.
- configs/: environment, nominal training, randomized training, eval suites.
- tests/: focused simulator, sensor, verifier and experiment-integrity tests.
- README.md, DECISIONS.md, REPORT.md, pyproject.toml and dependency lock file.

Implement pip-installable package entry point `roborobust` and module entry point `python -m roborobust` with identical CLI behavior.

Required command interface:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
roborobust doctor
roborobust generate-scenarios --config configs/scenarios.yaml
roborobust validate-env
pytest -q
roborobust benchmark --workers 1 4 8
roborobust train --config configs/nominal.yaml --seed 0 --steps 10000 --run-id smoke
roborobust train --config configs/nominal.yaml --seed 0 --steps 500000 --run-id nominal_s0
roborobust train --config configs/randomized.yaml --seed 0 --steps 500000 --run-id randomized_s0
roborobust evaluate --suite configs/test_suite.yaml --runs runs/nominal_s0 runs/randomized_s0 --baselines scripted random --output results/weekend
roborobust report --input results/weekend --output reports/weekend
mjpython -m roborobust replay --episode results/weekend/trajectories/EXACT_EPISODE_ID.npz
```

The replay command above contains an explicit example placeholder; README must show an actual generated filename after execution. For viewer use on macOS, use mjpython. Run headless training through normal Python. `doctor` checks Python/architecture, imports, MuJoCo stepping, and optional renderer health; rendering failure must not block headless training. Do not force Linux EGL settings on macOS.

## 10. Logging and replay

One episode result includes: run_id, policy label, training_seed, scenario_id, layout_id, condition, manifest hash, environment/config hash, checkpoint hash, evaluation seed, terminal reason, success, collision, duration, path length, return, reward components and actual sampled parameters.

Detailed trajectories store simulator integration state at control boundaries, observations, actions, rewards, event timestamps and scenario/model configuration. Save sufficient state to restore viewer frames without pretending a new rollout is exact replay. If action re-simulation is offered, separately check state agreement with tolerance and disclose divergence. Capture substep collision event details when termination occurs mid-control interval.

Only detailed logs and evaluators may access hidden parameters; the actor never does. Save a small representative trajectory set by default and allow all trajectories optionally. Do not overwrite old runs silently.

## 11. Required correctness gates

Gate A: physics

- Straight equal-wheel commands move roughly straight on an empty nominal floor.
- Opposite commands rotate with the expected sign.
- Robot remains stable at rest and under bounded actions.
- A wall prevents passage; floor contacts never count as obstacle collisions.
- A weakened motor changes the measured trajectory under a suitable command; record the result.

Gate B: environment

- Gymnasium/SB3 environment checker passes.
- Observation shape, bounds and dtype are correct; all values finite.
- Known raycast distances and robot exclusion are correct.
- Reset plus identical actions reproduces the trajectory within tolerance on the same machine/version.
- Randomization does not accumulate across resets; seeded worker streams do not duplicate unintentionally.
- Manifests are disjoint and starts/goals valid.
- Success, collision priority, dwell requirement, timeout and shaping tests pass.

Gate C: learning

- Scripted baseline can solve simple empty-room tasks.
- PPO smoke run saves/reloads successfully.
- Reloaded policy gives the same deterministic actions for fixed observations.
- Empty-room PPO shows real validation improvement before scaling difficulty.

Gate D: experiment

- Equal training transition budgets and paired test manifests.
- No test-time learning or observation/reward-normalization updates.
- All expected episode records present, or missing runs visibly marked incomplete.
- Aggregate reports reconcile against raw terminal outcomes.
- All plots derive from completed logs; no hand-entered scores.

## 12. Weekend sequence and scope cuts

Saturday morning (3-4 hours): asset, physical controller checks, sensors and renderer. Gate A.

Saturday afternoon (3-4 hours): environment, verifier, rewards, manifests and tests. Gate B.

Saturday evening: smoke training, empty-room diagnostic, throughput measurement. Gate C. Start main paired training only after these pass.

Sunday morning: inspect development failures, finish the paired training runs. Freeze final protocol before test evaluation.

Sunday afternoon: held-out evaluation, raw-data checks, plots, failure replays. Gate D.

Sunday evening: README, concise report, three-minute demo recording. Add more training seeds if runtime allows; otherwise prominently label results preliminary.

If time is tight, cut three-obstacle maps, videos, extra stress conditions and training ablations first. Preserve two trained conditions, independent verification, held-out evaluation and failure traces. If even these cannot complete, deliver the working environment and explicitly incomplete experiment; do not claim a completed robustness study.

## 13. Final deliverables and completion report

Deliver a repository with a working local setup, real tests, one-command training/evaluation, saved checkpoints if trained, raw results, generated report and replayable trajectories. Provide a brief implementation report listing what ran, what passed/failed, runtime, exact reproduction commands, and remaining limitations.

README should explain the task, observation/action spaces, dynamics, reward formula, terminal semantics, experiment protocol and macOS setup. REPORT.md should state the hypothesis, measured results, baseline comparison, uncertainty, failure examples and limits. Include a reproducibility table with code/config/manifest/checkpoint hashes.

Demo narrative: show the task and sensors; explain the two training conditions; replay a paired stress scenario; show aggregate results; show a failure; state that all evidence is simulation-only.

Do not gate completion on PPO beating the scripted controller or domain randomization winning. Correct negative results are valid research evidence.

## 14. What the human owner must understand

The generated code alone does not prove deep expertise. Before showing this project to a founder, be able to explain and modify:

1. How wheel torque/contact turns actions into motion, and how you checked it.
2. Why training reward and evaluation success are separate.
3. Why truncation and termination affect value bootstrapping differently.
4. How the policy might exploit a poorly designed reward.
5. Why test layouts, seeds and checkpoint selection matter.
6. Why noisy observations and hidden motor parameters introduce partial observability.
7. What a reactive controller reveals about task difficulty and the value of RL.
8. Whether randomization improved results, where it hurt, and what the evidence cannot establish.
9. Why simulation robustness does not prove hardware transfer.
10. What single follow-up experiment would discriminate between competing failure explanations.

Make one meaningful design decision yourself and reproduce at least one failure investigation. Disclose AI assistance if asked and distinguish your experimental decisions from generated implementation.

## Official technical references

- MuJoCo Python bindings, viewer and macOS launcher: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo project/platform support: https://github.com/google-deepmind/mujoco
- Gymnasium time-limit and termination semantics: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- SB3 custom environment integration: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- SB3 PPO guidance: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

The numerical environment/training settings in this specification are proposed engineering starting points, not externally validated benchmark standards. Verify APIs against the resolved dependency versions.
