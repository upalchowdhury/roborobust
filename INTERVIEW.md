# Presenting RoboRobust in an interview

## Run the demo

From the repository root, after activating `.venv`:

```bash
python -m roborobust showcase
open demo/showcase.html
```

The generated HTML is portable: copy it to another computer and open it without a server, internet, Python, or MuJoCo. It embeds the evaluation records and twelve pinned seed-zero recordings. The seed selector filters aggregate results; replay always uses seed zero, as its labels state. Select combined stress, press Play, enable recorded range rays, and scrub the timeline. These are recorded simulator states, not live inference. Rebuild after changing the input results:

```bash
python -m roborobust showcase --input results/weekend --output demo/showcase.html
```

The explorer displays descriptive rates; use REPORT.md for statistical inference. No new test evaluation is needed to build it. The input file hash is embedded for provenance.

## A 60-second explanation

“I built a MuJoCo differential-drive navigation experiment to test whether training-time actuator and sensor randomization improves safe arrival under unseen shifts. Motion comes from wheel contact, and success is checked independently of training reward. I compared nominal and randomized PPO against reactive and random baselines on frozen held-out layouts with paired disturbances and five training seeds.

“The result was negative: combined-stress success was 76.0% versus 76.4%, with uncertainty spanning zero. A reactive controller reached 98%, which tells me this task doesn't justify RL by itself. The more useful finding was a training failure: four original runs learned to idle. The validation investigation led to a discount-factor change, after which none of the ten final runs stalled. This is simulation only, with privileged goal and velocity inputs.”

Use “I” only for work you personally understand and can defend. Name AI assistance and your own contribution accurately when asked.

## Demonstrate engineering judgment

| Interview question | Explain | Inspect |
|---|---|---|
| Why does a weak motor matter if velocity is controlled? | A servo can compensate until its torque saturates; scaling gain and torque limit matters under demanding commands. | `randomization.py`, `validation.py` |
| Could reward be high while performance is poor? | Yes. Discounted shaping and a distant terminal bonus produced an idling attractor; evaluation counts safe arrival, not return. | `rewards.py`, `docs/dev_log.md` |
| Why distinguish termination from truncation? | A time cutoff still has future value; a genuine terminal event does not. | `tests/test_env.py` |
| Is the comparison fair? | Same layouts, matched training seeds, paired noise, equal budgets, final checkpoints, and an explicit test-access record. | `evaluate.py`, `docs/protocol_freeze.md` |
| Why no claim of improvement? | The effect is small relative to seed/layout variability; overlapping-zero intervals do not establish equivalence. | `REPORT.md` |
| Why use RL when a baseline wins? | The experiment measures the limits of the approach; baseline strength prevents confusing complexity with value. | `baselines.py` |
| Does this transfer to a robot? | No evidence yet. Motor/noise distributions are uncalibrated; observations include privileged quantities. | `README.md` |

## Reproduce one investigation yourself

1. Run `python -m roborobust validate-env` and explain the weak-wheel measurements.
2. Run the saved failure check below. Read the collision event and identify the obstacle in the explorer.
3. Read the gamma investigation in `docs/dev_log.md`. Explain why increasing gamma could help and what alternative reward change you would test.
4. Make one documented design decision yourself and validate it on validation scenarios. Keep new research separate from the completed held-out study.

```bash
python -m roborobust replay \
  --episode results/weekend/trajectories/randomized_s0__combined_stress__main_test_0004.npz \
  --resimulate-check
```

## A focused next experiment

Run the provided motor-only and noise-only interventions with matched budgets and seeds to separate their effects. Predeclare the endpoint and selection rule, tune only on validation, then use a new held-out test set for any adaptive follow-up. Do not interpret the existing combined intervention as evidence about either component independently.

## Resume wording

“Developed a reproducible MuJoCo/Gymnasium navigation benchmark with PPO, actuator/sensor domain randomization, independent safety verification, paired held-out evaluation across five training seeds, bootstrap uncertainty, and an offline trajectory explorer; diagnosed reward-driven policy idling and documented negative robustness results.”

Adjust “developed” to reflect your own contribution. Avoid claims of improved robustness, autonomous hardware deployment, or sim-to-real transfer.
