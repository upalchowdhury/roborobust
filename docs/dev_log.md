# Development log (validation data only; the test split was never evaluated here)

Everything below was decided on the **empty-room validation** set (20 cases) or the **main validation** set
(40 cases). `results/test_access_log.jsonl` records every evaluation of the held-out test split; development never touches it.

## 1. Gate C empty-room diagnostic: initial failure and fix

Spec starting point (alpha = 1, 4 workers, 200k transitions, nominal PPO): **2/20** safe arrivals on empty-room validation
(scripted 20/20, random 0/20). The spec says failure triggers debugging, not a lowered bar.

Checklist from the spec, in order:

* observations / wheel signs / actuators: covered by Gate A/B tests (analytic ray distances, rotation sign, saturation measured at ~4%).
* collisions: 0. Failures were timeouts: speed ~0 with the goal still 1.8-4 m away after only 1-3 m of travel.
* scripted feasibility: 20/20, so the task and verifier are solvable.
* reward magnitude: per-step shaping alpha*(gamma*Phi' - Phi) is at most 0.45 m/s * 0.05 s / 8.49 m = **0.0027**, against a step cost of 0.01,
  so progress was worth less than the cost of taking the step. The policy's action std was still 0.89 (initial 1.0) after 200k steps.

Sweep (alpha, then other knobs; every row is a 200k-transition nominal PPO on `empty_train`, scored on the 20 `empty_val` cases):

| tag | alpha | step cost | n_envs | PPO override | seeds: val successes /20 | mean | collisions (sum over seeds) |
|---|---|---|---|---|---|---|---|
| a1 | 1.0 | default | 4 | - | s0=2, s1=0 | 1.0 | 0 |
| a5 | 5.0 | default | 4 | - | s0=4, s1=4 | 4.0 | 2 |
| a10 | 10.0 | default | 4 | - | s0=14, s1=3 | 8.5 | 3 |
| a10_n1 | 10.0 | default | 1 | - | s0=3, s1=12 | 7.5 | 19 |
| a10_n2 | 10.0 | default | 2 | - | s0=11, s1=2 | 6.5 | 0 |
| a10_n8 | 10.0 | default | 8 | - | s0=7, s1=4 | 5.5 | 2 |
| A_a25 | 25.0 | default | 4 | - | s0=9, s1=8, s2=11 | 9.3 | 6 |
| C_a25_lr1e3 | 25.0 | default | 4 | {"learning_rate": 0.001} | s0=19, s1=14, s2=6 | 13.0 | 4 |
| F_a25_sc002 | 25.0 | 0.002 | 4 | - | s0=16, s1=9, s2=5 | 10.0 | 2 |
| a25 | 25.0 | default | 4 | - | s0=9, s1=8 | 8.5 | 4 |
| a25_n1 | 25.0 | default | 1 | - | s0=15, s1=6 | 10.5 | 12 |
| a25_n2 | 25.0 | default | 2 | - | s0=8, s1=7 | 7.5 | 4 |
| a25_n8 | 25.0 | default | 8 | - | s0=8, s1=9 | 8.5 | 4 |
| B_a50 | 50.0 | default | 4 | - | s0=16, s1=16, s2=18 | 16.7 | 10 |
| E_a50_n1 | 50.0 | default | 1 | - | s0=15, s1=16, s2=18 | 16.3 | 7 |
| E_a50_n8 | 50.0 | default | 8 | - | s0=10, s1=12, s2=18 | 13.3 | 4 |
| D_a100 | 100.0 | default | 4 | - | s0=15, s1=18, s2=17 | 16.7 | 10 |

Observations on the non-successful episodes of the better runs: (a) *stopping just short* (final distance 0.31-0.47 m, never entered the
0.30 m disc, speed 0); (b) *circling* near the goal; (c) *freezing far away* (deterministic evaluation of an under-trained policy sitting on a zero-action fixed point).

**Decision:** shaping_alpha = **50** for both conditions (smallest value that met the >= 80% target on all three seeds; alpha = 100 matched but did not beat it: 16.7 vs 16.7 mean successes, same collision count).
Official CLI re-runs at that point (alpha = 50, gamma still 0.99; gamma was changed later, see sections 4-6, and the diagnostic is re-run under the final protocol in section 6):

| run | PPO success on 20 empty-room validation cases | collisions |
|---|---|---|
| alpha=1, gamma=0.99 (spec default), seed 0 | 2/20 | 0 |
| alpha=50, gamma=0.99, seed 0 | 16/20 | 4 |
| alpha=50, gamma=0.99, seed 1 | 16/20 | 4 |
| alpha=50, gamma=0.99, seed 2 | 18/20 | 2 |

Caveats: alpha was tuned on 20 validation cases and 2-3 seeds; the 80% bar is met narrowly on two of three seeds (16/20).
Potential-based shaping is policy-invariant only under its assumptions (see `rewards.py`); a large alpha changes learning dynamics, not the optimal policy.
`n_envs`: the benchmark's throughput winner is 8 workers (9.95k steps/s vs 7.2k at 4), but at alpha = 50 eight workers learned worse per transition
in this sweep (mean 13.3 vs 16.7 successes) and a 1M-step run takes ~3 minutes at 4 workers, so **4 workers** is used (deviation from "pick the
throughput winner", justified by sample efficiency; recorded in DECISIONS.md).

## 2. Reward-design audit: why returns mislead and why "freeze" is a rational failure mode

* With gamma < 1, shaping pays alpha*(1-gamma)*d/diag per step for *standing still* at distance d (+0.0059*d with alpha = 50). Undiscounted episode returns of idling
  timeouts can therefore exceed those of quick successes; **training return is never reported as task performance**, success is verified independently (`verifier.py`).
  Measured on the 1M-step seed-0 policies, validation split, nominal condition (undiscounted episode return by verified outcome):

| terminal_reason   |   count |   mean |   min |   max |
|:------------------|--------:|-------:|------:|------:|
| collision         |      15 |   34.5 |  12.1 |  76.3 |
| success           |      58 |   55.3 |  21.5 | 144.7 |
| timeout           |       7 |   81.5 |  28.8 | 142.5 |

* Discounted, the shaping telescopes to -Phi(s0) however the episode ends (terminal Phi = 0), so it neither creates nor removes an incentive to arrive; the incentive to
  arrive is the discounted +10 bonus versus the step cost. At gamma = 0.99 a bonus 300 steps (15 s) away is worth only ~0.5 now, so freezing far from the goal is a
  near-optimal *local* behaviour under this reward. This is a design limitation we did not tune away (higher gamma or a larger bonus would be the lever; not explored).

## 3. Main-task development runs (validation split)

Single-seed look at two budgets (seed 0), nominal vs randomized, same protocol:

500k transitions:

| policy     |   nominal |   within_support |   weak_left |   weak_right |   noisy_sensing |   combined_stress |
|:-----------|----------:|-----------------:|------------:|-------------:|----------------:|------------------:|
| nominal    |        25 |               27 |          18 |           23 |              26 |                19 |
| random     |         0 |                0 |           0 |            0 |               0 |                 0 |
| randomized |        14 |               16 |          13 |           14 |              16 |                13 |
| scripted   |        39 |               38 |          39 |           39 |              39 |                39 |

(successes out of 40 validation scenarios per cell)

1M transitions:

| policy     |   nominal |   within_support |   weak_left |   weak_right |   noisy_sensing |   combined_stress |
|:-----------|----------:|-----------------:|------------:|-------------:|----------------:|------------------:|
| nominal    |        28 |               28 |          24 |           19 |              27 |                24 |
| randomized |        30 |               30 |          30 |           30 |              30 |                30 |
| scripted   |        39 |               38 |          39 |           39 |              39 |                39 |

(successes out of 40 validation scenarios per cell)

The ordering between the two conditions **flipped between 500k and 1M** for this seed and curves were still rising at 500k; a single seed at a single budget supports neither claim.
Final-study budget was therefore set to 1M transitions (the spec's pre-stated option when runtime allows; ~170 s per run) with several seeds.

## 4. Original protocol across seeds: stalls (validation split, 1M transitions, final checkpoints)

|                     |   nominal |   within_support |   weak_left |   weak_right |   noisy_sensing |   combined_stress |
|:--------------------|----------:|-----------------:|------------:|-------------:|----------------:|------------------:|
| ('nominal', 0.0)    |        28 |               28 |          24 |           19 |              27 |                24 |
| ('nominal', 1.0)    |        32 |               31 |          31 |           35 |              30 |                30 |
| ('nominal', 2.0)    |         0 |                0 |           0 |            0 |               0 |                 0 |
| ('nominal', 3.0)    |        33 |               29 |          31 |           25 |              33 |                29 |
| ('nominal', 4.0)    |         0 |                0 |           0 |            0 |               0 |                 0 |
| ('randomized', 0.0) |        30 |               30 |          30 |           30 |              30 |                30 |
| ('randomized', 1.0) |         7 |                6 |           7 |            9 |               6 |                 7 |
| ('randomized', 2.0) |        24 |               27 |          28 |           25 |              25 |                27 |
| ('randomized', 3.0) |        28 |               26 |          19 |           25 |              27 |                20 |
| ('randomized', 4.0) |         0 |                0 |           0 |            0 |               0 |                 0 |

(successes out of 40 validation scenarios per cell)

A *stall* = nominal-condition validation success below 8/40. Failed runs show ~0 collisions, 85-100% timeouts and ~0% training success for the whole run (a safe idling policy; checked in the training logs of nominal_s2 and randomized_s1).
Stall count under the original protocol (alpha 50, gamma 0.99, ent_coef 0): **4 of 10 runs {'nominal_s2': 0, 'nominal_s4': 0, 'randomized_s1': 7, 'randomized_s4': 0}**. Seed variance, not the stress condition, dominates these numbers.

## 5. Stability sweep on dev seeds 10-13 (both conditions, 1M transitions, validation split)

Selection rule written before any result (DECISIONS.md): fewest stalled runs summed over both conditions, ties by higher mean nominal-condition validation success; keep `base` if nothing beats it.

|                          |   10 |   11 |   12 |   13 |
|:-------------------------|-----:|-----:|-----:|-----:|
| ('base', 'nominal')      |    1 |   29 |   30 |   31 |
| ('base', 'randomized')   |    2 |   27 |   22 |    5 |
| ('ent005', 'nominal')    |   25 |   23 |   33 |   27 |
| ('ent005', 'randomized') |   30 |   21 |   19 |   33 |
| ('g995', 'nominal')      |   34 |   32 |   31 |   36 |
| ('g995', 'randomized')   |   29 |   31 |   33 |   37 |
| ('g999', 'nominal')      |   34 |   36 |   30 |   26 |
| ('g999', 'randomized')   |   36 |   37 |   33 |   34 |

(nominal-condition validation successes out of 40; one column per dev seed)

Per config (both conditions pooled):

| config   |   runs |   stalls |   mean_nominal |   mean_combined |
|:---------|-------:|---------:|---------------:|----------------:|
| base     |      8 |        3 |          18.38 |           16.12 |
| ent005   |      8 |        0 |          26.38 |           22.62 |
| g995     |      8 |        0 |          32.88 |           28.75 |
| g999     |      8 |        0 |          33.25 |           28.62 |

**Rule outcome: `g999`** (stalls: base=3, ent005=0, g995=0, g999=0). `g995` and `g999` differ by less than half a success out of 40 in mean nominal-condition score, i.e. this is a near-tie that the pre-declared tie-break resolved; the robust finding is that raising gamma removes the stalls, not that 0.999 is special.

## 6. Checks under the final protocol (before the freeze; validation data only)

Gate C empty-room diagnostic re-run with the final configuration:

| run (gamma 0.999, alpha 50, 200k) | empty-room validation success | collisions |
|---|---|---|
| seed 0 | 18/20 | 2 |
| seed 1 | 18/20 | 2 |
| seed 2 | 17/20 | 3 |

Validation-split stall check of the ten final runs (no tuning followed from it): stalls **0 of 10**.

|                     |   nominal |   within_support |   weak_left |   weak_right |   noisy_sensing |   combined_stress |
|:--------------------|----------:|-----------------:|------------:|-------------:|----------------:|------------------:|
| ('nominal', 0.0)    |        34 |               34 |          29 |           34 |              35 |                29 |
| ('nominal', 1.0)    |        26 |               28 |          33 |           26 |              26 |                33 |
| ('nominal', 2.0)    |        30 |               31 |          30 |           30 |              29 |                30 |
| ('nominal', 3.0)    |        35 |               34 |          28 |           33 |              35 |                28 |
| ('nominal', 4.0)    |        37 |               36 |          27 |           29 |              37 |                28 |
| ('randomized', 0.0) |        28 |               29 |          27 |           23 |              29 |                28 |
| ('randomized', 1.0) |        27 |               27 |          30 |           22 |              28 |                29 |
| ('randomized', 2.0) |        34 |               36 |          33 |           34 |              35 |                34 |
| ('randomized', 3.0) |        34 |               33 |          27 |           36 |              32 |                27 |
| ('randomized', 4.0) |        37 |               37 |          29 |           28 |              36 |                29 |

(successes out of 40 validation scenarios per cell)

## 7. Protocol freeze

Frozen before any test evaluation (see DECISIONS.md for the full reasoning):

* reward: alpha 50 (empty-room validation), success bonus 10, failure penalty 10, step cost 0.01, **gamma 0.999** (stability sweep, rule pre-registered)
* PPO: spec defaults (MlpPolicy 64x64, lr 3e-4, gae_lambda 0.95, clip 0.2, 10 epochs, batch 256, n_steps 1024, ent_coef 0) except gamma 0.999; 4 workers
* budget: 1M requested transitions = 1,003,520 actual (245 whole rollouts), identical for both conditions
* seeds 0-4 per condition (the earlier gamma = 0.99 runs are archived and appear only in section 4 above)
* primary comparison: final checkpoints; primary endpoint: safe arrival under `combined_stress` on the 50 held-out layouts; `configs/test_suite.yaml`
* pre-declared secondary: the ~500k-transition checkpoints of the same runs, and every single condition
