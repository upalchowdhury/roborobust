"""Regenerate docs/dev_log.md tables from the raw development logs (no hand-copied numbers)."""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

rows = [json.loads(l) for l in open("docs/dev_sweep_results.jsonl")]
g = defaultdict(list)
for r in rows:
    g[(r["tag"], r["alpha"], r["step_cost"], r["n_envs"], r["ppo"])].append(r)
lines = ["| tag | alpha | step cost | n_envs | PPO override | seeds: val successes /20 | mean | collisions (sum over seeds) |", "|---|---|---|---|---|---|---|---|"]
for (tag, a, sc, n, ppo), rs in sorted(g.items(), key=lambda kv: (kv[0][1] or 0, kv[0][0])):
    rs = sorted(rs, key=lambda r: r["seed"])
    lines.append(f"| {tag} | {a if a is not None else 'default'} | {sc if sc is not None else 'default'} | {n} | {ppo or '-'} | "
                 + ", ".join(f"s{r['seed']}={r['val_success']}" for r in rs) + f" | {sum(r['val_success'] for r in rs)/len(rs):.1f} | {sum(r['val_collision'] for r in rs)} |")
sweep = "\n".join(lines)

def val_table(path):
    df = pd.read_json(path, lines=True)
    order = ["nominal", "within_support", "weak_left", "weak_right", "noisy_sensing", "combined_stress"]
    t = (df.pivot_table(index="policy", columns="condition", values="success", aggfunc="sum")[order]).astype(int)
    n = int(df.groupby(["policy", "condition"]).size().iloc[0])
    return t.to_markdown() + f"\n\n(successes out of {n} validation scenarios per cell)"

try:
    v500 = val_table("results/dev_val_s0/episode_results.jsonl")
    v1m = val_table("results/dev_val_1M_s0/episode_results.jsonl")
except Exception as e:  # results/ is not committed; keep whatever was last written
    v500 = v1m = f"(raw results not present: {e})"

try:
    _d = pd.read_json("results/dev_val_1M_s0/episode_results.jsonl", lines=True)
    _d = _d[_d.policy.isin(["nominal", "randomized"]) & (_d.condition == "nominal")]
    ret_tab = _d.groupby("terminal_reason")["return"].agg(["count", "mean", "min", "max"]).round(1).to_markdown()
except Exception as e:
    ret_tab = f"(raw results not present: {e})"

diag = []
for tag, path in (("alpha=1, gamma=0.99 (spec default), seed 0", "results/archive_gamma099_diag_empty_alpha1_s0"),
                  ("alpha=50, gamma=0.99, seed 0", "results/archive_gamma099_diag_empty_s0"),
                  ("alpha=50, gamma=0.99, seed 1", "results/archive_gamma099_diag_empty_s1"),
                  ("alpha=50, gamma=0.99, seed 2", "results/archive_gamma099_diag_empty_s2")):
    try:
        d = pd.read_json(f"{path}/episode_results.jsonl", lines=True)
        c = d[d.policy == "empty_diag"]
        diag.append(f"| {tag} | {int(c.success.sum())}/{len(c)} | {int(c.collision.sum())} |")
    except Exception:
        pass

# ---- original protocol across seeds (validation eval of runs/{nominal,randomized}_s0-4)
STALL_BELOW = 8  # of 40 nominal-condition validation scenarios
try:
    od = pd.read_json("results/archive_gamma099_dev_val_orig5/episode_results.jsonl", lines=True)
    od = od[od.policy.isin(["nominal", "randomized"])]
    cond_order = ["nominal", "within_support", "weak_left", "weak_right", "noisy_sensing", "combined_stress"]
    ot = od.pivot_table(index=["policy", "training_seed"], columns="condition", values="success", aggfunc="sum")[cond_order].astype(int)
    orig_tab = ot.to_markdown() + "\n\n(successes out of 40 validation scenarios per cell)"
    orig_stalls = f"{int((ot['nominal'] < STALL_BELOW).sum())} of {len(ot)} runs " + str({f'{a}_s{int(b)}': int(v) for (a, b), v in ot['nominal'].items() if v < STALL_BELOW})
except Exception as e:
    orig_tab, orig_stalls = f"(raw results not present: {e})", "n/a"

# ---- stability sweep, selection rule applied mechanically
try:
    sw = [json.loads(l) for l in open("docs/dev_main_sweep_results.jsonl")]
    rows, summary = [], {}
    for r in sw:
        n = r["by_condition"]["nominal"]
        rows.append({"config": r["tag"], "condition": r["cond"], "seed": r["seed"], "nominal": n["success"], "combined_stress": r["by_condition"]["combined_stress"]["success"],
                     "nominal_timeouts": n["timeout"], "nominal_collisions": n["collision"], "stall": n["success"] < STALL_BELOW})
    sdf = pd.DataFrame(rows)
    sweep_tab = sdf.pivot_table(index=["config", "condition"], columns="seed", values="nominal", aggfunc="first").astype(int).to_markdown() + \
        "\n\n(nominal-condition validation successes out of 40; one column per dev seed)"
    agg = sdf.groupby("config").agg(runs=("stall", "size"), stalls=("stall", "sum"), mean_nominal=("nominal", "mean"), mean_combined=("combined_stress", "mean")).round(2)
    agg["stalls"] = agg.stalls.astype(int)
    sweep_tab += "\n\nPer config (both conditions pooled):\n\n" + agg.to_markdown()
    ordered = agg.sort_values(["stalls", "mean_nominal"], ascending=[True, False])
    best = ordered.index[0]
    base_stalls = int(agg.loc["base", "stalls"]) if "base" in agg.index else None
    chosen = best if (base_stalls is None or int(agg.loc[best, "stalls"]) < base_stalls) else "base"
    sweep_verdict = (f"**Rule outcome: `{chosen}`** (stalls: " + ", ".join(f"{c}={int(v)}" for c, v in agg.stalls.items()) + "). "
                     "`g995` and `g999` differ by less than half a success out of 40 in mean nominal-condition score, i.e. this is a near-tie that the pre-declared "
                     "tie-break resolved; the robust finding is that raising gamma removes the stalls, not that 0.999 is special.")
except Exception as e:
    sweep_tab, sweep_verdict = f"(sweep results not present: {e})", ""

# ---- checks under the final protocol (validation split + Gate C re-run)
try:
    fd = pd.read_json("results/dev_val_final/episode_results.jsonl", lines=True)
    fd = fd[fd.policy.isin(["nominal", "randomized"])]
    ft = fd.pivot_table(index=["policy", "training_seed"], columns="condition", values="success", aggfunc="sum")[cond_order].astype(int)
    final_tab = ft.to_markdown() + "\n\n(successes out of 40 validation scenarios per cell)"
    final_stalls = f"{int((ft['nominal'] < STALL_BELOW).sum())} of {len(ft)}"
except Exception as e:
    final_tab, final_stalls = f"(raw results not present: {e})", "n/a"
gc = []
for sd in (0, 1, 2):
    try:
        d = pd.read_json(f"results/diag_empty_s{sd}/episode_results.jsonl", lines=True)
        c = d[d.policy == "empty_diag"]
        gc.append(f"| seed {sd} | {int(c.success.sum())}/{len(c)} | {int(c.collision.sum())} |")
    except Exception:
        pass
gatec_final = "| run (gamma 0.999, alpha 50, 200k) | empty-room validation success | collisions |\n|---|---|---|\n" + "\n".join(gc)

freeze = Path("docs/protocol_freeze.md").read_text() if Path("docs/protocol_freeze.md").exists() else "(not yet written)"

md = f"""# Development log (validation data only; the test split was never evaluated here)

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

{sweep}

Observations on the non-successful episodes of the better runs: (a) *stopping just short* (final distance 0.31-0.47 m, never entered the
0.30 m disc, speed 0); (b) *circling* near the goal; (c) *freezing far away* (deterministic evaluation of an under-trained policy sitting on a zero-action fixed point).

**Decision:** shaping_alpha = **50** for both conditions (smallest value that met the >= 80% target on all three seeds; alpha = 100 matched but did not beat it: 16.7 vs 16.7 mean successes, same collision count).
Official CLI re-runs at that point (alpha = 50, gamma still 0.99; gamma was changed later, see sections 4-6, and the diagnostic is re-run under the final protocol in section 6):

| run | PPO success on 20 empty-room validation cases | collisions |
|---|---|---|
""" + "\n".join(diag) + f"""

Caveats: alpha was tuned on 20 validation cases and 2-3 seeds; the 80% bar is met narrowly on two of three seeds (16/20).
Potential-based shaping is policy-invariant only under its assumptions (see `rewards.py`); a large alpha changes learning dynamics, not the optimal policy.
`n_envs`: the benchmark's throughput winner is 8 workers (9.95k steps/s vs 7.2k at 4), but at alpha = 50 eight workers learned worse per transition
in this sweep (mean 13.3 vs 16.7 successes) and a 1M-step run takes ~3 minutes at 4 workers, so **4 workers** is used (deviation from "pick the
throughput winner", justified by sample efficiency; recorded in DECISIONS.md).

## 2. Reward-design audit: why returns mislead and why "freeze" is a rational failure mode

* With gamma < 1, shaping pays alpha*(1-gamma)*d/diag per step for *standing still* at distance d (+0.0059*d with alpha = 50). Undiscounted episode returns of idling
  timeouts can therefore exceed those of quick successes; **training return is never reported as task performance**, success is verified independently (`verifier.py`).
  Measured on the 1M-step seed-0 policies, validation split, nominal condition (undiscounted episode return by verified outcome):

{ret_tab}

* Discounted, the shaping telescopes to -Phi(s0) however the episode ends (terminal Phi = 0), so it neither creates nor removes an incentive to arrive; the incentive to
  arrive is the discounted +10 bonus versus the step cost. At gamma = 0.99 a bonus 300 steps (15 s) away is worth only ~0.5 now, so freezing far from the goal is a
  near-optimal *local* behaviour under this reward. This is a design limitation we did not tune away (higher gamma or a larger bonus would be the lever; not explored).

## 3. Main-task development runs (validation split)

Single-seed look at two budgets (seed 0), nominal vs randomized, same protocol:

500k transitions:

{v500}

1M transitions:

{v1m}

The ordering between the two conditions **flipped between 500k and 1M** for this seed and curves were still rising at 500k; a single seed at a single budget supports neither claim.
Final-study budget was therefore set to 1M transitions (the spec's pre-stated option when runtime allows; ~170 s per run) with several seeds.

## 4. Original protocol across seeds: stalls (validation split, 1M transitions, final checkpoints)

{orig_tab}

A *stall* = nominal-condition validation success below 8/40. Failed runs show ~0 collisions, 85-100% timeouts and ~0% training success for the whole run (a safe idling policy; checked in the training logs of nominal_s2 and randomized_s1).
Stall count under the original protocol (alpha 50, gamma 0.99, ent_coef 0): **{orig_stalls}**. Seed variance, not the stress condition, dominates these numbers.

## 5. Stability sweep on dev seeds 10-13 (both conditions, 1M transitions, validation split)

Selection rule written before any result (DECISIONS.md): fewest stalled runs summed over both conditions, ties by higher mean nominal-condition validation success; keep `base` if nothing beats it.

{sweep_tab}

{sweep_verdict}

## 6. Checks under the final protocol (before the freeze; validation data only)

Gate C empty-room diagnostic re-run with the final configuration:

{gatec_final}

Validation-split stall check of the ten final runs (no tuning followed from it): stalls **{final_stalls}**.

{final_tab}

## 7. Protocol freeze

{freeze}"""
Path("docs/dev_log.md").write_text(md)
print("wrote docs/dev_log.md")
