"""Aggregation, uncertainty, failure taxonomy, plots and REPORT.md -- all derived from raw logs.

Nothing here accepts a hand-entered score. Every table is computed from ``episode_results.jsonl``
(+ run ``meta.json`` / ``progress.csv`` for training cost and curves); integrity checks run first and a
mismatch aborts the report rather than producing numbers that do not reconcile with raw outcomes.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION_ORDER = ["nominal", "within_support", "weak_left", "weak_right", "noisy_sensing", "combined_stress"]
POLICY_ORDER = ["nominal", "randomized", "scripted", "random"]
POLICY_LABEL = {"nominal": "PPO-nominal", "randomized": "PPO-randomized", "scripted": "Scripted-reactive", "random": "Random"}
# Validated categorical slots 1-4 (scripts/validate_palette.js: all hard gates pass, light surface #fcfcfb).
COLORS = {"nominal": "#2a78d6", "randomized": "#eb6834", "scripted": "#1baf7a", "random": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# Failure-taxonomy thresholds: documented, fixed, applied identically to every policy.
TAXONOMY = {
    "turning_yaw_rate": 0.8,        # rad/s at the terminal step -> collision_while_turning
    "passage_clearance": 0.9,       # m, both lateral clearances below this at collision -> collision_near_passage
    "stop_speed": 0.02,             # m/s: counts as "stopped"
    "short_band": 0.75,             # m: stopped within (goal_radius, short_band] -> stopping_short
    "circling_yaw_travel": 4 * math.pi,  # rad of accumulated |heading change| -> circling_oscillation
    "drift_yaw_rate": 0.10,         # rad/s mean yaw rate during near-straight commands
    "drift_min_steps": 20,
    "asymmetry": 0.2,               # |left - right| motor multiplier for the drift tag
    "near_goal": 1.0,               # m: collision with the goal this close -> collision_near_goal, else collision_en_route
}


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def success_matrix(df: pd.DataFrame, policy: str, condition: str) -> np.ndarray:
    """(n_seeds, n_scenarios) 0/1 matrix, scenarios in manifest order (the pairing key)."""
    d = df[(df.policy == policy) & (df.condition == condition)]
    rows = []
    for _, g in d.groupby("seed", sort=True):
        rows.append(g.sort_values("scenario_index").success.to_numpy(dtype=float))
    return np.array(rows)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0, hierarchical: bool = False) -> dict:
    """Bootstrap of mean(a) - mean(b) over SCENARIO ids; the same resampled ids are used for both
    policies (and, by calling this per condition with the same seed, for every condition), so
    pairing is retained. With hierarchical=True, training seeds are additionally resampled
    (independently per policy: seed i of one policy is not matched to seed i of the other)."""
    n = a.shape[1]
    assert b.shape[1] == n, "policies must be evaluated on the same scenario set"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if hierarchical:
        sa = rng.integers(0, a.shape[0], size=(n_boot, a.shape[0]))
        sb = rng.integers(0, b.shape[0], size=(n_boot, b.shape[0]))
        ma = a[sa[:, :, None], idx[:, None, :]].mean(axis=(1, 2))
        mb = b[sb[:, :, None], idx[:, None, :]].mean(axis=(1, 2))
    else:
        ma = a.mean(axis=0)[idx].mean(axis=1)
        mb = b.mean(axis=0)[idx].mean(axis=1)
    d = ma - mb
    return {"observed": float(a.mean() - b.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "frac_boot_gt0": float((d > 0).mean()), "hierarchical": hierarchical, "n_boot": n_boot}


def did_bootstrap(a_c, a_0, b_c, b_0, n_boot: int = 10000, seed: int = 0, hierarchical: bool = False) -> dict:
    """Exploratory difference-in-differences of robustness gaps, a = randomized, b = nominal:
        DiD = (a_cond - a_nominal) - (b_cond - b_nominal)
    i.e. how much LESS (+) or MORE (-) the randomized policy loses than the nominal policy when moving from the
    nominal condition to a shifted one. All four (seeds x scenarios) matrices share the resampled scenario ids;
    seeds are resampled once per policy and applied to both of that policy's conditions."""
    n = a_c.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if hierarchical:
        sa = rng.integers(0, a_c.shape[0], size=(n_boot, a_c.shape[0]))
        sb = rng.integers(0, b_c.shape[0], size=(n_boot, b_c.shape[0]))
        m = lambda M, S: M[S[:, :, None], idx[:, None, :]].mean(axis=(1, 2))  # noqa: E731
        d = (m(a_c, sa) - m(a_0, sa)) - (m(b_c, sb) - m(b_0, sb))
    else:
        m = lambda M: M.mean(axis=0)[idx].mean(axis=1)  # noqa: E731
        d = (m(a_c) - m(a_0)) - (m(b_c) - m(b_0))
    obs = (a_c.mean() - a_0.mean()) - (b_c.mean() - b_0.mean())
    return {"observed": float(obs), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


# --------------------------------------------------------------------------------------
# Loading + integrity
# --------------------------------------------------------------------------------------
def load_results(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((input_dir / "eval_meta.json").read_text())
    rows = [json.loads(line) for line in open(input_dir / "episode_results.jsonl")]
    df = pd.DataFrame(rows)
    df["seed"] = df.training_seed.fillna(-1).astype(int)
    feats = pd.json_normalize(df["features"]).add_prefix("f_")
    comps = pd.json_normalize(df["reward_components"]).add_prefix("rc_")
    params = pd.json_normalize(df["sampled_params"]).add_prefix("p_")
    df = pd.concat([df.reset_index(drop=True), feats, comps, params], axis=1)
    return df, meta


def integrity_checks(df: pd.DataFrame, meta: dict) -> dict:
    """Gate D: all expected records present; outcomes reconcile; no duplicates; paired design intact."""
    problems = []
    suite = meta["suite"]
    n_scen = len(set(df.scenario_id))
    cells = df.groupby(["policy", "seed", "condition"]).size()
    expected_cells = len(meta["policies"]) * len(suite["conditions"])
    if len(cells) != expected_cells:
        problems.append(f"{len(cells)} policy x seed x condition cells present, expected {expected_cells}")
    bad = cells[cells != n_scen]
    if len(bad):
        problems.append(f"cells with != {n_scen} episodes: {bad.to_dict()}")
    if df.episode_id.duplicated().any():
        problems.append("duplicate episode_id")
    if len(df) != meta["expected_episodes"]:
        problems.append(f"{len(df)} episodes recorded, {meta['expected_episodes']} expected")
    reasons = {"success", "collision", "timeout", "tipped", "out_of_bounds"}
    if not set(df.terminal_reason) <= reasons:
        problems.append(f"unknown terminal reasons {set(df.terminal_reason) - reasons}")
    if (df.success != (df.terminal_reason == "success")).any():
        problems.append("success flag disagrees with terminal_reason")
    if ((df.terminal_reason == "timeout") != df.truncated).any():
        problems.append("truncated flag disagrees with timeout")
    # paired design: every policy saw the same noise seeds per (scenario, condition)
    ns = df.groupby(["scenario_id", "condition"]).noise_seed.nunique()
    if (ns != 1).any():
        problems.append("noise seeds not shared across policies (pairing broken)")
    if df.manifest_sha256.nunique() != 1 or df.env_cfg_hash.nunique() != 1:
        problems.append("mixed manifest or environment hashes")
    return {"ok": not problems and bool(meta.get("complete", False)), "problems": problems,
            "complete_flag": bool(meta.get("complete", False)), "cells": int(len(cells)), "episodes": int(len(df))}


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def aggregate_cells(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    ident = {(p["policy"], -1 if p["training_seed"] is None else p["training_seed"]): p for p in meta["policies"]}
    out = []
    for (pol, seed, cond), g in df.groupby(["policy", "seed", "condition"], sort=False):
        n, k = len(g), int(g.success.sum())
        lo, hi = wilson(k, n)
        coll = int((g.terminal_reason == "collision").sum())
        tout = int((g.terminal_reason == "timeout").sum())
        other = n - k - coll - tout
        assert k + coll + tout + other == n
        s, f = g[g.success], g[~g.success]
        p = ident.get((pol, seed), {})
        out.append({
            "policy": pol, "training_seed": seed if seed >= 0 else None, "condition": cond, "n_episodes": n,
            "successes": k, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
            "collisions": coll, "collision_rate": coll / n, "timeouts": tout, "timeout_rate": tout / n,
            "other_failures": other, "other_failure_rate": other / n,
            "duration_mean_s_all": g.duration_s.mean(), "duration_median_s_all": g.duration_s.median(),
            "time_to_goal_mean_s_successes": s.time_to_goal_s.mean() if len(s) else float("nan"), "n_time_to_goal": len(s),
            "path_length_mean_m_successes": s.path_length_m.mean() if len(s) else float("nan"),
            "path_length_mean_m_failures": f.path_length_m.mean() if len(f) else float("nan"), "n_failures": len(f),
            "TRAINING_REWARD_return_mean": g["return"].mean(),
            "TRAINING_REWARD_step_cost": g.rc_step_cost.mean(), "TRAINING_REWARD_shaping": g.rc_shaping.mean(),
            "TRAINING_REWARD_success_bonus": g.rc_success_bonus.mean(), "TRAINING_REWARD_failure_penalty": g.rc_failure_penalty.mean(),
            "train_transitions": p.get("train_transitions"), "train_wallclock_s": p.get("train_elapsed_s"),
            "run_id": p.get("run_id"),
        })
    t = pd.DataFrame(out)
    t["_po"] = t.policy.map({p: i for i, p in enumerate(POLICY_ORDER)})
    t["_co"] = t.condition.map({c: i for i, c in enumerate(CONDITION_ORDER)})
    return t.sort_values(["_po", "training_seed", "_co"]).drop(columns=["_po", "_co"]).reset_index(drop=True)


def across_seeds(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pol, cond), g in cells.groupby(["policy", "condition"], sort=False):
        r = g.success_rate.to_numpy()
        rows.append({"policy": pol, "condition": cond, "n_seeds": len(g), "mean_success_rate": r.mean(),
                     "std_success_rate": r.std(ddof=1) if len(r) > 1 else float("nan"), "min_seed": r.min(), "max_seed": r.max(),
                     "pooled_successes": int(g.successes.sum()), "pooled_episodes": int(g.n_episodes.sum()),
                     "mean_collision_rate": g.collision_rate.mean(), "mean_timeout_rate": g.timeout_rate.mean()})
    t = pd.DataFrame(rows)
    t["_po"] = t.policy.map({p: i for i, p in enumerate(POLICY_ORDER)})
    t["_co"] = t.condition.map({c: i for i, c in enumerate(CONDITION_ORDER)})
    return t.sort_values(["_po", "_co"]).drop(columns=["_po", "_co"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------------------
def tag_failures(df: pd.DataFrame, manual: dict[str, str] | None = None) -> pd.DataFrame:
    T = TAXONOMY
    tags_col = []
    for r in df.itertuples():
        tags = []
        if not r.success:
            gr = 0.30
            if r.terminal_reason == "collision":
                ev = r.collision_event if isinstance(r.collision_event, dict) else {}
                tags.append("collision_with_wall" if str(ev.get("hazard_geom", "")).startswith("wall") else "collision_with_obstacle")
                tags.append("collision_near_goal" if r.f_final_goal_distance < T["near_goal"] else "collision_en_route")
                if abs(r.f_final_yaw_rate) > T["turning_yaw_rate"]:
                    tags.append("collision_while_turning")
                if max(r.f_left_clearance_end_m, r.f_right_clearance_end_m) < T["passage_clearance"]:
                    tags.append("collision_near_passage")
            if r.terminal_reason == "timeout":
                stopped = r.f_final_speed < T["stop_speed"]
                if stopped and gr < r.f_final_goal_distance <= T["short_band"]:
                    tags.append("stopping_short")
                if r.f_yaw_travel_rad > T["circling_yaw_travel"]:
                    tags.append("circling_oscillation")
                if not r.f_straight_line_blocked and r.f_final_goal_distance > T["short_band"]:
                    tags.append("timeout_despite_clear_route")
                if r.f_straight_line_blocked and r.f_final_goal_distance > T["short_band"] and stopped:
                    tags.append("stalled_before_obstructed_route")  # extra tag, see REPORT
            weak, strong = r.p_left_motor_mult, r.p_right_motor_mult
            if abs(weak - strong) >= T["asymmetry"] and r.f_straight_cmd_steps >= T["drift_min_steps"] \
                    and r.f_straight_cmd_mean_yaw_rate is not None and not np.isnan(r.f_straight_cmd_mean_yaw_rate):
                expect = 1.0 if weak < strong else -1.0  # weaker left motor veers left (CCW, +yaw rate)
                if expect * r.f_straight_cmd_mean_yaw_rate >= T["drift_yaw_rate"]:
                    tags.append("motor_asymmetry_drift")
            if not tags:
                tags.append("unclassified")
        if manual and r.episode_id in manual:
            tags.append(f"manual:{manual[r.episode_id]}")
        tags_col.append(tags)
    df = df.copy()
    df["tags"] = tags_col
    return df


def taxonomy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pol, cond), g in df.groupby(["policy", "condition"], sort=False):
        f = g[~g.success]
        row = {"policy": pol, "condition": cond, "episodes": len(g), "failures": len(f)}
        for tag in sorted({t for ts in f.tags for t in ts}):
            row[tag] = sum(tag in ts for ts in f.tags)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_comparison(across: pd.DataFrame, cells: pd.DataFrame, out: Path, n_scen: int, primary: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    pols = [p for p in POLICY_ORDER if p in set(across.policy)]
    conds = [c for c in CONDITION_ORDER if c in set(across.condition)]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    w, gap = 0.19, 0.012
    for gi, c in enumerate(conds):
        if c == primary:
            ax.axvspan(gi - 0.5, gi + 0.5, color="#f1efe8", zorder=0)
        for pi, p in enumerate(pols):
            row = across[(across.policy == p) & (across.condition == c)]
            if row.empty:
                continue
            x = gi + (pi - (len(pols) - 1) / 2) * (w + gap)
            m = float(row.mean_success_rate.iloc[0])
            ax.bar(x, m, width=w, color=COLORS[p], zorder=2, label=POLICY_LABEL[p] if gi == 0 else None)
            seeds = cells[(cells.policy == p) & (cells.condition == c)].success_rate.to_numpy()
            if len(seeds) > 1:  # individual training seeds: variability is shown, not averaged away
                ax.scatter(np.full(len(seeds), x), seeds, s=16, color=SURFACE, edgecolor=INK, linewidth=1.0, zorder=4)
            ax.text(x, min(m, 1.0) + (0.035 if len(seeds) > 1 and seeds.max() > m else 0.012) + (seeds.max() - m if len(seeds) > 1 and seeds.max() > m else 0),
                    f"{100 * m:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", "\n") + ("\n(primary)" if c == primary else "") for c in conds])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel(f"safe-arrival rate (n = {n_scen} held-out layouts per cell)")
    ax.set_title("Safe arrival under held-out actuator / sensor shifts (bars: mean over training seeds; dots: each seed)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(pols), frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_learning_curves(run_dirs: dict[str, Path], out: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    panels = [("custom/rate_success", "training-episode success rate", True), ("custom/ep_return", "mean episode return (training reward)", True),
              ("train/entropy_loss", "entropy loss (= -policy entropy)", False), ("train/approx_kl", "approx KL", False),
              ("train/value_loss", "value loss", False), ("train/explained_variance", "explained variance", False)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6), dpi=150)
    any_data = False
    for ax, (col, title, smooth) in zip(axes.ravel(), panels):
        for rid, d in run_dirs.items():
            f = d / "progress.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            if col not in p:
                continue
            pol = "randomized" if "randomized" in rid else "nominal"
            y = p[col]
            if smooth:
                y = y.rolling(8, min_periods=1).mean()
            ax.plot(p["time/total_timesteps"] / 1e6, y, color=COLORS[pol], lw=1.6, alpha=0.9, label=POLICY_LABEL[pol] if rid.endswith("_s0") else None)
            any_data = True
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("transitions (millions)")
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("Training diagnostics, every seed shown (success/return are over each run's own training distribution: "
                 "randomized runs face harder episodes)", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    return any_data


def plot_examples(df: pd.DataFrame, input_dir: Path, out_fail: Path, out_pair: Path, primary: str) -> dict:
    """Selection rule (fixed before looking): primary condition, training seed 0 of each learned policy:
    the first success and the first two failures in scenario order. Paired view: pinned scenarios 0-2,
    nominal_s0 vs randomized_s0 overlaid on the same scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from roborobust.replay import load_episode, plot_topdown

    _style(plt)
    chosen: dict[str, list[str]] = {}
    learned = [p for p in ("nominal", "randomized") if p in set(df.policy)]
    if not learned:
        return {}
    fig, axes = plt.subplots(len(learned), 3, figsize=(10, 3.4 * len(learned)), dpi=140, squeeze=False)
    for r, pol in enumerate(learned):
        d = df[(df.policy == pol) & (df.seed == df[df.policy == pol].seed.min()) & (df.condition == primary)].sort_values("scenario_index")
        picks = list(d[d.success].head(1).itertuples()) + list(d[~d.success].head(2).itertuples())
        chosen[pol] = [x.episode_id for x in picks]
        for c in range(3):
            ax = axes[r, c]
            if c >= len(picks) or not has_traj(picks[c].trajectory_file):
                ax.axis("off")
                continue
            ep = picks[c]
            arrays, meta = load_episode(input_dir / ep.trajectory_file)
            plot_topdown(ax, arrays, meta, color=COLORS[pol])
            tags = "" if ep.success else ", ".join(ep.tags)
            ax.set_title(f"{POLICY_LABEL[pol]} · {ep.scenario_id}\n{ep.terminal_reason}" + (f": {tags}" if tags else ""), fontsize=8, loc="left")
    fig.suptitle(f"Selected episodes, condition = {primary}: first success + first two failures in scenario order (rule fixed in advance)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_fail)
    plt.close(fig)

    if len(learned) == 2:
        pins = sorted(df.scenario_index.unique())[:3]
        fig, axes = plt.subplots(1, len(pins), figsize=(3.6 * len(pins), 4.6), dpi=140, squeeze=False)
        for c, si in enumerate(pins):
            ax = axes[0, c]
            drew = False
            for pol in learned:
                row = df[(df.policy == pol) & (df.seed == df[df.policy == pol].seed.min()) & (df.condition == primary) & (df.scenario_index == si)]
                if row.empty or not has_traj(row.iloc[0].trajectory_file):
                    continue
                arrays, meta = load_episode(input_dir / row.iloc[0].trajectory_file)
                plot_topdown(ax, arrays, meta, color=COLORS[pol], draw_room=not drew, label=f"{POLICY_LABEL[pol]}: {row.iloc[0].terminal_reason}")
                drew = True
            if drew:
                ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
            ax.set_title(f"scenario index {si}", fontsize=8.5, loc="left")
        fig.suptitle(f"Paired replay: same scenario, same domain shift and noise stream ({primary})", fontsize=9.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(out_pair)
        plt.close(fig)
    return chosen


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def has_traj(x) -> bool:
    """trajectory_file is None/NaN for episodes whose trajectory was not saved (NaN is truthy: test the type)."""
    return isinstance(x, str) and len(x) > 0


def _pct(k, n):
    return f"{100 * k / n:.1f}% ({k}/{n})"


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) and not math.isnan(v) else ("" if (isinstance(v, float) and math.isnan(v)) or v is None else str(v)) for v in row) + " |")
    return "\n".join(out)


def report(input_dir: str | Path, output_dir: str | Path, runs_dir: str | Path = "runs", manual_labels: str | None = None,
           secondary: str | Path | None = None) -> Path:
    input_dir, output_dir, runs_dir = Path(input_dir), Path(output_dir), Path(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, meta = load_results(input_dir)
    integ = integrity_checks(df, meta)
    if integ["problems"]:
        raise RuntimeError("integrity check failed -- refusing to produce a report:\n  " + "\n  ".join(integ["problems"]))
    manual = None
    if manual_labels:
        m = pd.read_csv(manual_labels)
        manual = dict(zip(m.episode_id, m.label))
    df = tag_failures(df, manual)
    suite = meta["suite"]
    primary = suite["primary_condition"]
    n_scen = df.scenario_id.nunique()
    cells = aggregate_cells(df, meta)
    across = across_seeds(cells)

    # ---- results.csv (one row per policy x seed x condition) + seed-aggregated rows
    cells.to_csv(output_dir / "results.csv", index=False)
    across.to_csv(output_dir / "results_across_seeds.csv", index=False)
    shutil.copy(input_dir / "episode_results.jsonl", output_dir / "episode_results.jsonl")
    tax = taxonomy_table(df)
    tax.to_csv(output_dir / "failure_taxonomy.csv", index=False)
    df.assign(tags=df.tags.map(lambda t: ";".join(t)))[["episode_id", "policy", "seed", "condition", "scenario_id", "terminal_reason", "tags"]] \
        .to_csv(output_dir / "episode_tags.csv", index=False)

    # ---- reconcile aggregated numbers with raw outcomes (Gate D)
    recon = df.groupby(["policy", "seed", "condition"]).agg(k=("success", "sum"), n=("success", "size"))
    chk = cells.set_index([cells.policy, cells.training_seed.fillna(-1).astype(int), cells.condition])
    for (pol, seed, cond), row in recon.iterrows():
        assert chk.loc[(pol, seed, cond), "successes"] == row.k and chk.loc[(pol, seed, cond), "n_episodes"] == row.n
    assert (cells.successes + cells.collisions + cells.timeouts + cells.other_failures == cells.n_episodes).all()

    # ---- bootstrap (randomized - nominal), same seed -> same resampled scenario ids for every condition
    boots = []
    have_pair = {"nominal", "randomized"} <= set(df.policy)
    if have_pair:
        for c in [c for c in CONDITION_ORDER if c in suite["conditions"]]:
            a, b = success_matrix(df, "randomized", c), success_matrix(df, "nominal", c)
            r1 = paired_bootstrap(a, b, seed=12345, hierarchical=False)
            r2 = paired_bootstrap(a, b, seed=12345, hierarchical=True) if min(a.shape[0], b.shape[0]) >= 2 else None
            boots.append({"condition": c, "diff_obs_pp": 100 * r1["observed"], "scenario_boot_lo_pp": 100 * r1["lo"], "scenario_boot_hi_pp": 100 * r1["hi"],
                          "hier_lo_pp": 100 * r2["lo"] if r2 else float("nan"), "hier_hi_pp": 100 * r2["hi"] if r2 else float("nan")})
    boot_df = pd.DataFrame(boots)
    if len(boot_df):
        boot_df.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    dids = []
    if have_pair and "nominal" in suite["conditions"]:
        a0, b0 = success_matrix(df, "randomized", "nominal"), success_matrix(df, "nominal", "nominal")
        for c in [c for c in CONDITION_ORDER if c in suite["conditions"] and c != "nominal"]:
            ac, bc = success_matrix(df, "randomized", c), success_matrix(df, "nominal", c)
            r1 = did_bootstrap(ac, a0, bc, b0, seed=12345)
            r2 = did_bootstrap(ac, a0, bc, b0, seed=12345, hierarchical=True) if min(ac.shape[0], bc.shape[0]) >= 2 else None
            dids.append({"condition": c, "randomized_drop_pp": 100 * (a0.mean() - ac.mean()), "nominal_drop_pp": 100 * (b0.mean() - bc.mean()),
                         "did_pp": 100 * r1["observed"], "scenario_lo_pp": 100 * r1["lo"], "scenario_hi_pp": 100 * r1["hi"],
                         "hier_lo_pp": 100 * r2["lo"] if r2 else float("nan"), "hier_hi_pp": 100 * r2["hi"] if r2 else float("nan")})
    did_df = pd.DataFrame(dids)
    if len(did_df):
        did_df.to_csv(output_dir / "robustness_gap_did.csv", index=False)

    # ---- plots
    plot_comparison(across, cells, output_dir / "comparison.png", n_scen, primary)
    run_dirs = {}
    for p in meta["policies"]:
        if p["training_seed"] is not None and (runs_dir / p["run_id"]).exists():
            run_dirs[p["run_id"]] = runs_dir / p["run_id"]
    curves = plot_learning_curves(run_dirs, output_dir / "learning_curves.png") if run_dirs else False
    chosen = plot_examples(df, input_dir, output_dir / "failure_examples.png", output_dir / "paired_examples.png", primary)

    # fill identity fields that older evaluations did not record, straight from the run's own meta.json
    for pol in meta["policies"]:
        mj = runs_dir / str(pol.get("run_id")) / "meta.json"
        if pol.get("train_config_hash") is None and mj.exists():
            pol["train_config_hash"] = json.loads(mj.read_text()).get("config_hash")

    # ---- REPORT.md
    md = build_markdown(df, meta, cells, across, boot_df, tax, integ, chosen, primary, n_scen, curves, run_dirs, did_df,
                        secondary_md=secondary_section(Path(secondary)) if secondary else None)
    (output_dir / "REPORT.md").write_text(md)
    print(f"[report] wrote {output_dir}/REPORT.md, results.csv, comparison.png"
          f"{', learning_curves.png' if curves else ''} ({integ['episodes']} episodes, integrity OK)")
    return output_dir


def secondary_section(sec_dir: Path) -> str:
    """Pre-declared secondary analysis: the same runs evaluated at an earlier periodic checkpoint (read from that report's CSVs)."""
    a, b = sec_dir / "results_across_seeds.csv", sec_dir / "paired_bootstrap.csv"
    if not (a.exists() and b.exists()):
        return f"## Pre-declared secondary: budget sensitivity\n\n(secondary report `{sec_dir}` not found)\n"
    across = pd.read_csv(a)
    across = across[across.policy.isin(["nominal", "randomized"])]
    boot = pd.read_csv(b)
    wide = across.assign(v=across.mean_success_rate.map(lambda x: f"{100 * x:.1f}%")).pivot(index="policy", columns="condition", values="v")
    wide = wide.loc[[p for p in POLICY_ORDER if p in wide.index], [c for c in CONDITION_ORDER if c in wide.columns]]
    wide.index = [POLICY_LABEL[p] for p in wide.index]
    bt = boot.copy()
    bt["diff (pp)"] = bt.diff_obs_pp.round(1)
    bt["scenario 95% (pp)"] = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(bt.scenario_boot_lo_pp, bt.scenario_boot_hi_pp)]
    bt["hierarchical 95% (pp)"] = [("n/a" if pd.isna(lo) else f"[{lo:.1f}, {hi:.1f}]") for lo, hi in zip(bt.hier_lo_pp, bt.hier_hi_pp)]
    return ("## Pre-declared secondary: budget sensitivity (checkpoints at ~500k transitions)\n\n"
            "The same ten runs, evaluated with the identical frozen suite at their periodic checkpoint `step_000503808` (123 rollouts = 503,808 transitions), "
            "declared as a secondary analysis in `DECISIONS.md` before any test evaluation. Full tables: `" + str(sec_dir) + "/REPORT.md`.\n\n"
            + _md_table(wide.reset_index().rename(columns={"index": "policy"})) + "\n\nRandomized minus nominal at ~500k (same bootstrap as above):\n\n"
            + _md_table(bt[["condition", "diff (pp)", "scenario 95% (pp)", "hierarchical 95% (pp)"]], "{:.1f}") + "\n\n"
            "Reading this alongside the final-checkpoint result shows how sensitive the comparison is to the training budget; it is not a second independent test "
            "(same runs, same layouts).\n")


def build_markdown(df, meta, cells, across, boot_df, tax, integ, chosen, primary, n_scen, curves, run_dirs, did_df=None, secondary_md=None) -> str:
    suite = meta["suite"]
    learned = [p for p in ("nominal", "randomized") if p in set(df.policy)]
    seeds = {p: sorted(df[df.policy == p].seed.unique().tolist()) for p in learned}
    n_seeds = min((len(v) for v in seeds.values()), default=0)
    prelim = n_seeds < 3 or not integ["ok"]
    status = ("**INCOMPLETE** -- some expected episodes are missing; see eval_meta.json. " if not integ["complete_flag"] else "") + \
             ("**PRELIMINARY**: fewer than three training seeds per condition. " if n_seeds < 3 else "") + \
             (f"{n_seeds} training seeds per condition (intervals over seeds are exploratory)." if n_seeds >= 3 else "")
    cells = cells.assign(seed=cells.training_seed.map(lambda v: "-" if pd.isna(v) else str(int(v))))
    L: list[str] = []
    L.append(f"# RoboRobust results -- suite `{suite['suite_name']}`\n")
    L.append(f"> {status}\n> **Simulation only.** No physical robot, no sim-to-real claim. Goal position and velocities are privileged simulator "
             "inputs; the policy is a feedforward MLP. All numbers below are computed from `episode_results.jsonl`.\n")
    L.append("## Hypothesis and design\n")
    L.append("Hypothesis: a PPO policy trained with modest variation in wheel-motor strength and range-sensor noise (*PPO-randomized*) has a higher "
             "safe-arrival rate under held-out actuator and sensor shifts than a PPO policy trained under nominal conditions (*PPO-nominal*), "
             "possibly at a nominal-performance cost. Both conditions share the same PPO settings, reward, scenario distribution, seeds and transition budget. "
             "Because the intervention is *combined* (motor + noise), any effect is attributed to the combined intervention, not to either component.\n")
    L.append(f"* Evaluation: {n_scen} held-out layouts x {len(suite['conditions'])} conditions x each policy, deterministic inference, paired scenarios and noise streams "
             f"(eval_seed={suite['eval_seed']}). Primary endpoint (specified before the final evaluation): safe-arrival rate under **{primary}** on unseen layouts.")
    L.append("* Safe arrival = verifier success (within 0.30 m, <= 0.10 m/s for 5 consecutive control steps, no wall/obstacle contact). Timeouts at 600 control steps (30 s) are an external cutoff.\n")
    L.append(f"Evaluation episodes: {integ['episodes']} ({integ['cells']} policy x seed x condition cells). Complete: **{meta['complete']}**.\n")

    L.append("## Primary endpoint and secondary conditions\n")
    tab = across.copy()
    tab["safe_arrival_mean"] = tab.mean_success_rate.map(lambda v: f"{100 * v:.1f}%")
    tab["std_over_seeds_pp"] = tab.std_success_rate.map(lambda v: "n/a (1 seed)" if np.isnan(v) else f"{100 * v:.1f}")
    tab["per_seed"] = [", ".join(f"{100 * r:.0f}%" for r in cells[(cells.policy == a) & (cells.condition == b)].success_rate)
                       for a, b in zip(tab.policy, tab.condition)]
    wide = tab.pivot(index="policy", columns="condition", values="safe_arrival_mean")
    wide = wide.loc[[p for p in POLICY_ORDER if p in wide.index], [c for c in CONDITION_ORDER if c in wide.columns]]
    wide.index = [POLICY_LABEL[p] for p in wide.index]
    L.append(f"Safe-arrival rate, mean over training seeds ({n_scen} episodes per seed per cell; `{primary}` is primary):\n")
    L.append(_md_table(wide.reset_index().rename(columns={"index": "policy"})) + "\n")
    L.append("Per-seed values and spread (std across training seeds, in percentage points):\n")
    L.append(_md_table(tab[["policy", "condition", "n_seeds", "per_seed", "std_over_seeds_pp", "pooled_successes", "pooled_episodes"]]) + "\n")
    L.append("Per-seed Wilson 95% intervals describe uncertainty over the sampled *scenarios* only; they do not capture training-seed variability:\n")
    w = cells[["policy", "seed", "condition", "successes", "n_episodes", "success_rate", "wilson_lo", "wilson_hi"]].copy()
    w["wilson_95"] = [f"[{100 * a:.0f}%, {100 * b:.0f}%]" for a, b in zip(w.wilson_lo, w.wilson_hi)]
    w["success_rate"] = (100 * w.success_rate).round(1).astype(str) + "%"
    L.append(_md_table(w[["policy", "seed", "condition", "successes", "n_episodes", "success_rate", "wilson_95"]]) + "\n")

    if len(boot_df):
        L.append("## Paired comparison: PPO-randomized minus PPO-nominal\n")
        L.append("Bootstrap over scenario IDs (10,000 resamples; the same resampled IDs are used for both policies, so the pairing is retained). "
                 "`scenario` interval averages over training seeds and resamples scenarios only. `hierarchical` additionally resamples training seeds "
                 "independently within each policy; with few seeds it is **exploratory** and limited in precision.\n")
        b = boot_df.copy()
        b["diff (pp)"] = b.diff_obs_pp.round(1)
        b["scenario 95% (pp)"] = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(b.scenario_boot_lo_pp, b.scenario_boot_hi_pp)]
        b["hierarchical 95% (pp)"] = [("n/a (<2 seeds)" if np.isnan(lo) else f"[{lo:.1f}, {hi:.1f}]") for lo, hi in zip(b.hier_lo_pp, b.hier_hi_pp)]
        L.append(_md_table(b[["condition", "diff (pp)", "scenario 95% (pp)", "hierarchical 95% (pp)"]], "{:.1f}") + "\n")
        pr = boot_df[boot_df.condition == primary].iloc[0]
        has_h = not np.isnan(pr.hier_lo_pp)
        lo, hi = (pr.hier_lo_pp, pr.hier_hi_pp) if has_h else (pr.scenario_boot_lo_pp, pr.scenario_boot_hi_pp)
        verdict = ("lies entirely above zero" if lo > 0 else "lies entirely below zero" if hi < 0 else "includes zero")
        L.append(f"**Primary endpoint reading ({primary}):** difference {pr.diff_obs_pp:+.1f} pp (randomized minus nominal). "
                 f"Scenario-only 95% interval [{pr.scenario_boot_lo_pp:.1f}, {pr.scenario_boot_hi_pp:.1f}] pp"
                 + (f"; hierarchical (seeds and scenarios) 95% interval [{pr.hier_lo_pp:.1f}, {pr.hier_hi_pp:.1f}] pp, which {verdict}." if has_h else f", which {verdict}.")
                 + " This is a statement about these seeds and these held-out layouts under simulated shifts; it is not evidence about hardware.\n")
        if not (lo > 0):
            L.append("The data show no advantage for randomization on the primary endpoint; an interval that includes zero means no difference could be detected, "
                     "not that none exists. Candidate explanations listed here were **not tested** (§Limits gives a discriminating follow-up): "
                     "stress levels that the nominal policy already largely tolerates; a hidden-parameter / noisy-observation problem that is partially observed for a "
                     "feedforward policy; a transition budget at which neither policy has converged to its robust form; seed-to-seed variance larger than the effect.\n")

    if did_df is not None and len(did_df):
        L.append("### Exploratory (not pre-specified): robustness gap\n")
        L.append("Each policy's drop from its own nominal-condition score to a shifted condition, and the difference of the two drops "
                 "(`did` > 0: the randomized policy loses *less* than the nominal policy; < 0: more). Same bootstrap as above (scenario IDs shared across all four cells). "
                 "Added after seeing validation behaviour to match the hypothesis's wording about robustness under shifts and a possible nominal-performance trade-off; "
                 "treat intervals that include zero as no evidence either way.\n")
        g = did_df.copy()
        g["randomized drop (pp)"] = g.randomized_drop_pp.round(1)
        g["nominal drop (pp)"] = g.nominal_drop_pp.round(1)
        g["did (pp)"] = g.did_pp.round(1)
        g["scenario 95% (pp)"] = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(g.scenario_lo_pp, g.scenario_hi_pp)]
        g["hierarchical 95% (pp)"] = [("n/a" if np.isnan(lo) else f"[{lo:.1f}, {hi:.1f}]") for lo, hi in zip(g.hier_lo_pp, g.hier_hi_pp)]
        L.append(_md_table(g[["condition", "randomized drop (pp)", "nominal drop (pp)", "did (pp)", "scenario 95% (pp)", "hierarchical 95% (pp)"]], "{:.1f}") + "\n")

    if secondary_md:
        L.append(secondary_md)

    L.append("## Baselines\n")
    base = across[across.policy.isin(["scripted", "random"])]
    if len(base):
        bw = base.assign(rate=base.mean_success_rate.map(lambda v: f"{100 * v:.1f}%")).pivot(index="policy", columns="condition", values="rate")
        bw = bw.loc[[p for p in POLICY_ORDER if p in bw.index], [c for c in CONDITION_ORDER if c in bw.columns]]
        bw.index = [POLICY_LABEL[p] for p in bw.index]
        L.append(_md_table(bw.reset_index().rename(columns={"index": "policy"})) + "\n")
        L.append("The scripted reactive controller uses the same 29-element observation as PPO and no map. Its score is a measure of task difficulty "
                 "for a hand-coded controller and bounds how much headroom RL has on this task; random is a sanity floor.\n")

    L.append("## Outcome breakdown, durations and path lengths\n")
    L.append("Outcomes sum to the episode count in every cell (checked). Durations cover all outcomes; time-to-goal and success path length use successes only (n shown); "
             "failure path length is separate. Return columns are **training reward, not task performance**.\n")
    o = cells[["policy", "seed", "condition", "n_episodes", "successes", "collisions", "timeouts", "other_failures",
               "duration_mean_s_all", "time_to_goal_mean_s_successes", "n_time_to_goal", "path_length_mean_m_successes",
               "path_length_mean_m_failures", "TRAINING_REWARD_return_mean"]].copy()
    L.append(_md_table(o, "{:.2f}") + "\n")

    L.append("## Failure taxonomy (automatic tags, multi-label)\n")
    L.append("Heuristic tags from logged episode features; thresholds: " + ", ".join(f"`{k}`={v:.3g}" for k, v in TAXONOMY.items()) +
             ". `stalled_before_obstructed_route` is an extra tag (stopped far from the goal when the straight line was blocked). "
             "Tags are defensible heuristics, not ground-truth causes; manual labels can be merged with `--manual-labels`.\n")
    L.append(_md_table(tax.astype({c: int for c in tax.columns if c not in ("policy", "condition")}), "{:.0f}") + "\n")

    L.append("## Selected episodes\n")
    L.append(f"Rule, fixed before results: in `{primary}`, for training seed {min(sum(seeds.values(), []), default=0)} of each learned policy: the first success and the first two failures by scenario order. "
             "Paired replays use the first three scenarios, pinned in the suite config before evaluation.\n")
    for pol, ids in chosen.items():
        L.append(f"* {POLICY_LABEL[pol]}: " + ", ".join(f"`{i}`" for i in ids))
    L.append("\n![selected episodes](failure_examples.png)\n\n![paired replays](paired_examples.png)\n")

    L.append("## Comparison and learning curves\n")
    L.append("![comparison](comparison.png)\n")
    if curves:
        L.append("![learning curves](learning_curves.png)\n")
        L.append("Learning curves are diagnostics, not task results: the success/return panels are measured on each run's own *training* distribution "
                 "(randomized runs face harder episodes) with stochastic actions.\n")

    L.append("## Training cost\n")
    tc = cells.drop_duplicates(["policy", "seed"])
    tc = tc[tc.policy.isin(["nominal", "randomized"])][["policy", "seed", "run_id", "train_transitions", "train_wallclock_s"]]
    L.append(_md_table(tc, "{:.0f}") + "\n")
    L.append("Wall-clock as measured: the nominal and randomized run of each seed trained concurrently (4 worker processes each) on an otherwise idle machine; "
             "it is not a clean single-run throughput benchmark (see `results/benchmark.json`).\n")

    L.append("## Reproducibility table\n")
    git = meta.get("git") or {}
    valid_commit = bool(re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit") or "")))  # early runs recorded the literal "HEAD" (bug, see DECISIONS.md)
    git_txt = (str(git.get("commit")) + (" (uncommitted changes)" if git.get("dirty") else "")) if valid_commit else "none recorded at run time: these results were produced before the repository's first commit; the source code hash below identifies the code that produced them (DECISIONS.md, Provenance note)"
    rows = [("evaluation suite sha256", meta["suite_sha256"]), (f"{suite['manifest'].split('/')[-1]} sha256", meta["manifest_sha256"]),
            ("environment config + robot.xml hash", ", ".join(sorted(set(df.env_cfg_hash)))), ("source code hash (package .py/.xml)", meta["code_hash"]),
            ("git commit", git_txt),
            ("python / platform", f"{meta['platform']['python']} / {meta['platform']['platform']}")] + \
           [(f"package {k}", v) for k, v in meta["packages"].items()]
    L.append(_md_table(pd.DataFrame(rows, columns=["item", "value"])) + "\n")
    ck = pd.DataFrame([{"run_id": p["run_id"], "policy": p["policy"], "seed": "-" if p["training_seed"] is None else str(int(p["training_seed"])),
                        "train_config_hash": p.get("train_config_hash") or "n/a", "checkpoint": p.get("checkpoint") or "n/a",
                        "checkpoint_sha256": p["checkpoint_sha256"] or "n/a"} for p in meta["policies"]])
    L.append(_md_table(ck) + "\n")

    L.append("## Integrity checks (Gate D)\n")
    L.append(f"* all expected episode records present: **{integ['complete_flag']}**; reconciled counts per cell: outcomes sum to episodes; success flags agree with terminal reasons; "
             "noise seeds shared across policies for every (scenario, condition); single manifest and environment hash.\n"
             "* aggregate tables above are computed from the raw JSONL and re-checked against it before this file is written.\n")

    L.append("## Limits\n")
    L.append("* Simulation only. Sim-to-real transfer and general robotics superiority are not tested.\n"
             "* Goal localisation and velocity are privileged simulator-derived inputs; range noise touches only the 24 range readings.\n"
             "* Stress levels are proposed values, not calibrated hardware distributions; the 0.6 motor multiplier bites mainly under demanding (near-saturating) commands.\n"
             "* Hidden motor gains + noisy ranges make the task partially observed; the policy is feedforward by design.\n"
             "* Grid connectivity is an approximate feasibility check, not a guarantee of dynamic solvability.\n"
             "* One combined intervention: no attribution to motor randomization versus noise randomization.\n"
             f"* {n_seeds} training seeds per condition: bootstrap intervals over seeds are exploratory. Scenarios (not timesteps, not repeated evaluations) are the resampling unit.\n"
             "* A discriminating follow-up experiment: train both conditions to convergence with a longer budget (or a memory-equipped policy) and re-run this exact suite on a *newly generated* held-out set.\n")
    return "\n".join(L)
