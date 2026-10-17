import numpy as np
import pandas as pd
import pytest

from roborobust.analyze import TAXONOMY, paired_bootstrap, tag_failures, wilson


def test_wilson_known_values():
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=2e-3) and hi == pytest.approx(0.9433, abs=2e-3)
    assert wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12) and wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    lo, hi = wilson(25, 50)
    assert lo == pytest.approx(0.3664, abs=2e-3) and hi == pytest.approx(0.6336, abs=2e-3)
    assert all(np.isnan(wilson(0, 0)))


def test_paired_bootstrap_retains_pairing():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, (3, 50)).astype(float)
    same = paired_bootstrap(a, a.copy(), n_boot=500, seed=1)
    assert same["observed"] == 0 and same["lo"] == 0 and same["hi"] == 0  # identical policies: diff is 0 in EVERY resample
    assert paired_bootstrap(a, a, n_boot=500, seed=1, hierarchical=False) == paired_bootstrap(a, a, n_boot=500, seed=1, hierarchical=False)


def test_paired_bootstrap_detects_a_real_difference_and_is_not_dominated_by_noise():
    rng = np.random.default_rng(1)
    base = rng.random(50)
    a = (rng.random((3, 50)) < np.clip(base + 0.3, 0, 1)[None, :]).astype(float)   # strong
    b = (rng.random((3, 50)) < np.clip(base - 0.3, 0, 1)[None, :]).astype(float)   # weak
    r = paired_bootstrap(a, b, n_boot=2000, seed=2, hierarchical=True)
    assert r["observed"] > 0.2 and r["lo"] > 0
    r1 = paired_bootstrap(a[:1], b[:1], n_boot=2000, seed=2)
    assert r1["lo"] <= r1["observed"] <= r1["hi"]


def test_hierarchical_interval_is_wider_than_scenario_only_when_seeds_disagree():
    a = np.array([np.ones(50), np.zeros(50), np.ones(50) * 0.0], dtype=float)  # seeds disagree wildly
    b = np.full((3, 50), 0.5)
    s = paired_bootstrap(a, b, n_boot=3000, seed=3)
    h = paired_bootstrap(a, b, n_boot=3000, seed=3, hierarchical=True)
    assert (h["hi"] - h["lo"]) > (s["hi"] - s["lo"])


def _row(**kw):
    base = dict(episode_id="e", success=False, terminal_reason="timeout", f_final_yaw_rate=0.0, f_left_clearance_end_m=2.0,
                f_right_clearance_end_m=2.0, f_final_speed=0.0, f_final_goal_distance=2.0, f_yaw_travel_rad=2.0,
                f_straight_line_blocked=False, f_straight_cmd_steps=0, f_straight_cmd_mean_yaw_rate=np.nan,
                p_left_motor_mult=1.0, p_right_motor_mult=1.0, collision_event={"hazard_geom": "obs_0"})
    base.update(kw)
    return base


def test_taxonomy_tags():
    rows = [
        _row(episode_id="turn", terminal_reason="collision", f_final_yaw_rate=1.5),
        _row(episode_id="passage", terminal_reason="collision", f_left_clearance_end_m=0.4, f_right_clearance_end_m=0.5),
        _row(episode_id="short", f_final_goal_distance=0.45),
        _row(episode_id="circle", f_yaw_travel_rad=20.0, f_final_speed=0.5, f_final_goal_distance=1.0),
        _row(episode_id="clear", f_final_goal_distance=2.5, f_final_speed=0.3),
        _row(episode_id="stall_blocked", f_straight_line_blocked=True, f_final_goal_distance=2.5),
        _row(episode_id="drift", terminal_reason="collision", p_left_motor_mult=0.6, f_straight_cmd_steps=60, f_straight_cmd_mean_yaw_rate=0.25),
        _row(episode_id="no_drift_symmetric", terminal_reason="collision", f_straight_cmd_steps=60, f_straight_cmd_mean_yaw_rate=0.25),
        _row(episode_id="win", success=True, terminal_reason="success"),
    ]
    df = tag_failures(pd.DataFrame(rows))
    t = dict(zip(df.episode_id, df.tags))
    assert "collision_while_turning" in t["turn"]
    assert "collision_near_passage" in t["passage"]
    assert "stopping_short" in t["short"]
    assert "circling_oscillation" in t["circle"]
    assert "timeout_despite_clear_route" in t["clear"]
    assert "stalled_before_obstructed_route" in t["stall_blocked"] and "timeout_despite_clear_route" not in t["stall_blocked"]
    assert "motor_asymmetry_drift" in t["drift"] and "motor_asymmetry_drift" not in t["no_drift_symmetric"]
    assert "collision_with_obstacle" in t["turn"] and "collision_en_route" in t["turn"]
    wall = tag_failures(pd.DataFrame([_row(episode_id="w", terminal_reason="collision", f_final_goal_distance=0.5,
                                           collision_event={"hazard_geom": "wall_e"})]))
    assert {"collision_with_wall", "collision_near_goal"} <= set(wall.tags.iloc[0])
    assert t["win"] == []
    manual = tag_failures(pd.DataFrame([_row(episode_id="m")]), {"m": "my label"})
    assert "manual:my label" in manual.tags.iloc[0]


def test_did_bootstrap_identical_policies_give_zero_and_detects_asymmetric_loss():
    from roborobust.analyze import did_bootstrap
    rng = np.random.default_rng(0)
    a0 = (rng.random((3, 50)) < 0.8).astype(float)
    ac = (rng.random((3, 50)) < 0.7).astype(float)
    same = did_bootstrap(ac, a0, ac.copy(), a0.copy(), n_boot=300, seed=1)
    assert same["observed"] == 0 and same["lo"] == 0 and same["hi"] == 0  # pairing retained across all four cells
    b0 = a0.copy()
    bc = np.zeros_like(a0)  # nominal policy collapses under shift, randomized does not
    r = did_bootstrap(a0.copy(), a0, bc, b0, n_boot=2000, seed=2, hierarchical=True)
    assert r["observed"] > 0.5 and r["lo"] > 0
