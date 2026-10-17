"""Independent verifier + adversarial scorer tests (spec section 5)."""
import numpy as np

from roborobust.config import VerifierCfg
from roborobust.env import RoboRobustEnv
from roborobust.verifier import COLLISION, OUT_OF_BOUNDS, SUCCESS, TIPPED, Verifier
from conftest import make_scenario, run_episode


def v():
    return Verifier(VerifierCfg())


def upd(ver, dist=0.1, speed=0.0, tilt=0.0, x=0.0, y=0.0, collided=False):
    return ver.update(goal_distance=dist, planar_speed=speed, tilt_rad=tilt, x=x, y=y, collided=collided)


def test_success_needs_five_consecutive_steps():
    ver = v()
    outs = [upd(ver) for _ in range(5)]
    assert [o.success for o in outs] == [False, False, False, False, True]
    assert outs[-1].reason == SUCCESS and outs[-1].terminated


def test_entering_goal_too_fast_is_not_success():
    ver = v()
    assert not any(upd(ver, dist=0.05, speed=0.4).terminated for _ in range(50))
    assert ver.dwell == 0


def test_leaving_goal_resets_dwell_counter():
    """3 good steps, one excursion, then 5 good steps. Without a reset success would fire at index 5
    (3 + 1 skipped... dwell would reach 5); with a correct reset it needs five fresh steps -> index 8."""
    ver = v()
    seq = [(0.1, 0)] * 3 + [(0.5, 0)] + [(0.1, 0)] * 5
    outs = [upd(ver, dist=d, speed=s) for d, s in seq]
    assert [o.success for o in outs] == [False] * 8 + [True]
    assert outs[3].dwell == 0 and outs[4].dwell == 1


def test_speed_gate_also_resets_dwell():
    ver = v()
    for _ in range(4):
        upd(ver)
    assert upd(ver, speed=0.3).dwell == 0


def test_collision_same_step_as_arrival_is_failure():
    ver = v()
    for _ in range(4):
        upd(ver)
    out = upd(ver, collided=True)  # 5th qualifying step, but with contact
    assert out.reason == COLLISION and not out.success and out.terminated


def test_collision_beats_tip_and_oob_beats_success():
    assert upd(v(), tilt=1.0, collided=True).reason == COLLISION
    assert upd(v(), tilt=1.0).reason == TIPPED
    assert upd(v(), x=3.2).reason == OUT_OF_BOUNDS


def test_verifier_does_not_import_rewards():
    import ast
    import roborobust.verifier as mod
    tree = ast.parse(open(mod.__file__).read())
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    assert not [m for m in imported if "reward" in m], imported


def test_update_after_terminal_raises():
    ver = v()
    upd(ver, tilt=1.0)
    try:
        upd(ver)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------- env-level adversarial tests
def test_staying_still_does_not_succeed(cfg):
    env = RoboRobustEnv(cfg)
    sc = make_scenario(goal=(1.5, 0.0))
    out = run_episode(env, sc, lambda o: np.zeros(2))
    s = out["infos"][-1]["episode_summary"]
    assert out["truncated"] and not out["terminated"]
    assert s["terminal_reason"] == "timeout" and not s["success"] and s["steps"] == 600


def test_rushing_through_goal_is_not_success_and_ends_in_wall_collision(cfg):
    env = RoboRobustEnv(cfg)
    sc = make_scenario(start=(-1.5, 0.0, 0.0), goal=(0.5, 0.0))
    out = run_episode(env, sc, lambda o: np.array([1.0, 1.0]))
    s = out["infos"][-1]["episode_summary"]
    assert s["terminal_reason"] == "collision" and not s["success"]
    assert s["first_goal_entry_s"] is not None  # it did pass through the goal disc, too fast to count


def _slow_step_with_dwell_preloaded(cfg, yaw):
    """Robot at rest 0.1 mm from contact with the east wall (contact at x = 3.0 - 0.18 = 2.82), goal on
    top of it, dwell counter preloaded to 4 so the goal condition would complete on THIS step.
    yaw=0 drives into the wall, yaw=pi drives away."""
    env = RoboRobustEnv(cfg)
    sc = make_scenario(start=(2.8199, 0.0, yaw), goal=(2.80, 0.0))
    env.reset(seed=0, options={"scenario": sc})
    env.verifier.dwell = 4
    return env.step(np.array([0.1, 0.1]))


def test_touching_wall_and_reaching_goal_on_same_step_is_collision(cfg):
    _, _, te, _, info = _slow_step_with_dwell_preloaded(cfg, yaw=0.0)
    assert te and info["terminal_reason"] == "collision" and not info["success"]
    assert info["collision_event"]["hazard_geom"] == "wall_e"


def test_control_same_state_without_contact_is_success(cfg):
    """Guards the test above: with no contact the identical goal state IS a success."""
    _, _, te, _, info = _slow_step_with_dwell_preloaded(cfg, yaw=np.pi)
    assert te and info["terminal_reason"] == "success" and info["success"]


def test_positive_training_return_without_arrival_is_not_success(cfg):
    """Reward and verifier are independent: exaggerate shaping so the return is positive while the
    robot stops 0.6 m short of the goal."""
    d = cfg.to_dict()
    d["reward"]["shaping_alpha"] = 40.0
    d["max_episode_steps"] = 120
    from roborobust.config import EnvConfig
    env = RoboRobustEnv(EnvConfig.from_dict(d))
    sc = make_scenario(start=(-1.5, 0.0, 0.0), goal=(0.5, 0.0))  # distance 2.0

    def policy(obs):
        dist = obs[24] * env.cfg.room_diagonal
        return np.array([1.0, 1.0]) * float(np.clip(2.0 * (dist - 0.6), 0.0, 1.0))

    out = run_episode(env, sc, policy)
    s = out["infos"][-1]["episode_summary"]
    assert s["return"] > 0.0 and not s["success"] and s["terminal_reason"] == "timeout"
    assert s["features"]["final_goal_distance"] > cfg.verifier.goal_radius


def test_success_bonus_paid_once_and_finished_env_raises(cfg):
    from roborobust.baselines import ScriptedReactive
    env = RoboRobustEnv(cfg)
    pol = ScriptedReactive(cfg)
    sc = make_scenario(start=(-1.0, 0.0, 0.0), goal=(1.0, 0.0))
    out = run_episode(env, sc, pol.act)
    s = out["infos"][-1]["episode_summary"]
    assert s["success"] and s["terminal_reason"] == "success"
    bonuses = [i["reward_components"]["success_bonus"] for i in out["infos"]]
    assert sum(b > 0 for b in bonuses) == 1 and sum(bonuses) == cfg.reward.success_bonus
    try:
        env.step(np.zeros(2))
    except RuntimeError as e:
        assert "reset" in str(e)
    else:
        raise AssertionError("stepping a finished env must raise")
    env.reset(seed=0, options={"scenario": sc})  # and works again after reset
    env.step(np.zeros(2))


def test_timeout_is_truncation_not_termination_and_terminal_event_wins(cfg):
    from roborobust.config import EnvConfig
    d = cfg.to_dict(); d["max_episode_steps"] = 3
    env = RoboRobustEnv(EnvConfig.from_dict(d))
    sc = make_scenario(goal=(2.0, 0.0))
    out = run_episode(env, sc, lambda o: np.zeros(2))
    assert out["truncated"] and not out["terminated"]
    assert out["infos"][-1]["terminal_reason"] == "timeout"
    # a collision on the final allowed step is a termination, not a truncation
    d["max_episode_steps"] = 600
    env2 = RoboRobustEnv(EnvConfig.from_dict(d))
    sc2 = make_scenario(start=(2.5, 0.0, 0.0), goal=(0.0, 0.0))
    out2 = run_episode(env2, sc2, lambda o: np.array([1.0, 1.0]))
    assert out2["terminated"] and not out2["truncated"]


def test_collision_detected_mid_interval_stops_immediately(cfg):
    env = RoboRobustEnv(cfg)
    sc = make_scenario(start=(2.4, 0.0, 0.0), goal=(-2.0, 0.0))
    out = run_episode(env, sc, lambda o: np.array([1.0, 1.0]))
    ev = out["infos"][-1]["collision_event"]
    assert ev is not None and ev["hazard_geom"] == "wall_e"
    assert 1 <= ev["substep"] <= cfg.substeps + 1
    # simulator time stops at the first contact, not at the control boundary
    n_full = len(out["infos"]) * cfg.control_dt
    assert out["infos"][-1]["episode_summary"]["duration_s"] <= n_full + 1e-9


def test_tip_failure_is_reported(cfg):
    env = RoboRobustEnv(cfg)
    sc = make_scenario(goal=(2.0, 0.0))
    env.reset(seed=0, options={"scenario": sc})
    # force a tilted state through the simulator (a test hook, not an action)
    import math, mujoco
    d = env._b.data
    a = math.radians(60)
    d.qpos[3:7] = (math.cos(a / 2), math.sin(a / 2), 0, 0)
    mujoco.mj_forward(env._b.model, d)
    _, _, te, tr, info = env.step(np.zeros(2))
    assert te and info["terminal_reason"] in ("tipped",)
