"""Gate B: environment contract."""
import math

import gymnasium
import numpy as np
import pytest

from roborobust.env import RoboRobustEnv
from roborobust.randomization import DomainParams, RandomizationConfig
from conftest import DATA, make_scenario, run_episode


@pytest.fixture
def env(cfg):
    return RoboRobustEnv(cfg, DATA / "main_val.jsonl")


def test_gymnasium_checker_passes(cfg):
    from gymnasium.utils.env_checker import check_env
    check_env(RoboRobustEnv(cfg, DATA / "main_val.jsonl"), skip_render_check=True)


def test_sb3_checker_passes(cfg):
    from stable_baselines3.common.env_checker import check_env
    check_env(RoboRobustEnv(cfg, DATA / "main_val.jsonl"), warn=True, skip_render_check=True)


def test_observation_contract(env):
    obs, _ = env.reset(seed=1)
    assert obs.shape == (29,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, *_, te, tr, _ = (lambda r: (r[0], r[1], r[2], r[3], r[4]))(env.step(rng.uniform(-1, 1, 2)))
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        if te or tr:
            obs, _ = env.reset()


def test_action_space_and_clipping_is_logged(env):
    assert env.action_space.shape == (2,) and env.action_space.dtype == np.float32
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5, -0.5]))
    assert not info["action_clipped"]
    _, _, _, _, info = env.step(np.array([3.0, -2.0]))
    assert info["action_clipped"]
    assert info["wheel_cmd"] == [10.0, -10.0]  # clipped, then mapped to wheel targets
    assert info["wheel_speed"] is not None and info["actuator_torque"] is not None  # effective actuation recorded


@pytest.mark.parametrize("bad", [[np.nan, 0.0], [0.0, np.inf], [-np.inf, 0.0]])
def test_nonfinite_actions_rejected(env, bad):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array(bad))


def test_episode_clip_count_in_summary(cfg):
    env = RoboRobustEnv(cfg)
    out = run_episode(env, make_scenario(), lambda o: np.array([5.0, 5.0]))
    assert out["infos"][-1]["episode_summary"]["clip_count"] == len(out["infos"])


def test_reset_plus_identical_actions_reproduces_trajectory(cfg):
    env = RoboRobustEnv(cfg, DATA / "main_val.jsonl", randomization=RandomizationConfig(enabled=True))
    sc = env.scenarios[3]
    opts = {"scenario": sc, "domain": DomainParams(0.8, 1.15, 0.02), "noise_seed": 99}
    acts = np.random.default_rng(5).uniform(-1, 1, (150, 2))

    def roll():
        obs, _ = env.reset(seed=11, options=opts)
        rows = [obs]
        for a in acts:
            obs, r, te, tr, _ = env.step(a)
            rows.append(obs)
            if te or tr:
                break
        return np.array(rows), env._b.data.qpos.copy()

    (o1, q1), (o2, q2) = roll(), roll()
    assert o1.shape == o2.shape
    np.testing.assert_allclose(o1, o2, atol=1e-9)
    np.testing.assert_allclose(q1, q2, atol=1e-12)


def test_randomization_does_not_accumulate_across_resets(cfg):
    env = RoboRobustEnv(cfg, DATA / "main_val.jsonl", randomization=RandomizationConfig(enabled=True))
    env.reset(seed=0)
    b = env._b
    for i in range(30):
        env.reset()
        b = env._b
        m, mm = b.model, b.motors
        for aid, mult in zip(mm.ids, (env.domain.left_motor_mult, env.domain.right_motor_mult)):
            np.testing.assert_allclose(m.actuator_forcerange[aid], mm.frange0[aid] * mult, rtol=1e-12)
            np.testing.assert_allclose(m.actuator_gainprm[aid, 0], mm.gain0[aid, 0] * mult, rtol=1e-12)
            np.testing.assert_allclose(m.actuator_biasprm[aid, 2], mm.bias0[aid, 2] * mult, rtol=1e-12)
    # pinning the nominal domain returns exactly to nominal arrays on the same cached model
    sc = env.scenario
    env.reset(options={"scenario": sc, "domain": DomainParams()})
    np.testing.assert_array_equal(env._b.model.actuator_forcerange, env._b.motors.frange0)
    # and the nominal arrays themselves match the documented config
    assert env._b.motors.frange0[0, 1] == pytest.approx(cfg.robot.motor_force_limit)


def test_training_randomization_ranges_and_independence(cfg):
    env = RoboRobustEnv(cfg, DATA / "main_val.jsonl", randomization=RandomizationConfig(enabled=True))
    env.reset(seed=3)
    ds = []
    for _ in range(400):
        env.reset()
        ds.append(env.domain)
    L = np.array([d.left_motor_mult for d in ds]); R = np.array([d.right_motor_mult for d in ds])
    N = np.array([d.range_noise_std for d in ds])
    assert 0.8 <= L.min() and L.max() <= 1.2 and 0.8 <= R.min() and R.max() <= 1.2 and 0 <= N.min() and N.max() <= 0.03
    assert L.std() > 0.08 and N.std() > 0.006  # genuinely varied
    assert abs(np.corrcoef(L, R)[0, 1]) < 0.2 and abs(np.corrcoef(L, N)[0, 1]) < 0.2  # independent draws


def test_nominal_env_never_randomizes(cfg):
    env = RoboRobustEnv(cfg, DATA / "main_val.jsonl", randomization=RandomizationConfig(enabled=False))
    env.reset(seed=0)
    for _ in range(20):
        env.reset()
        assert env.domain == DomainParams()


def test_seeded_streams_do_not_duplicate_unintentionally(cfg):
    """SB3 seeds worker i with seed+i. Different workers must see different scenario sequences, and the
    three internal streams must be mutually independent."""
    seqs = []
    for s in (0, 1, 2):
        e = RoboRobustEnv(cfg, DATA / "main_val.jsonl")
        e.reset(seed=s)
        ids = [e.scenario.scenario_id]
        for _ in range(15):
            e.reset(); ids.append(e.scenario.scenario_id)
        seqs.append(ids)
    assert len({tuple(s) for s in seqs}) == 3
    e = RoboRobustEnv(cfg, DATA / "main_val.jsonl"); e.reset(seed=7)
    draws = [e._rng_geometry.random(8), e._rng_domain.random(8), e._rng_noise.random(8)]
    assert not np.allclose(draws[0], draws[1]) and not np.allclose(draws[1], draws[2]) and not np.allclose(draws[0], draws[2])


def test_nominal_and_randomized_envs_share_scenario_sequence(cfg):
    """Paired training distribution: same seed -> same scenario order regardless of randomization."""
    def ids(rand):
        e = RoboRobustEnv(cfg, DATA / "main_val.jsonl", randomization=RandomizationConfig(enabled=rand))
        e.reset(seed=4)
        out = [e.scenario.scenario_id]
        for _ in range(25):
            e.reset(); out.append(e.scenario.scenario_id)
        return out
    assert ids(False) == ids(True)


def test_noise_only_affects_ranges_and_is_clipped(cfg):
    sc = make_scenario(start=(-1.0, 0.2, 0.3), goal=(1.0, -1.0), obstacles=[(0.5, 0.5, 0.3, 0.3)])
    e = RoboRobustEnv(cfg)
    clean, _ = e.reset(seed=0, options={"scenario": sc, "domain": DomainParams(1, 1, 0.0), "noise_seed": 1})
    noisy, _ = e.reset(seed=0, options={"scenario": sc, "domain": DomainParams(1, 1, 0.06), "noise_seed": 1})
    assert not np.allclose(clean[:24], noisy[:24])
    np.testing.assert_array_equal(clean[24:], noisy[24:])  # goal + velocity features untouched
    assert noisy[:24].min() >= 0.0 and noisy[:24].max() <= 1.0
    # large noise still clipped
    wild, _ = e.reset(seed=0, options={"scenario": sc, "domain": DomainParams(1, 1, 5.0), "noise_seed": 1})
    assert wild[:24].min() >= 0.0 and wild[:24].max() <= 1.0


def test_noise_stream_is_pinned_by_noise_seed_and_independent_of_other_streams(cfg):
    sc = make_scenario(obstacles=[(1.0, 0.0, 0.3, 0.3)], goal=(2.0, 2.0))
    e = RoboRobustEnv(cfg)
    dom = DomainParams(1, 1, 0.05)
    a, _ = e.reset(seed=0, options={"scenario": sc, "domain": dom, "noise_seed": 5})
    b, _ = e.reset(seed=999, options={"scenario": sc, "domain": dom, "noise_seed": 5})  # different seed, same noise_seed
    c, _ = e.reset(seed=0, options={"scenario": sc, "domain": dom, "noise_seed": 6})
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_motor_gains_are_not_visible_in_the_observation(cfg):
    """Hidden dynamics: at t=0 the observation is identical across motor multipliers."""
    sc = make_scenario(obstacles=[(1.0, 0.0, 0.3, 0.3)], goal=(2.0, 2.0))
    e = RoboRobustEnv(cfg)
    a, _ = e.reset(seed=0, options={"scenario": sc, "domain": DomainParams(0.6, 1.0, 0.0)})
    b, _ = e.reset(seed=0, options={"scenario": sc, "domain": DomainParams(1.0, 1.0, 0.0)})
    np.testing.assert_array_equal(a, b)


def test_all_episode_state_reset(cfg):
    e = RoboRobustEnv(cfg)
    sc = make_scenario()
    run_episode(e, sc, lambda o: np.array([2.0, 2.0]))  # leaves clip counts, path length etc.
    e.reset(seed=0, options={"scenario": sc})
    assert e._step_count == 0 and e._clip_count == 0 and e._path_len == 0.0 and e.verifier.dwell == 0
    assert not e.verifier.ever_collided and e._collision_event is None
    np.testing.assert_array_equal(e._last_action, np.zeros(2))
    assert all(v == 0.0 for v in e._comp_sum.values()) and e._ret == 0.0 and not e._done


def test_robot_config_matches_model(cfg):
    from roborobust.env import build_model, check_robot_config
    assert check_robot_config(build_model([]), cfg) == []
    bad = cfg.to_dict(); bad["robot"]["wheel_radius"] = 0.07
    from roborobust.config import EnvConfig
    assert check_robot_config(build_model([]), EnvConfig.from_dict(bad))  # mismatch is detected


def test_render_rgb_array_optional(cfg):
    env = RoboRobustEnv(cfg, DATA / "main_val.jsonl", render_mode="rgb_array")
    env.reset(seed=0)
    try:
        frame = env.render()
    except Exception as e:  # rendering must never block headless use
        pytest.skip(f"renderer unavailable here: {e}")
    assert frame.shape == (480, 640, 3)
    env.close()


# ------------------------------------------------------------------ SB3 time-limit semantics
def _tiny_cfg(cfg, steps):
    from roborobust.config import EnvConfig
    d = cfg.to_dict(); d["max_episode_steps"] = steps
    return EnvConfig.from_dict(d)


def test_vecenv_marks_timeout_as_truncation_and_collision_as_termination(cfg):
    from stable_baselines3.common.vec_env import DummyVecEnv
    sc_timeout = make_scenario(goal=(2.0, 0.0))
    v = DummyVecEnv([lambda: RoboRobustEnv(_tiny_cfg(cfg, 4), [sc_timeout])])
    v.reset()
    for _ in range(4):
        _, _, dones, infos = v.step(np.zeros((1, 2)))
    assert dones[0] and infos[0]["TimeLimit.truncated"] is True and "terminal_observation" in infos[0]

    sc_crash = make_scenario(start=(2.3, 0.0, 0.0), goal=(0.0, 0.0))
    v = DummyVecEnv([lambda: RoboRobustEnv(cfg, [sc_crash])])
    v.reset()
    for _ in range(60):  # ~0.52 m at ~0.45 m/s is ~24 control steps
        _, _, dones, infos = v.step(np.ones((1, 2)))
        if dones[0]:
            break
    assert dones[0] and infos[0]["TimeLimit.truncated"] is False and infos[0]["terminal_reason"] == "collision"


def test_ppo_bootstraps_value_through_time_limit_but_not_through_terminal(cfg):
    """Numerical check of the SB3 contract on the *actual* collected rollout:
       truncation  -> stored reward = r + gamma * V(s_T)   (bootstrapped through the cutoff)
       termination -> stored reward = r                      (value of a terminal state is 0)"""
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def collect(scenario, n_steps, max_steps, force_action=None):
        seen = []

        class Spy(RoboRobustEnv):
            def step(self, a):
                # force_action overrides what the env executes (the PPO buffer logic under test only
                # depends on the env's reward/terminated/truncated stream, not on the policy)
                obs, r, te, tr, info = super().step(a if force_action is None else force_action)
                seen.append({"r": r, "te": te, "tr": tr, "obs": obs.copy()})
                return obs, r, te, tr, info

        v = DummyVecEnv([lambda: Spy(_tiny_cfg(cfg, max_steps), [scenario])])
        model = PPO("MlpPolicy", v, n_steps=n_steps, batch_size=n_steps, n_epochs=1, learning_rate=0.0,
                    gamma=0.99, seed=0, device="cpu")
        model.learn(total_timesteps=n_steps)  # lr=0 -> parameters unchanged, so V is what collection used
        return model, seen[:n_steps], model.rollout_buffer.rewards[:, 0].copy()

    # --- truncation: 4-step episodes, never reaches a terminal event ---
    model, seen, buf = collect(make_scenario(goal=(2.0, 0.0)), n_steps=8, max_steps=4)
    trunc = [i for i, x in enumerate(seen) if x["tr"]]
    assert trunc == [3, 7] and not any(x["te"] for x in seen)
    for i in trunc:
        with th.no_grad():
            v_T = model.policy.predict_values(th.as_tensor(seen[i]["obs"][None]))[0, 0].item()
        assert abs(v_T) > 1e-6  # the check is not vacuous
        assert buf[i] == pytest.approx(seen[i]["r"] + 0.99 * v_T, abs=1e-5)
    for i in (0, 1, 2, 4, 5, 6):  # non-final steps are untouched
        assert buf[i] == pytest.approx(seen[i]["r"], abs=1e-6)

    # --- termination: collision episode, stored reward must equal the raw (penalty-bearing) reward ---
    model, seen, buf = collect(make_scenario(start=(2.5, 0.0, 0.0), goal=(0.0, 0.0)), n_steps=64, max_steps=600,
                                  force_action=np.ones(2))
    term = [i for i, x in enumerate(seen) if x["te"]]
    assert term, "expected a collision inside the rollout"
    for i in term:
        assert not seen[i]["tr"]
        assert buf[i] == pytest.approx(seen[i]["r"], abs=1e-6)
