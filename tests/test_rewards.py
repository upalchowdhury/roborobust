import math

import numpy as np
import pytest

from roborobust.config import RewardCfg
from roborobust.rewards import RewardFn
from roborobust.verifier import Outcome

NONTERM = Outcome(False, False, None, 0)
SUCC = Outcome(True, True, "success", 5)
FAIL = Outcome(True, False, "collision", 0)
DIAG = 6 * 2 ** 0.5


def fn(**kw):
    kw.setdefault("shaping_alpha", 1.0)  # unit tests of the formula use alpha=1, gamma=0.99; the env uses its configured values
    kw.setdefault("gamma", 0.99)
    return RewardFn(RewardCfg(**kw), DIAG)


def test_components_sum_to_total_and_values():
    r = fn().compute(2.0, 1.9, NONTERM)
    assert r.step_cost == -0.01 and r.success_bonus == 0 and r.failure_penalty == 0
    assert r.shaping == pytest.approx(0.99 * (-1.9 / DIAG) - (-2.0 / DIAG))
    assert r.total == pytest.approx(r.step_cost + r.shaping)


def test_terminal_next_potential_is_zero_for_success_and_failure():
    f = fn()
    for out in (SUCC, FAIL):
        r = f.compute(0.4, 0.2, out)
        assert r.shaping == pytest.approx(0.0 - (-0.4 / DIAG))
    assert f.compute(0.4, 0.2, SUCC).success_bonus == 10.0
    assert f.compute(0.4, 0.2, FAIL).failure_penalty == -10.0


def test_truncation_keeps_true_next_potential():
    """A time-limit step is NOT terminal: the shaping term uses the real next potential."""
    r = fn().compute(2.0, 1.5, NONTERM)  # what the env passes on the truncation step
    assert r.shaping == pytest.approx(0.99 * (-1.5 / DIAG) + 2.0 / DIAG)


def test_discounted_shaping_telescopes_over_oscillating_trajectory():
    """sum_t gamma^t F_t == gamma^T Phi(s_T) - Phi(s_0) for non-terminal end (alpha=1)."""
    f = fn(shaping_alpha=1.0, gamma=0.99)
    d = [3.0, 2.5, 2.9, 2.2, 2.6, 2.1, 2.4, 2.0]  # toward / away / toward ...
    total = sum(0.99 ** t * f.compute(d[t], d[t + 1], NONTERM).shaping for t in range(len(d) - 1))
    T = len(d) - 1
    assert total == pytest.approx(0.99 ** T * f.potential(d[-1]) - f.potential(d[0]))
    # undiscounted sum is NOT required to be zero (and generally is not)
    undiscounted = sum(f.compute(d[t], d[t + 1], NONTERM).shaping for t in range(T))
    assert abs(undiscounted) > 1e-6


def test_discounted_shaping_telescopes_with_terminal_zero():
    f = fn()
    d = [2.0, 1.2, 1.5, 0.9, 0.2]
    outs = [NONTERM] * 3 + [SUCC]
    total = sum(0.99 ** t * f.compute(d[t], d[t + 1], outs[t]).shaping for t in range(4))
    assert total == pytest.approx(0.99 ** 4 * 0.0 - f.potential(d[0]))


def test_env_shaping_matches_identity_on_real_oscillating_rollout(cfg):
    from roborobust.env import RoboRobustEnv
    from conftest import make_scenario
    env = RoboRobustEnv(cfg)
    sc = make_scenario(start=(-1.5, 0.0, 0.0), goal=(0.0, 1.5))
    env.reset(seed=0, options={"scenario": sc})
    dists, shaping = [env._prev_dist], []
    for t in range(60):
        a = np.array([1.0, 1.0]) if (t // 10) % 2 == 0 else np.array([-1.0, -1.0])
        _, _, te, tr, info = env.step(a)
        shaping.append(info["reward_components"]["shaping"]); dists.append(info["goal_distance"])
        assert not (te or tr)
    g, alpha = cfg.reward.gamma, cfg.reward.shaping_alpha
    total = sum(g ** t * s for t, s in enumerate(shaping))
    T = len(shaping)
    phi = lambda d: -d / env.cfg.room_diagonal  # noqa: E731
    assert total == pytest.approx(alpha * (g ** T * phi(dists[-1]) - phi(dists[0])))
    assert max(dists) > dists[0] > min(dists) or max(dists) != min(dists)
