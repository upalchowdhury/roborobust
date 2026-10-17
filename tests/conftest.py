import math
from pathlib import Path

import numpy as np
import pytest

from roborobust.config import load_env_config
from roborobust.env import RoboRobustEnv
from roborobust.scenarios import Obstacle, Scenario, load_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scenarios"


@pytest.fixture(scope="session")
def cfg():
    return load_env_config(ROOT / "configs" / "env.yaml")[0]


@pytest.fixture(scope="session")
def val_scenarios():
    return load_manifest(DATA / "main_val.jsonl")


def make_scenario(start=(0.0, 0.0, 0.0), goal=(2.0, 0.0), obstacles=(), sid="t0"):
    obs = [Obstacle(*o) for o in obstacles]
    return Scenario(
        scenario_id=sid, split="dev", family="main" if obs else "empty",
        layout_id=f"T{abs(hash(tuple(obstacles))) % 10**8}" if obs else "empty",
        start={"x": start[0], "y": start[1], "yaw": start[2]}, goal={"x": goal[0], "y": goal[1]}, obstacles=obs,
    )


@pytest.fixture
def mk():
    return make_scenario


def run_episode(env, scenario, action_fn, seed=0, options=None, max_steps=10_000):
    obs, info = env.reset(seed=seed, options={"scenario": scenario, **(options or {})})
    out = {"obs": [obs], "info": [info], "rewards": [], "infos": []}
    for _ in range(max_steps):
        obs, r, te, tr, inf = env.step(action_fn(obs))
        out["obs"].append(obs); out["rewards"].append(r); out["infos"].append(inf)
        if te or tr:
            out["terminated"], out["truncated"] = te, tr
            return out
    raise AssertionError("episode did not finish")
