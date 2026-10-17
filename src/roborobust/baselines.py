"""Baseline policies that use exactly the same 29-element observation interface as PPO.

ScriptedReactive: goal attraction + obstacle avoidance by *free-corridor* steering. From the 24 range
rays it reconstructs hit points, then among candidate headings picks the one closest to the goal
bearing whose forward corridor (robot half-width + margin wide) is free of hits. Speed is limited by
turn error, nearest obstacle ahead, and distance to the goal (so it can satisfy the verifier's
"slow enough to count as arrived" condition). It reads only `obs`: no scenario, map or true state.

It is a *task-difficulty probe*, not a strong planner: it is stateless and cannot escape traps
(no memory), which is exactly what the comparison with PPO is meant to expose. Parameters were tuned on
validation scenarios only.
"""
from __future__ import annotations

import math

import numpy as np

from roborobust.config import EnvConfig


class Policy:
    label = "policy"

    def reset(self, seed: int | None = None) -> None:  # per-episode hook
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniform random wheel commands in [-1, 1]^2, from an explicit seed (diagnostic floor)."""

    label = "random"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(np.random.SeedSequence(0 if seed is None else seed))

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def decode_observation(obs: np.ndarray, cfg: EnvConfig) -> dict:
    n = cfg.sensing.n_rays
    return {
        "ranges": obs[:n].astype(np.float64) * cfg.sensing.max_range,
        "goal_dist": float(obs[n]) * cfg.room_diagonal,
        "bearing": math.atan2(float(obs[n + 1]), float(obs[n + 2])),
        "v_fwd": float(obs[n + 3]) * cfg.sensing.v_bound,
        "yaw_rate": float(obs[n + 4]) * cfg.sensing.w_bound,
    }


class ScriptedReactive(Policy):
    label = "scripted"

    def __init__(
        self,
        cfg: EnvConfig,
        v_max: float = 0.42,          # m/s; ~0.45 is the speed a full-scale command actually delivers
        corridor_half_width: float = 0.27,  # robot radius 0.18 + 0.09 margin
        lookahead: float = 0.9,       # m
        k_turn: float = 2.2,          # rad/s per rad of heading error
        goal_gain: float = 0.45,      # v <= goal_gain * goal_distance  (slow arrival)
        stop_dist: float = 0.10,      # m, command zero inside this
    ):
        self.cfg = cfg
        self.v_max, self.w_half, self.L = v_max, corridor_half_width, lookahead
        self.k_turn, self.goal_gain, self.stop_dist = k_turn, goal_gain, stop_dist
        n = cfg.sensing.n_rays
        self.ray_angles = 2 * math.pi * np.arange(n) / n
        # Candidate headings: twice the ray density.
        self.cands = 2 * math.pi * np.arange(2 * n) / (2 * n)
        self.max_range = cfg.sensing.max_range
        self.half_track = cfg.robot.track_width / 2
        self.v_ideal = cfg.robot.max_wheel_speed * cfg.robot.wheel_radius  # 0.5 m/s at action 1.0

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = decode_observation(obs, self.cfg)
        r, d_goal, bearing = o["ranges"], o["goal_dist"], o["bearing"]

        hit = r < self.max_range - 1e-6
        hx, hy = r * np.cos(self.ray_angles), r * np.sin(self.ray_angles)
        L = min(self.L, max(d_goal + 0.15, 0.5))
        # corridor test for every candidate heading: any hit inside the (L x 2*w_half) strip ahead?
        c, s = np.cos(self.cands)[:, None], np.sin(self.cands)[:, None]
        along = hx[None, :] * c + hy[None, :] * s
        lateral = -hx[None, :] * s + hy[None, :] * c
        blocked = hit[None, :] & (along > 0.0) & (along < L) & (np.abs(lateral) < self.w_half)
        free = ~blocked.any(axis=1)

        if free.any():
            err_all = np.abs(self._wrap(self.cands - bearing))
            err_all = np.where(free, err_all, np.inf)
            k = int(np.argmin(err_all))
            heading = float(self._wrap(self.cands[k]))
        else:  # boxed in: turn toward the most open direction, do not drive
            heading = float(self._wrap(self.ray_angles[int(np.argmax(r))]))
            w = float(np.clip(self.k_turn * heading, -3.0, 3.0))
            return self._to_action(0.0, w)

        # Speed: slow for large turn error, near obstacles ahead, and on approach to the goal.
        ahead = float(r[np.abs(self._wrap(self.ray_angles - heading)) < math.radians(35)].min())
        v = self.v_max * max(0.0, math.cos(min(abs(heading), math.pi / 2))) ** 2
        v = min(v, self.v_max * float(np.clip((ahead - 0.45) / 0.9, 0.25, 1.0)))
        v = min(v, self.goal_gain * d_goal)
        if d_goal < self.stop_dist:
            v = 0.0
        w = float(np.clip(self.k_turn * heading, -3.0, 3.0))
        if d_goal < 0.30 and abs(bearing) > 1.0:
            v = 0.0  # inside the goal disc but facing away: do not drive off, just stop
            w = 0.0
        return self._to_action(v, w)

    def _to_action(self, v: float, w: float) -> np.ndarray:
        v_l, v_r = v - w * self.half_track, v + w * self.half_track
        a = np.array([v_l, v_r]) / self.v_ideal
        m = np.abs(a).max()
        if m > 1.0:
            a = a / m
        return a.astype(np.float32)
