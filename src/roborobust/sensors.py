"""Range sensing and observation assembly.

Observation (float32, 29 elements):
  [0:24]  planar ranges / max_range      ray 0 = straight ahead, rays CCW, 15 deg apart
  [24]    goal distance / room diagonal   clipped to [0, 1]
  [25]    sin(goal bearing - heading)
  [26]    cos(goal bearing - heading)
  [27]    body-frame forward velocity / v_bound, clipped to [-1, 1]
  [28]    yaw rate / w_bound, clipped to [-1, 1]

Goal position and velocities are *privileged simulator-derived* inputs (disclosed in the README).
Noise touches only the range readings.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from roborobust.config import SensingCfg

OBS_DIM_EXTRA = 5
# Geom group 0 = walls + obstacles (the only ray-visible group). Robot (1), floor/goal marker (2) are not.
RAY_GEOMGROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class RangeSensor:
    def __init__(self, cfg: SensingCfg, chassis_body_id: int):
        self.cfg = cfg
        self.chassis_body_id = chassis_body_id
        self.angles = 2.0 * math.pi * np.arange(cfg.n_rays) / cfg.n_rays
        self._geomid = np.zeros(cfg.n_rays, dtype=np.int32)
        self._dist = np.zeros(cfg.n_rays, dtype=np.float64)

    def ranges(self, model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, yaw: float) -> np.ndarray:
        """True (noise-free) ranges in metres from the robot centre; `max_range` where nothing is hit.

        Robot geometry is excluded twice over: its body is passed as `bodyexclude` and its geoms are
        in a group the mask does not include. The goal marker is non-colliding and in group 2, so it
        is invisible to rays as well.
        """
        a = yaw + self.angles
        vec = np.zeros((self.cfg.n_rays, 3))
        vec[:, 0], vec[:, 1] = np.cos(a), np.sin(a)
        pnt = np.array([x, y, self.cfg.ray_height])
        mujoco.mj_multiRay(
            model, data, pnt, vec.ravel(), RAY_GEOMGROUP, 1, self.chassis_body_id,
            self._geomid, self._dist, None, self.cfg.n_rays, self.cfg.max_range,
        )
        # mj_multiRay's `cutoff` only culls by bounding volume; it does NOT truncate hit distances
        # (a wall 3.8 m away was reported as 3.8 m). Clip explicitly so readings live in [0, max_range].
        return np.minimum(np.where(self._geomid >= 0, self._dist, self.cfg.max_range), self.cfg.max_range)


def add_range_noise(ranges_m: np.ndarray, std: float, rng: np.random.Generator, max_range: float) -> np.ndarray:
    """Fresh i.i.d. Gaussian noise per reading, clipped back to the valid [0, max_range] interval."""
    if std <= 0.0:
        return ranges_m
    return np.clip(ranges_m + rng.normal(0.0, std, size=ranges_m.shape), 0.0, max_range)


def build_observation(
    ranges_m: np.ndarray,
    goal_dist: float,
    bearing_rel: float,
    v_forward: float,
    yaw_rate: float,
    cfg: SensingCfg,
    room_diagonal: float,
) -> np.ndarray:
    obs = np.empty(cfg.n_rays + OBS_DIM_EXTRA, dtype=np.float32)
    obs[: cfg.n_rays] = ranges_m / cfg.max_range
    obs[cfg.n_rays + 0] = min(max(goal_dist / room_diagonal, 0.0), 1.0)
    obs[cfg.n_rays + 1] = math.sin(bearing_rel)
    obs[cfg.n_rays + 2] = math.cos(bearing_rel)
    obs[cfg.n_rays + 3] = min(max(v_forward / cfg.v_bound, -1.0), 1.0)
    obs[cfg.n_rays + 4] = min(max(yaw_rate / cfg.w_bound, -1.0), 1.0)
    return obs


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
