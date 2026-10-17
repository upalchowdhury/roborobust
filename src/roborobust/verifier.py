"""Independent, state-based success/failure verifier.

Deliberately imports nothing from rewards.py and never sees a reward or an observation: success is
decided from *true simulator state* and the collision flag only. Reward thresholds can be changed
without changing what counts as success.

Priority within one control step:  collision > tipped > out_of_bounds > success.
(A step that both touches an obstacle and satisfies the goal condition is a collision.)
Timeout is not decided here: it is an external step cutoff applied by the environment, and only when
no terminal event occurred on the same step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from roborobust.config import VerifierCfg

SUCCESS = "success"
COLLISION = "collision"
TIPPED = "tipped"
OUT_OF_BOUNDS = "out_of_bounds"
TIMEOUT = "timeout"
FAILURE_REASONS = (COLLISION, TIPPED, OUT_OF_BOUNDS)


@dataclass(frozen=True)
class Outcome:
    terminated: bool
    success: bool
    reason: str | None  # one of SUCCESS / COLLISION / TIPPED / OUT_OF_BOUNDS, else None
    dwell: int  # consecutive qualifying steps so far (after this update)


class Verifier:
    def __init__(self, cfg: VerifierCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.dwell = 0
        self.ever_collided = False
        self._finished = False

    def update(
        self,
        *,
        goal_distance: float,
        planar_speed: float,
        tilt_rad: float,
        x: float,
        y: float,
        collided: bool,
    ) -> Outcome:
        """Evaluate one control boundary from true state. Raises if called after a terminal outcome."""
        if self._finished:
            raise RuntimeError("Verifier.update() after a terminal outcome; call reset().")
        for v in (goal_distance, planar_speed, tilt_rad, x, y):
            if not math.isfinite(v):
                raise FloatingPointError("non-finite state passed to verifier")

        if collided:
            self.ever_collided = True

        in_goal = goal_distance <= self.cfg.goal_radius and planar_speed <= self.cfg.max_success_speed
        self.dwell = self.dwell + 1 if in_goal else 0

        reason: str | None = None
        if self.ever_collided:
            reason = COLLISION
        elif tilt_rad > math.radians(self.cfg.max_tilt_deg):
            reason = TIPPED
        elif max(abs(x), abs(y)) > self.cfg.arena_half_extent:
            reason = OUT_OF_BOUNDS
        elif self.dwell >= self.cfg.dwell_steps:
            reason = SUCCESS

        if reason is not None:
            self._finished = True
        return Outcome(
            terminated=reason is not None,
            success=reason == SUCCESS,
            reason=reason,
            dwell=self.dwell,
        )
