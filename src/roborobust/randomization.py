"""Domain-randomization parameters, sampling and *clean* application to a MuJoCo model.

The motor multiplier scales the motor's whole torque output: the velocity-actuator gain
(gainprm[0], biasprm[2]) and its torque limit (forcerange) together. A multiplier of 0.6 therefore
means "this motor delivers 60 % of the torque for the same command and saturates at 60 % of the
nominal limit". Under a light command a velocity servo hides much of this (it simply works a bit
harder); under a demanding command the torque limit binds and wheel speed really drops. That is
why the effect is verified under a demanding command (validation.py, tests/test_physics.py).

Application always *restores the nominal arrays first*, then multiplies, so repeated resets can
never compound a multiplier (Gate B).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

LEFT_MOTOR = "left_wheel_motor"
RIGHT_MOTOR = "right_wheel_motor"


@dataclass(frozen=True)
class DomainParams:
    left_motor_mult: float = 1.0
    right_motor_mult: float = 1.0
    range_noise_std: float = 0.0  # metres, std of Gaussian noise added to every range reading

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_any(cls, x) -> "DomainParams":
        if isinstance(x, cls):
            return x
        if x is None:
            return cls()
        return cls(**{k: float(v) for k, v in dict(x).items()})


@dataclass
class RandomizationConfig:
    enabled: bool = False
    motor_mult_range: tuple[float, float] = (0.8, 1.2)
    range_noise_std_range: tuple[float, float] = (0.0, 0.03)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RandomizationConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", False)),
            motor_mult_range=tuple(d.get("motor_mult_range", (0.8, 1.2))),
            range_noise_std_range=tuple(d.get("range_noise_std_range", (0.0, 0.03))),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "motor_mult_range": list(self.motor_mult_range),
            "range_noise_std_range": list(self.range_noise_std_range),
        }


def sample_domain(cfg: RandomizationConfig, rng: np.random.Generator) -> DomainParams:
    """Independent left/right multipliers and noise level, drawn from the *domain* RNG stream."""
    if not cfg.enabled:
        return DomainParams()
    lo, hi = cfg.motor_mult_range
    nlo, nhi = cfg.range_noise_std_range
    return DomainParams(
        left_motor_mult=float(rng.uniform(lo, hi)),
        right_motor_mult=float(rng.uniform(lo, hi)),
        range_noise_std=float(rng.uniform(nlo, nhi)),
    )


class MotorModel:
    """Snapshot of a model's nominal actuator arrays + restore/apply."""

    def __init__(self, model: mujoco.MjModel):
        self.ids = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_MOTOR),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_MOTOR),
        )
        if min(self.ids) < 0:
            raise ValueError("model has no left/right wheel motors")
        self.gain0 = model.actuator_gainprm.copy()
        self.bias0 = model.actuator_biasprm.copy()
        self.frange0 = model.actuator_forcerange.copy()

    def restore(self, model: mujoco.MjModel) -> None:
        model.actuator_gainprm[:] = self.gain0
        model.actuator_biasprm[:] = self.bias0
        model.actuator_forcerange[:] = self.frange0

    def apply(self, model: mujoco.MjModel, params: DomainParams) -> None:
        self.restore(model)
        for aid, m in zip(self.ids, (params.left_motor_mult, params.right_motor_mult)):
            if not (m > 0 and np.isfinite(m)):
                raise ValueError(f"motor multiplier must be positive and finite, got {m}")
            model.actuator_gainprm[aid, 0] = self.gain0[aid, 0] * m
            model.actuator_biasprm[aid, 2] = self.bias0[aid, 2] * m
            model.actuator_forcerange[aid] = self.frange0[aid] * m
