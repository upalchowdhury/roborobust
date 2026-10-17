"""Config loading, canonical hashing and typed environment config.

Configs are YAML. A config may declare ``base: other.yaml`` (path relative to the file) and is
deep-merged over it, so the nominal and randomized training configs can share one PPO block and
differ only where they intend to.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
ROBOT_XML = PACKAGE_DIR / "robot.xml"
REPO_ROOT = PACKAGE_DIR.parent.parent


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    """Resolve a config-relative path: absolute stays, else try cwd, then repo root."""
    p = Path(p)
    if p.is_absolute():
        return p
    for root in ([base] if base else []) + [Path.cwd(), REPO_ROOT]:
        if root is not None and (root / p).exists():
            return (root / p).resolve()
    return (Path.cwd() / p).resolve()


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    path = resolve_path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_yaml(path.parent / cfg.pop("base"))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode())[:16]


# --------------------------------------------------------------------------------------
# Typed environment config
# --------------------------------------------------------------------------------------
@dataclass
class RobotCfg:
    body_radius: float = 0.18
    total_mass: float = 5.1
    wheel_radius: float = 0.05
    track_width: float = 0.30
    wheel_damping: float = 0.04
    wheel_armature: float = 0.001
    wheel_friction: float = 1.0
    caster_friction: float = 0.01
    motor_kv: float = 0.4
    motor_force_limit: float = 0.5
    max_wheel_speed: float = 10.0


@dataclass
class SensingCfg:
    n_rays: int = 24
    max_range: float = 3.0
    ray_height: float = 0.10
    v_bound: float = 0.5
    w_bound: float = 3.5


@dataclass
class VerifierCfg:
    goal_radius: float = 0.30
    max_success_speed: float = 0.10
    dwell_steps: int = 5
    max_tilt_deg: float = 45.0
    arena_half_extent: float = 3.0


@dataclass
class RewardCfg:
    success_bonus: float = 10.0
    failure_penalty: float = 10.0  # magnitude; applied as a negative reward
    step_cost: float = 0.01
    shaping_alpha: float = 50.0  # see configs/env.yaml and docs/dev_log.md
    gamma: float = 0.999  # see configs/env.yaml and docs/dev_log.md


@dataclass
class EnvConfig:
    env_version: str = "roborobust-env-1.0"
    timestep: float = 0.005
    substeps: int = 10
    max_episode_steps: int = 600
    room_half_size: float = 3.0
    robot: RobotCfg = field(default_factory=RobotCfg)
    sensing: SensingCfg = field(default_factory=SensingCfg)
    verifier: VerifierCfg = field(default_factory=VerifierCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps

    @property
    def room_diagonal(self) -> float:
        return float((2 * self.room_half_size) * 2 ** 0.5)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        d = copy.deepcopy(d)
        sub = {
            "robot": RobotCfg,
            "sensing": SensingCfg,
            "verifier": VerifierCfg,
            "reward": RewardCfg,
        }
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k in sub:
                kwargs[k] = sub[k](**v)
            elif k == "physics":
                kwargs["timestep"] = v["timestep"]
                kwargs["substeps"] = v["substeps"]
            elif k == "room":
                kwargs["room_half_size"] = v["half_size"]
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def load_env_config(path: str | Path = "configs/env.yaml") -> tuple[EnvConfig, dict]:
    raw = load_yaml(path)
    return EnvConfig.from_dict(raw), raw
