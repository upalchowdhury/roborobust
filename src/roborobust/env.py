"""Gymnasium environment: differential-drive robot in a 6 m x 6 m MuJoCo room.

Lifecycle summary
  reset(seed, options)  -> picks scenario / domain params / noise stream, builds or reuses the
                           MuJoCo model for that layout, restores nominal motor arrays, applies the
                           domain multipliers, places robot + goal, zeroes every episode counter.
  step(action)          -> validates + clips the action, runs `substeps` physics steps checking
                           contacts after EVERY physics step (stops on the first robot<->wall/obstacle
                           contact), re-checks contacts at the final pose, asks the independent
                           verifier for the outcome, computes the training reward, builds the
                           (possibly noisy) observation.

Randomness uses three independent streams spawned from one SeedSequence: geometry (which scenario),
domain (motor/noise parameters) and measurement noise. Nominal and randomized training runs with the
same seed therefore see the *same scenario sequence*; only the domain stream differs in use.

Privileged information (true state, sampled domain parameters, scenario metadata) is exposed only
through `info`, never through the observation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from roborobust.config import ROBOT_XML, EnvConfig, config_hash, sha256_file
from roborobust.randomization import DomainParams, MotorModel, RandomizationConfig, sample_domain
from roborobust.rewards import RewardFn
from roborobust.scenarios import Obstacle, Scenario, load_manifest
from roborobust.sensors import RangeSensor, add_range_noise, build_observation, wrap_angle
from roborobust.verifier import SUCCESS, TIMEOUT, Verifier

WALL_HEIGHT = 0.30
OBSTACLE_RGBA = (0.55, 0.30, 0.25, 1.0)
REST_HEIGHT = 0.05  # chassis body origin height at rest = wheel radius


class SimulationError(RuntimeError):
    """Non-finite / unstable simulation. Never swallowed: it invalidates the affected experiment."""


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------
_ROBOT_XML_TEXT: str | None = None


def build_model(obstacles: list[Obstacle]) -> mujoco.MjModel:
    """Compile robot.xml plus this layout's obstacle boxes (static world geoms, group 0)."""
    global _ROBOT_XML_TEXT
    if _ROBOT_XML_TEXT is None:
        _ROBOT_XML_TEXT = Path(ROBOT_XML).read_text()
    spec = mujoco.MjSpec.from_string(_ROBOT_XML_TEXT)
    for k, o in enumerate(obstacles):
        spec.worldbody.add_geom(
            name=f"obs_{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[o.cx, o.cy, WALL_HEIGHT / 2],
            size=[o.hx, o.hy, WALL_HEIGHT / 2],
            rgba=list(OBSTACLE_RGBA),
            group=0,
        )
    return spec.compile()


def check_robot_config(model: mujoco.MjModel, cfg: EnvConfig) -> list[str]:
    """Compare the documented config values with the compiled model. Returns mismatches."""
    r, bad = cfg.robot, []

    def chk(name, got, want, tol=1e-6):
        if not math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=tol):
            bad.append(f"{name}: model={got} config={want}")

    nm = lambda t, n: mujoco.mj_name2id(model, t, n)  # noqa: E731
    O = mujoco.mjtObj
    chk("timestep", model.opt.timestep, cfg.timestep)
    chk("total_mass", model.body_subtreemass[nm(O.mjOBJ_BODY, "chassis")], r.total_mass)
    chk("body_radius", model.geom_size[nm(O.mjOBJ_GEOM, "chassis_geom")][0], r.body_radius)
    chk("wheel_radius", model.geom_size[nm(O.mjOBJ_GEOM, "left_wheel_geom")][0], r.wheel_radius)
    lw, rw = nm(O.mjOBJ_BODY, "left_wheel"), nm(O.mjOBJ_BODY, "right_wheel")
    chk("track_width", model.body_pos[lw][1] - model.body_pos[rw][1], r.track_width)
    for side in ("left", "right"):
        j = nm(O.mjOBJ_JOINT, f"{side}_wheel_joint")
        dof = model.jnt_dofadr[j]
        chk(f"{side} wheel_damping", model.dof_damping[dof], r.wheel_damping)
        chk(f"{side} wheel_armature", model.dof_armature[dof], r.wheel_armature)
        chk(f"{side} wheel_friction", model.geom_friction[nm(O.mjOBJ_GEOM, f"{side}_wheel_geom")][0], r.wheel_friction)
        a = nm(O.mjOBJ_ACTUATOR, f"{side}_wheel_motor")
        chk(f"{side} motor_kv", model.actuator_gainprm[a][0], r.motor_kv)
        chk(f"{side} motor_force_limit", model.actuator_forcerange[a][1], r.motor_force_limit)
        chk(f"{side} max_wheel_speed", model.actuator_ctrlrange[a][1], r.max_wheel_speed)
    for c in ("front", "rear"):
        chk(f"caster_{c} friction", model.geom_friction[nm(O.mjOBJ_GEOM, f"caster_{c}")][0], r.caster_friction)
    chk("room_half_size", model.geom_pos[nm(O.mjOBJ_GEOM, "wall_e")][0] - model.geom_size[nm(O.mjOBJ_GEOM, "wall_e")][0],
        cfg.room_half_size)
    return bad


class _ModelBundle:
    """A compiled model plus everything derived from it (so it can be cached per layout)."""

    def __init__(self, obstacles: list[Obstacle], cfg: EnvConfig):
        self.model = build_model(obstacles)
        mismatches = check_robot_config(self.model, cfg)
        if mismatches:
            raise ValueError("robot.xml and env config disagree: " + "; ".join(mismatches))
        self.data = mujoco.MjData(self.model)
        self.motors = MotorModel(self.model)
        m = self.model
        self.chassis_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.goal_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")])
        root = m.body_rootid[m.geom_bodyid]
        self.is_robot = root == self.chassis_body
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.is_hazard = np.array([n.startswith(("wall_", "obs_")) for n in names])
        self.geom_names = names
        self.sensor = RangeSensor(cfg.sensing, self.chassis_body)
        self.wheel_qvel = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wheel_joint")] for s in ("left", "right")]
        )


class RoboRobustEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_cfg: EnvConfig | dict | None = None,
        scenarios: list[Scenario] | str | Path | None = None,
        randomization: RandomizationConfig | dict | None = None,
        render_mode: str | None = None,
        record: bool = False,
    ):
        super().__init__()
        if env_cfg is None:
            from roborobust.config import load_env_config

            env_cfg = load_env_config()[0]
        self.cfg = env_cfg if isinstance(env_cfg, EnvConfig) else EnvConfig.from_dict(env_cfg)
        self.cfg_hash = config_hash(self.cfg.to_dict()) + "-" + sha256_file(ROBOT_XML)[:8]
        self.randomization = (
            randomization if isinstance(randomization, RandomizationConfig) else RandomizationConfig.from_dict(randomization)
        )
        if isinstance(scenarios, (str, Path)):
            scenarios = load_manifest(scenarios)
        self.scenarios: list[Scenario] = list(scenarios or [])
        self._scenario_by_id = {s.scenario_id: s for s in self.scenarios}
        self.render_mode = render_mode
        self.record = record

        c = self.cfg.sensing
        lo = np.array([0.0] * c.n_rays + [0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        hi = np.ones(c.n_rays + 5, dtype=np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._bundles: dict[str, _ModelBundle] = {}
        self._b: _ModelBundle | None = None
        self.verifier = Verifier(self.cfg.verifier)
        self.reward_fn = RewardFn(self.cfg.reward, self.cfg.room_diagonal)
        self._streams_ready = False
        self._renderer = None
        self._renderer_model = None
        self._done = True  # must reset() before stepping
        self.scenario: Scenario | None = None
        self.domain = DomainParams()

    # ------------------------------------------------------------------ seeding
    def _seed_streams(self, seed: int | None) -> None:
        ss = np.random.SeedSequence(seed)
        g, d, n = ss.spawn(3)
        self._rng_geometry = np.random.default_rng(g)
        self._rng_domain = np.random.default_rng(d)
        self._rng_noise = np.random.default_rng(n)
        self._streams_ready = True

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None or not self._streams_ready:
            self._seed_streams(seed)

        # --- scenario (geometry stream) ---
        sc = options.get("scenario")
        if sc is None:
            if not self.scenarios:
                raise ValueError("no scenarios available: pass a manifest or options['scenario']")
            sc = self.scenarios[int(self._rng_geometry.integers(len(self.scenarios)))]
        elif isinstance(sc, str):
            sc = self._scenario_by_id[sc]
        elif isinstance(sc, dict):
            sc = Scenario.from_dict(sc)
        self.scenario = sc

        # --- domain parameters (domain stream) ---
        if options.get("domain") is not None:
            self.domain = DomainParams.from_any(options["domain"])
        else:
            self.domain = sample_domain(self.randomization, self._rng_domain)

        # --- measurement-noise stream (pinned for paired evaluation) ---
        if options.get("noise_seed") is not None:
            self._rng_noise = np.random.default_rng(np.random.SeedSequence(int(options["noise_seed"])))

        # --- model: cached per layout; nominal motor arrays restored THEN multipliers applied ---
        key = sc.layout_id
        b = self._bundles.get(key)
        if b is None:
            b = _ModelBundle(sc.obstacles, self.cfg)
            self._bundles[key] = b
        self._b = b
        b.motors.apply(b.model, self.domain)
        m, d = b.model, b.data
        mujoco.mj_resetData(m, d)
        yaw = sc.start["yaw"]
        d.qpos[0:2] = (sc.start["x"], sc.start["y"])
        d.qpos[2] = REST_HEIGHT
        d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.mocap_pos[b.goal_mocap] = (sc.goal["x"], sc.goal["y"], 0.003)
        mujoco.mj_forward(m, d)
        self._check_sim_health()

        # --- episode counters (everything episodic is reset here) ---
        self.verifier.reset()
        self._step_count = 0
        self._done = False
        self._last_action = np.zeros(2, dtype=np.float32)
        self._clip_count = 0
        self._ret = 0.0
        self._comp_sum = {"step_cost": 0.0, "shaping": 0.0, "success_bonus": 0.0, "failure_penalty": 0.0, "total": 0.0}
        self._sat = np.zeros(2)
        self._sat_n = 0
        self._straight_n = 0
        self._straight_wz = 0.0
        self._yaw_travel = 0.0
        self._path_len = 0.0
        self._min_range = math.inf
        self._first_goal_entry_t = None
        self._collision_event = None
        x, y, yaw0, _ = self._pose()
        self._prev_xy, self._prev_yaw = (x, y), yaw0
        self._prev_dist = self._goal_distance(x, y)
        obs = self._observe()
        if self.record:
            self._rec = {k: [] for k in ("t", "qpos", "qvel", "ctrl", "obs", "action_raw", "action", "reward")}
            self._rec_events: list[dict] = []
            self._record_frame(obs, np.zeros(2), np.zeros(2), np.zeros(5))
        return obs, self._reset_info()

    def _reset_info(self) -> dict:
        sc = self.scenario
        return {
            "scenario_id": sc.scenario_id,
            "layout_id": sc.layout_id,
            "family": sc.family,
            "domain": self.domain.to_dict(),  # privileged: logs/evaluator only
            "goal_distance": self._prev_dist,
        }

    # ------------------------------------------------------------------ state accessors (true state)
    def _pose(self) -> tuple[float, float, float, float]:
        q = self._b.data.qpos
        w, qx, qy, qz = q[3], q[4], q[5], q[6]
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        tilt = math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * (qx * qx + qy * qy))))
        return float(q[0]), float(q[1]), yaw, tilt

    def _goal_distance(self, x: float, y: float) -> float:
        g = self.scenario.goal
        return math.hypot(g["x"] - x, g["y"] - y)

    def _velocities(self, yaw: float) -> tuple[float, float, float]:
        """(planar speed, body-frame forward speed, yaw rate about world z)."""
        d = self._b.data
        vx, vy = d.qvel[0], d.qvel[1]
        q = d.qpos
        w, qx, qy, qz = q[3], q[4], q[5], q[6]
        wb = d.qvel[3:6]  # free-joint angular velocity is expressed in the body frame
        wz_world = 2 * (qx * qz - w * qy) * wb[0] + 2 * (qy * qz + w * qx) * wb[1] + (1 - 2 * (qx * qx + qy * qy)) * wb[2]
        return (
            float(math.hypot(vx, vy)),
            float(vx * math.cos(yaw) + vy * math.sin(yaw)),
            float(wz_world),
        )

    # ------------------------------------------------------------------ observation
    def _observe(self) -> np.ndarray:
        b = self._b
        x, y, yaw, _ = self._pose()
        true_ranges = b.sensor.ranges(b.model, b.data, x, y, yaw)
        self._min_range = min(self._min_range, float(true_ranges.min()))
        self._last_true_ranges = true_ranges
        noisy = add_range_noise(true_ranges, self.domain.range_noise_std, self._rng_noise, self.cfg.sensing.max_range)
        g = self.scenario.goal
        bearing = wrap_angle(math.atan2(g["y"] - y, g["x"] - x) - yaw)
        _, v_fwd, wz = self._velocities(yaw)
        return build_observation(noisy, self._goal_distance(x, y), bearing, v_fwd, wz, self.cfg.sensing, self.cfg.room_diagonal)

    # ------------------------------------------------------------------ contacts
    def _robot_hazard_contact(self) -> dict | None:
        """First contact between any robot geom and any wall/obstacle geom in the current contact
        list, else None. Floor and wheel/caster-floor contacts are not hazards by construction."""
        b, d = self._b, self._b.data
        if d.ncon == 0:
            return None
        g1, g2 = d.contact.geom1, d.contact.geom2
        hit = (b.is_robot[g1] & b.is_hazard[g2]) | (b.is_hazard[g1] & b.is_robot[g2])
        if not hit.any():
            return None
        k = int(np.argmax(hit))
        a, c = int(g1[k]), int(g2[k])
        robot_g, haz_g = (a, c) if b.is_robot[a] else (c, a)
        return {
            "sim_time": float(d.time),
            "robot_geom": b.geom_names[robot_g],
            "hazard_geom": b.geom_names[haz_g],
            "contact_pos": [float(v) for v in d.contact.pos[k]],
            "penetration": float(-d.contact.dist[k]),
        }

    def _check_sim_health(self) -> None:
        d = self._b.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise SimulationError(f"non-finite simulator state at t={d.time:.3f} in {self.scenario.scenario_id}")
        for i in range(len(d.warning)):
            if d.warning[i].number > 0:
                raise SimulationError(
                    f"MuJoCo warning {mujoco.mjtWarning(i).name} (x{d.warning[i].number}) in {self.scenario.scenario_id}"
                )

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite action rejected: {a}")
        clipped_flag = bool(np.any(np.abs(a) > 1.0))
        if clipped_flag:
            self._clip_count += 1
        a_clip = np.clip(a, -1.0, 1.0)

        b = self._b
        m, d = b.model, b.data
        cmd = a_clip * self.cfg.robot.max_wheel_speed  # rad/s wheel velocity targets
        d.ctrl[0], d.ctrl[1] = cmd[0], cmd[1]

        collision = None
        frange = m.actuator_forcerange[:, 1]
        for sub in range(self.cfg.substeps):
            mujoco.mj_step(m, d)
            self._sat += np.abs(d.actuator_force) >= 0.999 * frange
            self._sat_n += 1
            collision = self._robot_hazard_contact()
            if collision is not None:
                collision["substep"] = sub + 1
                break  # stop immediately on first contact: no contact is lost between control boundaries
        if collision is None:
            # Contacts found inside mj_step belong to the *pre-integration* pose of each substep, so
            # re-run collision at the final pose before the verifier looks at this control boundary.
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            collision = self._robot_hazard_contact()
            if collision is not None:
                collision["substep"] = self.cfg.substeps + 0.5  # detected at the boundary pose
        self._check_sim_health()
        if collision is not None and self._collision_event is None:
            self._collision_event = collision

        # --- true state -> verifier ---
        x, y, yaw, tilt = self._pose()
        speed, v_fwd, wz = self._velocities(yaw)
        dist = self._goal_distance(x, y)
        outcome = self.verifier.update(
            goal_distance=dist, planar_speed=speed, tilt_rad=tilt, x=x, y=y, collided=collision is not None
        )
        self._step_count += 1
        if self._first_goal_entry_t is None and dist <= self.cfg.verifier.goal_radius:
            self._first_goal_entry_t = float(d.time)

        # --- reward (training signal only) ---
        rc = self.reward_fn.compute(self._prev_dist, dist, outcome)
        self._ret += rc.total
        for k, v in rc.to_dict().items():
            self._comp_sum[k] += v

        # --- bookkeeping for diagnostics ---
        self._path_len += math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        self._yaw_travel += abs(wrap_angle(yaw - self._prev_yaw))
        self._prev_xy, self._prev_yaw, self._prev_dist = (x, y), yaw, dist
        if abs(a_clip[0] - a_clip[1]) < 0.1 and 0.5 * (a_clip[0] + a_clip[1]) > 0.3:
            self._straight_n += 1
            self._straight_wz += wz
        self._last_action = a_clip.astype(np.float32)

        terminated = outcome.terminated
        truncated = (not terminated) and self._step_count >= self.cfg.max_episode_steps
        reason = outcome.reason if terminated else (TIMEOUT if truncated else None)
        self._done = terminated or truncated

        obs = self._observe()
        info: dict[str, Any] = {
            "terminal_reason": reason,
            "success": outcome.success,
            "goal_distance": dist,
            "reward_components": rc.to_dict(),
            "action_clipped": clipped_flag,
            "wheel_cmd": cmd.tolist(),
            "wheel_speed": [float(d.qvel[i]) for i in b.wheel_qvel],
            "actuator_torque": d.actuator_force.tolist(),
            "collision_event": collision,
        }
        if self.record:
            self._record_frame(obs, a, a_clip, np.array([rc.total, rc.step_cost, rc.shaping, rc.success_bonus, rc.failure_penalty]))
            if collision is not None or reason is not None:
                self._rec_events.append({"step": self._step_count, "sim_time": float(d.time), "reason": reason, "collision": collision})
        if self._done:
            info["episode_summary"] = self._episode_summary(reason, truncated, x, y, speed, wz)
        return obs, float(rc.total), terminated, truncated, info

    # ------------------------------------------------------------------ summaries / recording
    def _episode_summary(self, reason, truncated, x, y, speed, wz) -> dict:
        d = self._b.data
        sc = self.scenario
        lr = self._last_true_ranges
        n = len(lr)
        left_min = float(lr[n // 8 : 3 * n // 8 + 1].min())  # rays between 45 and 135 deg (left side)
        right_min = float(lr[5 * n // 8 : 7 * n // 8 + 1].min())  # rays between 225 and 315 deg
        success = reason == SUCCESS
        return {
            "scenario_id": sc.scenario_id,
            "layout_id": sc.layout_id,
            "family": sc.family,
            "split": sc.split,
            "terminal_reason": reason,
            "success": success,
            "collision": reason == "collision",
            "terminated": not truncated,
            "truncated": bool(truncated),
            "steps": self._step_count,
            "duration_s": float(d.time),
            "time_to_goal_s": float(d.time) if success else None,
            "first_goal_entry_s": self._first_goal_entry_t,
            "path_length_m": self._path_len,
            "return": self._ret,
            "components": dict(self._comp_sum),
            "domain": self.domain.to_dict(),
            "clip_count": self._clip_count,
            "motor_saturation_frac": (self._sat / max(self._sat_n, 1)).tolist(),
            "collision_event": self._collision_event,
            "features": {
                "final_goal_distance": self._prev_dist,
                "final_speed": speed,
                "final_yaw_rate": wz,
                "yaw_travel_rad": self._yaw_travel,
                "min_range_m": self._min_range,
                "left_clearance_end_m": left_min,
                "right_clearance_end_m": right_min,
                "straight_cmd_steps": self._straight_n,
                "straight_cmd_mean_yaw_rate": self._straight_wz / self._straight_n if self._straight_n else None,
                "start_goal_distance": sc.meta.get("start_goal_distance"),
                "straight_line_blocked": sc.meta.get("straight_line_blocked"),
                "grid_path_length": sc.meta.get("grid_path_length"),
            },
        }

    def _record_frame(self, obs, a_raw, a_clip, rew):
        """Append the simulator state at a control boundary plus the observation the agent received."""
        d = self._b.data
        r = self._rec
        r["t"].append(d.time)
        r["qpos"].append(d.qpos.copy())
        r["qvel"].append(d.qvel.copy())
        r["ctrl"].append(d.ctrl.copy())
        r["obs"].append(np.asarray(obs, dtype=np.float32).copy())
        r["action_raw"].append(np.asarray(a_raw, dtype=np.float64))
        r["action"].append(np.asarray(a_clip, dtype=np.float64))
        r["reward"].append(np.asarray(rew, dtype=np.float64))

    def get_trajectory(self) -> dict:
        """Recorded episode (requires record=True): simulator integration state at every control
        boundary (qpos, qvel, ctrl, time), the observations the agent received, raw+clipped actions, reward
        components and terminal/collision events, plus the scenario and domain parameters."""
        if not self.record:
            raise RuntimeError("env was created with record=False")
        out = {k: np.asarray(v) for k, v in self._rec.items()}
        out["scenario"] = self.scenario.to_dict()
        out["domain"] = self.domain.to_dict()
        out["events"] = list(self._rec_events)
        out["cfg_hash"] = self.cfg_hash
        out["env_config"] = self.cfg.to_dict()  # full model/task configuration, not just its hash
        out["reward_columns"] = ["total", "step_cost", "shaping", "success_bonus", "failure_penalty"]
        return out

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "rgb_array" or self._b is None:
            return None
        if self._renderer is None or self._renderer_model is not self._b.model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self._b.model, height=480, width=640)
            self._renderer_model = self._b.model
        self._renderer.update_scene(self._b.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
