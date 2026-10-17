"""Physics + sensor correctness checks (Gate A and part of Gate B).

Each check returns a dict ``{"name", "passed", "measured", "criterion"}``. They are used by both
``roborobust validate-env`` (which writes the measurements to JSON so the numbers are on record) and
the pytest suite, so the CLI and the tests cannot disagree.

Open-loop checks drive the MuJoCo model directly (no episode termination), so that e.g. the wall can
be shown to physically block the robot rather than merely end the episode.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from roborobust.config import EnvConfig, load_env_config
from roborobust.env import REST_HEIGHT, _ModelBundle
from roborobust.randomization import DomainParams
from roborobust.scenarios import Obstacle
from roborobust.sensors import wrap_angle


def _result(name, passed, measured, criterion):
    return {"name": name, "passed": bool(passed), "measured": measured, "criterion": criterion}


class OpenLoopSim:
    """Direct MuJoCo stepping on an (optionally obstacle-containing) model."""

    def __init__(self, cfg: EnvConfig, obstacles: list[Obstacle] | None = None):
        self.cfg = cfg
        self.b = _ModelBundle(obstacles or [], cfg)
        self.m, self.d = self.b.model, self.b.data

    def reset(self, x=0.0, y=0.0, yaw=0.0, domain: DomainParams | None = None):
        self.b.motors.apply(self.m, domain or DomainParams())
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:2] = (x, y)
        self.d.qpos[2] = REST_HEIGHT
        self.d.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        mujoco.mj_forward(self.m, self.d)

    def run(self, action, seconds: float, record_every: int = 1):
        """Hold a constant action in [-1,1]^2 for `seconds`; returns array of [t, x, y, yaw, vx_body, wz]."""
        ctrl = np.clip(np.asarray(action, float), -1, 1) * self.cfg.robot.max_wheel_speed
        self.d.ctrl[:] = ctrl
        log = []
        n = int(round(seconds / self.m.opt.timestep))
        for k in range(n):
            mujoco.mj_step(self.m, self.d)
            if k % record_every == 0:
                log.append(self.state())
        return np.array(log)

    def state(self):
        q = self.d.qpos
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        vx = self.d.qvel[0] * math.cos(yaw) + self.d.qvel[1] * math.sin(yaw)
        return [self.d.time, q[0], q[1], yaw, vx, self.d.qvel[5]]

    def tilt(self):
        q = self.d.qpos
        return math.acos(min(1.0, max(-1.0, 1 - 2 * (q[4] ** 2 + q[5] ** 2))))


# ------------------------------------------------------------------------------ Gate A
def check_rest_stability(cfg: EnvConfig) -> dict:
    sim = OpenLoopSim(cfg)
    sim.reset(0.5, -0.5, 0.7)
    log = sim.run([0, 0], 5.0, record_every=20)
    drift = float(np.hypot(log[-1, 1] - 0.5, log[-1, 2] + 0.5))
    z_err = abs(sim.d.qpos[2] - REST_HEIGHT)
    tilt = math.degrees(sim.tilt())
    ok = drift < 1e-3 and z_err < 1e-3 and tilt < 0.5
    return _result("rest_stability", ok, {"drift_m": drift, "z_err_m": z_err, "tilt_deg": tilt},
                   "5 s at rest: drift < 1 mm, height error < 1 mm, tilt < 0.5 deg")


def check_straight_motion(cfg: EnvConfig) -> dict:
    sim = OpenLoopSim(cfg)
    sim.reset(-1.0, 0.0, 0.0)
    log = sim.run([1, 1], 3.0, record_every=20)
    dist = log[-1, 1] + 1.0
    lateral = abs(log[-1, 2])
    heading = abs(log[-1, 3])
    v_end = log[-1, 4]
    ideal = cfg.robot.max_wheel_speed * cfg.robot.wheel_radius
    ok = dist > 0.8 * ideal * 3.0 * 0.8 and lateral < 0.02 * dist and heading < 0.03
    return _result("straight_motion", ok,
                   {"distance_m": dist, "lateral_m": lateral, "heading_drift_rad": heading,
                    "steady_speed_mps": v_end, "ideal_speed_mps": ideal},
                   "equal wheel commands: lateral < 2% of distance, heading drift < 0.03 rad, moves forward")


def check_rotation_sign(cfg: EnvConfig) -> dict:
    out = {}
    ok = True
    for name, act, sign in (("left_back_right_fwd_ccw", [-1, 1], +1), ("left_fwd_right_back_cw", [1, -1], -1)):
        sim = OpenLoopSim(cfg)
        sim.reset(0, 0, 0)
        log = sim.run(act, 1.0, record_every=20)
        dyaw = wrap_angle(log[-1, 3])
        pos_drift = float(np.hypot(log[-1, 1], log[-1, 2]))
        out[name] = {"yaw_rad": dyaw, "yaw_rate_end": log[-1, 5], "position_drift_m": pos_drift}
        ok &= sign * log[-1, 5] > 1.0 and pos_drift < 0.05
    return _result("rotation_sign", ok, out,
                   "opposite wheel commands rotate in the expected direction (+ = CCW), < 5 cm translation")


def check_bounded_action_stability(cfg: EnvConfig, seed: int = 0, seconds: float = 30.0) -> dict:
    rng = np.random.default_rng(seed)
    sim = OpenLoopSim(cfg)
    sim.reset(0, 0, 0)
    max_tilt, max_z_err, ok = 0.0, 0.0, True
    n_ctrl = int(seconds / cfg.control_dt)
    for _ in range(n_ctrl):
        sim.d.ctrl[:] = rng.uniform(-1, 1, 2) * cfg.robot.max_wheel_speed
        for _ in range(cfg.substeps):
            mujoco.mj_step(sim.m, sim.d)
        # keep the robot inside the room: bounce the pose back if it nears a wall (physics unaffected)
        if max(abs(sim.d.qpos[0]), abs(sim.d.qpos[1])) > 2.4:
            sim.d.qpos[0:2] = 0.0
            mujoco.mj_forward(sim.m, sim.d)
        ok &= bool(np.isfinite(sim.d.qpos).all())
        max_tilt = max(max_tilt, math.degrees(sim.tilt()))
        max_z_err = max(max_z_err, abs(sim.d.qpos[2] - REST_HEIGHT))
    ok &= max_tilt < 5.0 and max_z_err < 0.01
    return _result("bounded_action_stability", ok, {"max_tilt_deg": max_tilt, "max_height_err_m": max_z_err, "seconds": seconds},
                   "30 s of random bounded commands: finite state, tilt < 5 deg, height within 1 cm")


def check_wall_blocks(cfg: EnvConfig) -> dict:
    sim = OpenLoopSim(cfg)
    sim.reset(1.5, 0.0, 0.0)
    log = sim.run([1, 1], 8.0, record_every=20)  # would travel ~3.6 m if unobstructed
    max_x = float(log[:, 1].max())
    limit = cfg.room_half_size - cfg.robot.body_radius
    ok = max_x <= limit + 0.01 and max_x > limit - 0.05  # reached the wall and was stopped by it
    # same test against an interior obstacle
    sim2 = OpenLoopSim(cfg, [Obstacle(0.5, 0.0, 0.3, 0.5)])
    sim2.reset(-1.0, 0.0, 0.0)
    log2 = sim2.run([1, 1], 6.0, record_every=20)
    face = 0.5 - 0.3 - cfg.robot.body_radius
    max_x2 = float(log2[:, 1].max())
    ok &= max_x2 <= face + 0.01 and max_x2 > face - 0.05
    return _result("wall_blocks_passage", ok,
                   {"wall_max_x": max_x, "wall_limit": limit, "obstacle_max_x": max_x2, "obstacle_limit": face},
                   "robot driven into wall / obstacle stops at contact (centre within 1 cm of face - radius)")


def check_floor_contacts_not_collisions(cfg: EnvConfig) -> dict:
    from roborobust.env import RoboRobustEnv
    from roborobust.scenarios import Scenario

    sc = Scenario("v_empty", "val", "empty", "empty", {"x": 0.0, "y": 0.0, "yaw": 0.0}, {"x": 2.0, "y": 2.0}, [])
    env = RoboRobustEnv(cfg, [sc])
    env.reset(seed=0, options={"scenario": sc})
    n_floor_contact_steps, false_collisions = 0, 0
    for k in range(60):
        _, _, te, tr, info = env.step(np.array([0.5, 0.3]))
        n_floor_contact_steps += int(env._b.data.ncon > 0)
        false_collisions += int(info["collision_event"] is not None)
        if te or tr:
            break
    ok = n_floor_contact_steps > 0 and false_collisions == 0
    return _result("floor_contacts_are_not_collisions", ok,
                   {"steps_with_contacts": n_floor_contact_steps, "false_collisions": false_collisions},
                   "wheel/caster-floor contacts exist every step yet never count as collisions")


def check_weak_motor_effect(cfg: EnvConfig) -> dict:
    """Weakened left motor must change the measured trajectory under a demanding command, and the
    mechanism (torque-limit saturation) must be visible. Also records the (small) effect of a light
    command so the report can state where the intervention bites."""
    res = {}
    for label, act in (("demanding_full_command", [1.0, 1.0]), ("light_half_command", [0.5, 0.5])):
        rows = {}
        for mult in (1.0, 0.85, 0.6):
            sim = OpenLoopSim(cfg)
            sim.reset(-1.0, 0.0, 0.0, DomainParams(left_motor_mult=mult))
            log = sim.run(act, 3.0, record_every=20)
            rows[str(mult)] = {
                "lateral_m": float(log[-1, 2]),
                "heading_rad": float(log[-1, 3]),
                "forward_m": float(log[-1, 1] + 1.0),
                "wheel_speed_l_rps": float(sim.d.qvel[sim.b.wheel_qvel[0]]),
                "wheel_speed_r_rps": float(sim.d.qvel[sim.b.wheel_qvel[1]]),
                "left_torque_Nm": float(sim.d.actuator_force[0]),
                "left_torque_limit_Nm": float(sim.m.actuator_forcerange[0, 1]),
            }
        res[label] = rows
    demanding = res["demanding_full_command"]
    effect = abs(demanding["0.6"]["lateral_m"] - demanding["1.0"]["lateral_m"])
    saturated = demanding["0.6"]["left_torque_Nm"] >= 0.99 * demanding["0.6"]["left_torque_limit_Nm"]
    ok = effect > 0.2 and saturated and demanding["0.6"]["wheel_speed_l_rps"] < 0.9 * demanding["1.0"]["wheel_speed_l_rps"]
    # the multiplier must restore cleanly
    sim = OpenLoopSim(cfg)
    base = sim.m.actuator_forcerange.copy()
    sim.b.motors.apply(sim.m, DomainParams(0.6, 0.6))
    sim.b.motors.apply(sim.m, DomainParams(1.0, 1.0))
    restored = bool(np.allclose(sim.m.actuator_forcerange, base))
    res["restored_after_reapply"] = restored
    res["lateral_effect_demanding_m"] = effect
    return _result("weak_motor_changes_trajectory", ok and restored, res,
                   "weak left motor (0.6) under full command: >0.2 m extra lateral deviation in 3 s, torque limit binding, "
                   "left wheel speed <90% of nominal; multipliers restore")


# ------------------------------------------------------------------------------ sensors
def check_known_raycasts(cfg: EnvConfig) -> dict:
    """Analytic ranges in simple scenes + robot/goal-marker exclusion."""
    c = cfg.sensing
    R = c.max_range
    sim = OpenLoopSim(cfg, [Obstacle(1.5, 0.0, 0.25, 0.5)])
    cases = []
    # robot at (0.5,-0.8) facing +x in a room with an obstacle: east ray hits obstacle's west face x=1.25
    # (ray at y=-0.8 is outside the obstacle's y-extent [-0.5,0.5], so it passes below it and hits the east wall at 2.5)
    sim.reset(0.5, -0.8, 0.0)
    r = sim.b.sensor.ranges(sim.m, sim.d, 0.5, -0.8, 0.0)
    cases.append(("east wall, passing below obstacle", r[0], 2.5))
    cases.append(("south wall", r[18], 2.2))               # 270 deg -> y = -3.0
    cases.append(("north out of range -> max", r[6], R))    # 3.8 m > 3 m
    cases.append(("west out of range -> max", r[12], R))    # 3.5 m > 3 m
    # robot at (0,0) facing +x: obstacle west face at x=1.25
    sim.reset(0.0, 0.0, 0.0)
    r = sim.b.sensor.ranges(sim.m, sim.d, 0.0, 0.0, 0.0)
    cases.append(("obstacle west face ahead", r[0], 1.25))
    # heading rotates the fan: yaw=+90deg -> ray 0 points north (y=3) 3.0 away (== max range, no hit by cutoff)
    sim.reset(0.0, 1.0, math.pi / 2)
    r = sim.b.sensor.ranges(sim.m, sim.d, 0.0, 1.0, math.pi / 2)
    cases.append(("yawed 90deg: ray0 -> north wall", r[0], 2.0))
    cases.append(("yawed 90deg: ray 6 (+90deg) -> west wall (x=-3)", r[6], 3.0))
    # diagonal ray to an obstacle corner-free face: from (0,0) angle 45deg hits face x=1.25 at y=1.25 (> 0.5 half-height)
    # -> misses the obstacle (y extent 0.5) and continues to the east wall; hypot(3,3)>3 so reads max.
    # exclusion: robot starts inside its own chassis cylinder; unmasked ray would read ~0.18 (verified below)
    sim.reset(0.0, 0.0, 0.0)
    gid = np.zeros(1, dtype=np.int32)
    dist = np.zeros(1)
    mujoco.mj_multiRay(sim.m, sim.d, np.array([0, 0, c.ray_height]), np.array([1.0, 0, 0]),
                       None, 1, -1, gid, dist, None, 1, R)  # no group mask, no body exclusion
    unmasked_hits_robot = bool(abs(dist[0] - cfg.robot.body_radius) < 1e-3)
    # goal marker placed directly on the ray must not be seen
    sim.d.mocap_pos[sim.b.goal_mocap] = (0.8, 0.0, 0.003)
    mujoco.mj_forward(sim.m, sim.d)
    r_goal = sim.b.sensor.ranges(sim.m, sim.d, 0.0, 0.0, 0.0)[0]
    errs = [(n, float(got), float(want), abs(got - want)) for n, got, want in cases]
    ok = all(e[3] < 1e-6 for e in errs) and unmasked_hits_robot and abs(r_goal - 1.25) < 1e-6
    return _result("known_raycasts_and_exclusions", ok,
                   {"cases": errs, "unmasked_ray_hits_robot_at_0.18": unmasked_hits_robot, "ray_through_goal_marker": float(r_goal)},
                   "ranges match analytic geometry within 1e-6 m; robot geometry and goal marker are excluded")


ALL_PHYSICS_CHECKS = [
    check_rest_stability, check_straight_motion, check_rotation_sign, check_bounded_action_stability,
    check_wall_blocks, check_floor_contacts_not_collisions, check_weak_motor_effect, check_known_raycasts,
]


def run_all(cfg: EnvConfig | None = None) -> list[dict]:
    cfg = cfg or load_env_config()[0]
    return [fn(cfg) for fn in ALL_PHYSICS_CHECKS]
