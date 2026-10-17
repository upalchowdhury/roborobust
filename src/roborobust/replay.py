"""Trajectory visualisation from saved episode logs.

A saved .npz holds the simulator integration state (qpos, qvel, ctrl, time) at every control boundary,
so viewer frames are *restored* from stored states. That is a replay of what was recorded, not a new
rollout; nothing here claims a fresh simulation reproduces it. ``--resimulate-check`` offers the
separate, explicit test: re-run the recorded actions and report how far the states diverge.

  * interactive viewer: needs ``mjpython`` on macOS (``mjpython -m roborobust replay --episode ...``)
  * ``--png`` / ``--gif``: headless (matplotlib / offscreen renderer), work under plain python
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def load_episode(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    return arrays, meta


def yaw_from_qpos(qpos: np.ndarray) -> np.ndarray:
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def plot_topdown(ax, arrays: dict, meta: dict, color: str = "#2a78d6", label: str | None = None,
                 draw_room: bool = True, ink: str = "#0b0b0b", muted: str = "#52514e") -> None:
    """Top-down view: room, obstacles, goal disc (radius = success radius), start pose, path."""
    from matplotlib.patches import Circle, Rectangle

    sc = meta["scenario"]
    if draw_room:
        ax.add_patch(Rectangle((-3, -3), 6, 6, fill=False, ec=muted, lw=1.2))
        for o in sc["obstacles"]:
            ax.add_patch(Rectangle((o["cx"] - o["hx"], o["cy"] - o["hy"]), 2 * o["hx"], 2 * o["hy"],
                                   fc="#c9c7c0", ec=muted, lw=0.8))
        ax.add_patch(Circle((sc["goal"]["x"], sc["goal"]["y"]), 0.30, fc="#bfe3cf", ec="#008300", lw=1.0, alpha=0.8))
        ax.plot(sc["start"]["x"], sc["start"]["y"], "o", ms=5, mfc="white", mec=ink, zorder=5)
        ax.annotate("", xy=(sc["start"]["x"] + 0.4 * math.cos(sc["start"]["yaw"]), sc["start"]["y"] + 0.4 * math.sin(sc["start"]["yaw"])),
                    xytext=(sc["start"]["x"], sc["start"]["y"]), arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.0), zorder=5)
    q = arrays["qpos"]
    ax.plot(q[:, 0], q[:, 1], "-", color=color, lw=2.0, label=label, zorder=4, solid_capstyle="round")
    ax.plot(q[-1, 0], q[-1, 1], "s", ms=6, color=color, mec="white", mew=1.2, zorder=6)
    ax.set_xlim(-3.15, 3.15)
    ax.set_ylim(-3.15, 3.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def check_resimulation(path: str | Path, tol: float = 1e-6) -> dict:
    """Re-run the recorded raw actions from the recorded scenario/domain/noise seed and compare states."""
    from roborobust.config import EnvConfig, load_env_config
    from roborobust.env import RoboRobustEnv
    from roborobust.randomization import DomainParams
    from roborobust.scenarios import Scenario

    arrays, meta = load_episode(path)
    rec = meta["episode_record"]
    cfg = load_env_config("configs/env.yaml")[0]
    env = RoboRobustEnv(cfg, record=True)
    env.reset(seed=rec["eval_seed"], options={"scenario": Scenario.from_dict(meta["scenario"]),
                                              "domain": DomainParams.from_any(meta["domain"]),
                                              "noise_seed": rec["noise_seed"]})
    for a in arrays["action_raw"][1:]:
        env.step(a)
    new = env.get_trajectory()
    n = min(len(new["qpos"]), len(arrays["qpos"]))
    div = np.abs(new["qpos"][:n] - arrays["qpos"][:n]).max(axis=1)
    return {"frames_recorded": len(arrays["qpos"]), "frames_resimulated": len(new["qpos"]),
            "max_abs_qpos_divergence": float(div.max()), "within_tolerance": bool(div.max() <= tol and len(new["qpos"]) == len(arrays["qpos"])),
            "tolerance": tol}


def render_gif(arrays: dict, meta: dict, out: str | Path, speed: float = 1.0, stride: int = 1) -> None:
    import mujoco
    from PIL import Image

    from roborobust.env import build_model
    from roborobust.scenarios import Scenario

    sc = Scenario.from_dict(meta["scenario"])
    model = build_model(sc.obstacles)
    data = mujoco.MjData(model)
    mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")])
    data.mocap_pos[mocap] = (sc.goal["x"], sc.goal["y"], 0.003)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0, 0, 0)
    cam.distance, cam.azimuth, cam.elevation = 9.0, 90.0, -75.0
    r = mujoco.Renderer(model, 360, 480)
    frames = []
    for i in range(0, len(arrays["qpos"]), stride):
        data.qpos[:] = arrays["qpos"][i]
        data.qvel[:] = arrays["qvel"][i]
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        frames.append(Image.fromarray(r.render()))
    r.close()
    dt = float(arrays["t"][1] - arrays["t"][0]) if len(arrays["t"]) > 1 else 0.05
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * dt * stride / speed), loop=0)


def view(arrays: dict, meta: dict, speed: float = 1.0) -> None:
    """Interactive passive viewer. On macOS this must be launched with ``mjpython``."""
    import mujoco
    import mujoco.viewer

    from roborobust.env import build_model
    from roborobust.scenarios import Scenario

    sc = Scenario.from_dict(meta["scenario"])
    model = build_model(sc.obstacles)
    data = mujoco.MjData(model)
    mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")])
    data.mocap_pos[mocap] = (sc.goal["x"], sc.goal["y"], 0.003)
    t = arrays["t"]
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 9.0, -60.0, 90.0
        while v.is_running():
            for i in range(len(t)):
                if not v.is_running():
                    break
                data.qpos[:] = arrays["qpos"][i]
                data.qvel[:] = arrays["qvel"][i]
                mujoco.mj_forward(model, data)
                v.sync()
                time.sleep(max(0.0, (t[i + 1] - t[i]) / speed) if i + 1 < len(t) else 0.5)
            time.sleep(1.0)


def replay(episode: str, speed: float = 1.0, png: str | None = None, gif: str | None = None, no_viewer: bool = False) -> None:
    arrays, meta = load_episode(episode)
    rec = meta.get("episode_record", {})
    print(f"episode {rec.get('episode_id')}: {rec.get('terminal_reason')} after {rec.get('steps')} steps "
          f"(restored from stored simulator states, not re-simulated)")
    if png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
        plot_topdown(ax, arrays, meta)
        ax.set_title(f"{rec.get('episode_id')}\n{rec.get('terminal_reason')}", fontsize=9)
        fig.savefig(png, bbox_inches="tight")
        print("wrote", png)
    if gif:
        render_gif(arrays, meta, gif, speed=speed, stride=2)
        print("wrote", gif)
    if no_viewer or png or gif:
        return
    try:
        view(arrays, meta, speed)
    except RuntimeError as e:
        print(f"viewer unavailable ({e}).\nOn macOS launch with: mjpython -m roborobust replay --episode {episode}\n"
              "or add --png out.png / --gif out.gif for headless output.")
