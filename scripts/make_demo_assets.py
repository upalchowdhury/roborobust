"""Build the demo assets for the three-minute narrative (see DEMO.md) from SAVED trajectories only.

  python scripts/make_demo_assets.py [results_dir] [out_dir]

Nothing here re-runs an experiment: every picture comes from results/<suite>/trajectories/*.npz.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roborobust.replay import load_episode, plot_sensors, plot_topdown, render_gif

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/weekend")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "demo")
out.mkdir(parents=True, exist_ok=True)
traj = res / "trajectories"


def ep(policy_seed: str, cond: str, sid: str):
    return load_episode(traj / f"{policy_seed}__{cond}__{sid}.npz")


# 1. Task and sensors: the 24-ray fan at three moments of one nominal-condition PPO-nominal episode.
a, m = ep("nominal_s0", "nominal", "main_test_0000")
n = len(a["qpos"])
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=140)
for ax, k in zip(axes, (0, n // 3, 2 * n // 3)):
    plot_sensors(ax, a, m, k)
    ax.set_title(f"control step {k} (t = {a['t'][k]:.1f} s)", fontsize=9, loc="left")
fig.suptitle("What the policy sees: 24 range rays (dashed = max range 3 m, red = hit). Goal bearing/distance and velocities are privileged inputs.",
             fontsize=9, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out / "01_task_and_sensors.png")
plt.close(fig)

# 2. Paired stress replay: same scenario, same weak-left-motor + noisy-sensing shift, both policies (seed 0).
sid = "main_test_0000"
for pol in ("nominal_s0", "randomized_s0"):
    arrays, meta = ep(pol, "combined_stress", sid)
    render_gif(arrays, meta, out / f"02_paired_{pol}_combined_stress_{sid}.gif", stride=2)
fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=140)
for pol, color in (("nominal_s0", "#2a78d6"), ("randomized_s0", "#eb6834")):
    arrays, meta = ep(pol, "combined_stress", sid)
    plot_topdown(ax, arrays, meta, color=color, label=f"{pol.split('_')[0]} -> {meta['episode_record']['terminal_reason']}", draw_room=(pol == "nominal_s0"))
ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.set_title(f"Paired replay, {sid}, combined stress", fontsize=9, loc="left")
fig.savefig(out / "02_paired_overlay.png", bbox_inches="tight")
plt.close(fig)

# 3. A failure, chosen by the documented rule (first failure by scenario order for PPO-randomized seed 0, combined stress).
arrays, meta = ep("randomized_s0", "combined_stress", "main_test_0004")
render_gif(arrays, meta, out / "03_failure_randomized_s0_combined_stress_main_test_0004.gif", stride=2)

# 4. Aggregate results and the failure examples already produced by `roborobust report`.
for name in ("comparison.png", "failure_examples.png", "learning_curves.png"):
    src = Path("reports/weekend") / name
    if src.exists():
        (out / f"04_{name}").write_bytes(src.read_bytes())
print("demo assets in", out, sorted(p.name for p in out.iterdir()))
