"""Development sweep on the EMPTY-ROOM task (Gate C debugging). Never touches the test split.

One job = train one nominal PPO on empty_train with a reward/env override, then evaluate it
deterministically on the 20 empty-room VALIDATION cases. Results are appended to
runs/dev/sweep_results.jsonl and summarised in docs/dev_log.md, so the tuning history (including the
settings that did NOT help) is on record.

  python scripts/dev_sweep.py --alpha 10 --n-envs 4 --seed 0 --steps 200000 --tag a10
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from roborobust.config import deep_merge, load_yaml
from roborobust.evaluate import evaluate
from roborobust.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--step-cost", type=float, default=None)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="configs/empty_diag_nominal.yaml")
    ap.add_argument("--ppo", default=None, help="JSON dict of PPO overrides, e.g. '{\"ent_coef\": 0.001}'")
    ap.add_argument("--suite", default="configs/empty_val_suite.yaml")
    a = ap.parse_args()

    dev = Path("runs/dev")
    (dev / "_cfg").mkdir(parents=True, exist_ok=True)
    env = load_yaml("configs/env.yaml")
    if a.alpha is not None:
        env["reward"]["shaping_alpha"] = a.alpha
    if a.step_cost is not None:
        env["reward"]["step_cost"] = a.step_cost
    env_path = dev / "_cfg" / f"env_{a.tag}.yaml"
    env_path.write_text(yaml.safe_dump(env))
    over = {"env_config": str(env_path.resolve())}
    if a.ppo:
        over["ppo"] = json.loads(a.ppo)
    run_id = f"dev_{a.tag}_s{a.seed}"
    run_dir = train(a.base, a.seed, a.steps, run_id, runs_dir=dev, n_envs=a.n_envs, overwrite=True, extra_overrides=over)
    # evaluate with the SAME (overridden) env config so reward overrides do not matter for the verifier
    suite = load_yaml(a.suite)
    suite_path = dev / "_cfg" / f"suite_{a.tag}.yaml"
    suite["env_config"] = str(env_path.resolve())
    suite_path.write_text(yaml.safe_dump(suite))
    out = dev / f"eval_{run_id}"
    evaluate(suite_path, [str(run_dir)], [], out, workers=1, save_trajectories="none", overwrite=True)
    rows = [json.loads(l) for l in open(out / "episode_results.jsonl")]
    n = len(rows)
    res = {"tag": a.tag, "seed": a.seed, "alpha": a.alpha, "step_cost": a.step_cost, "n_envs": a.n_envs, "steps": a.steps,
           "ppo": a.ppo, "val_success": sum(r["success"] for r in rows), "val_n": n,
           "val_collision": sum(r["collision"] for r in rows), "suite": a.suite,
           "mean_final_goal_dist": sum(r["features"]["final_goal_distance"] for r in rows) / n}
    with open(dev / "sweep_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
