"""Development sweep on the MAIN task (1-2 obstacles), training on main_train and scoring on main VALIDATION.

Motivation (docs/dev_log.md section 4): at the original protocol some seeds of BOTH conditions converged to a
safe idling policy (0 collisions, ~100% timeouts, ~0% success). This script tests training-stability levers
(discount factor, entropy bonus, ...) on DEV seeds that are disjoint from the final study's seeds, for both
conditions symmetrically. The test split is never touched.

  python scripts/dev_main_sweep.py --cond randomized --seed 10 --tag g995 --gamma 0.995
"""
import argparse
import json
from pathlib import Path

import yaml

from roborobust.config import load_yaml
from roborobust.evaluate import evaluate
from roborobust.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["nominal", "randomized"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--bonus", type=float, default=None)
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--ent", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-envs", type=int, default=4)
    a = ap.parse_args()

    dev = Path("runs/dev/main")
    (dev / "_cfg").mkdir(parents=True, exist_ok=True)
    env = load_yaml("configs/env.yaml")
    ppo: dict = {}
    if a.gamma is not None:
        env["reward"]["gamma"] = a.gamma
        ppo["gamma"] = a.gamma
    if a.alpha is not None:
        env["reward"]["shaping_alpha"] = a.alpha
    if a.bonus is not None:
        env["reward"]["success_bonus"] = a.bonus
    if a.penalty is not None:
        env["reward"]["failure_penalty"] = a.penalty
    if a.ent is not None:
        ppo["ent_coef"] = a.ent
    if a.lr is not None:
        ppo["learning_rate"] = a.lr
    env_path = (dev / "_cfg" / f"env_{a.tag}.yaml").resolve()
    env_path.write_text(yaml.safe_dump(env))
    over = {"env_config": str(env_path)}
    if ppo:
        over["ppo"] = ppo
    run_id = f"{a.tag}_{a.cond}_s{a.seed}"
    run_dir = train(f"configs/{a.cond}.yaml", a.seed, a.steps, run_id, runs_dir=dev, n_envs=a.n_envs, overwrite=True, extra_overrides=over)
    suite = load_yaml("configs/val_suite.yaml")
    suite["env_config"] = str(env_path)
    sp = dev / "_cfg" / f"suite_{a.tag}.yaml"
    sp.write_text(yaml.safe_dump(suite))
    out = dev / f"eval_{run_id}"
    evaluate(sp, [str(run_dir)], [], out, workers=1, save_trajectories="none", overwrite=True)
    rows = [json.loads(l) for l in open(out / "episode_results.jsonl")]
    res = {"tag": a.tag, "cond": a.cond, "seed": a.seed, "steps": a.steps, "gamma": a.gamma, "alpha": a.alpha, "bonus": a.bonus,
           "penalty": a.penalty, "ent": a.ent, "lr": a.lr, "n_envs": a.n_envs, "by_condition": {}}
    for c in suite["conditions"]:
        rc = [r for r in rows if r["condition"] == c]
        res["by_condition"][c] = {"n": len(rc), "success": sum(r["success"] for r in rc), "collision": sum(r["collision"] for r in rc),
                                  "timeout": sum(r["terminal_reason"] == "timeout" for r in rc)}
    with open(dev / "main_sweep_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    n = res["by_condition"]["nominal"]
    print(f"RESULT {run_id}: nominal {n['success']}/{n['n']} combined {res['by_condition']['combined_stress']['success']}/{n['n']}")


if __name__ == "__main__":
    main()
