"""Measured training throughput for 1/4/8 (or any) workers: env-only and full PPO (collect + update).

Worker count is chosen from these numbers (total transitions/s and memory), not from the core count.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from roborobust.config import load_env_config, load_yaml, resolve_path
from roborobust.randomization import RandomizationConfig


def _rss_mb(pid: int) -> float:
    """Resident memory of a process plus its direct children, MB (via ps; no psutil dependency)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    total = 0
    for line in out.splitlines():
        p, pp, rss = (int(x) for x in line.split())
        if p == pid or pp == pid:
            total += rss
    return total / 1024.0


def benchmark(workers: list[int], config: str = "configs/randomized.yaml", ppo_steps: int = 32768,
              env_steps: int = 20000, output: str | Path = "results/benchmark.json") -> dict:
    import torch
    from stable_baselines3 import PPO

    from roborobust.train import _limit_threads, make_vec_env

    cfg = load_yaml(config)
    env_cfg, _ = load_env_config(cfg["env_config"])
    manifest = str(resolve_path(cfg["train_manifest"]))
    rand = RandomizationConfig.from_dict(cfg.get("randomization")).to_dict()
    _limit_threads(1)
    torch.set_num_threads(1)
    rows = []
    for n in workers:
        ppo = cfg["ppo"]
        # --- env-only (random actions): how fast can the simulator feed PPO? ---
        venv = make_vec_env(env_cfg.to_dict(), manifest, rand, n)
        venv.seed(0)
        venv.reset()
        rng = np.random.default_rng(0)
        t0 = time.time()
        steps = 0
        while steps < env_steps:
            venv.step(rng.uniform(-1, 1, (n, 2)).astype(np.float32))
            steps += n
        env_sps = steps / (time.time() - t0)
        venv.close()

        # --- full PPO iterations (rollout collection with policy inference + gradient updates) ---
        rollout = ppo["n_steps"] * n
        if rollout % ppo["batch_size"]:
            raise ValueError("rollout not divisible by batch size")
        venv = make_vec_env(env_cfg.to_dict(), manifest, rand, n)
        model = PPO("MlpPolicy", venv, n_steps=ppo["n_steps"], batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
                    gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], seed=0, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=list(ppo["net_arch"]), vf=list(ppo["net_arch"]))))
        n_iters = max(2, round(ppo_steps / rollout))
        model.learn(total_timesteps=rollout)  # warm-up iteration (imports, spawn, JIT-ish effects)
        t0 = time.time()
        model.learn(total_timesteps=n_iters * rollout, reset_num_timesteps=False)
        dt = time.time() - t0
        mem = _rss_mb(os.getpid())
        venv.close()
        rows.append({"workers": n, "env_only_steps_per_s": env_sps, "ppo_steps_per_s": n_iters * rollout / dt,
                     "ppo_measured_transitions": n_iters * rollout, "rss_mb_parent_plus_children": mem})
        print(f"[benchmark] workers={n}: env-only {env_sps:,.0f} steps/s | PPO {rows[-1]['ppo_steps_per_s']:,.0f} steps/s "
              f"| ~{mem:,.0f} MB", flush=True)
    best = max(rows, key=lambda r: r["ppo_steps_per_s"])
    result = {"rows": rows, "best_by_ppo_throughput": best["workers"], "ncpu": os.cpu_count(), "config": config,
              "note": "Measured on this machine only; throughput depends on thermal state and background load."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return result
