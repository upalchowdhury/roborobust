"""PPO training with SB3 on the RoboRobust environment (CPU, MlpPolicy 64x64).

Honest accounting notes
  * "Transitions" = environment steps summed over all workers. PPO collects whole rollouts
    (n_steps * n_envs), so the actual count is the requested count rounded UP to a rollout multiple;
    both are recorded. Nominal and randomized runs use identical n_envs/n_steps, hence identical budgets.
  * Final checkpoint is the primary comparison artifact; periodic checkpoints exist for the optional,
    identically-defined validation-based secondary selection.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from roborobust.config import (
    EnvConfig, REPO_ROOT, config_hash, load_env_config, load_yaml, resolve_path, sha256_file,
)
from roborobust.randomization import RandomizationConfig
from roborobust.runtime import provenance


def _limit_threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)


def validate_train_config(cfg: dict, env_cfg: EnvConfig, n_envs: int) -> None:
    ppo = cfg["ppo"]
    if not math.isclose(ppo["gamma"], env_cfg.reward.gamma):
        raise ValueError(f"ppo.gamma={ppo['gamma']} must equal env reward.gamma={env_cfg.reward.gamma} "
                         "(potential-based shaping assumes the learner's discount)")
    rollout = ppo["n_steps"] * n_envs
    if rollout % ppo["batch_size"] != 0:
        raise ValueError(f"rollout size {rollout} (= n_steps*n_envs) not divisible by batch_size {ppo['batch_size']}")


def make_env_factory(env_cfg_dict: dict, manifest: str, randomization: dict, monitor: bool = True):
    """Picklable (cloudpickle) env thunk for spawn-based SubprocVecEnv. No rendering in workers."""

    def _thunk():
        from stable_baselines3.common.monitor import Monitor

        from roborobust.env import RoboRobustEnv

        env = RoboRobustEnv(EnvConfig.from_dict(env_cfg_dict), manifest, randomization=RandomizationConfig.from_dict(randomization))
        return Monitor(env) if monitor else env

    return _thunk


def make_vec_env(env_cfg_dict: dict, manifest: str, randomization: dict, n_envs: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env_factory(env_cfg_dict, manifest, randomization) for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def _callbacks(run_dir: Path, checkpoint_every: int, n_envs: int, t0: float):
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeStats(BaseCallback):
        """Writes every finished training episode to train_episodes.jsonl, logs per-rollout rates, saves checkpoints."""

        def __init__(self):
            super().__init__()
            self.f = open(run_dir / "train_episodes.jsonl", "w")
            self.buf: list[dict] = []
            self.next_ckpt = checkpoint_every
            (run_dir / "checkpoints").mkdir(exist_ok=True)

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                s = info.get("episode_summary")
                if s is not None:
                    row = {"num_timesteps": self.num_timesteps, **{k: v for k, v in s.items() if k != "collision_event"}}
                    self.f.write(json.dumps(row) + "\n")
                    self.buf.append(s)
            return True

        def _on_rollout_end(self) -> None:
            b, rec = self.buf, self.logger.record
            if b:
                n = len(b)
                rec("custom/episodes", n)
                for reason in ("success", "collision", "timeout", "tipped", "out_of_bounds"):
                    rec(f"custom/rate_{reason}", sum(s["terminal_reason"] == reason for s in b) / n)
                rec("custom/ep_return", float(np.mean([s["return"] for s in b])))
                for k in ("step_cost", "shaping", "success_bonus", "failure_penalty"):
                    rec(f"custom/ret_{k}", float(np.mean([s["components"][k] for s in b])))
                rec("custom/ep_steps", float(np.mean([s["steps"] for s in b])))
                rec("custom/clip_frac", float(np.mean([s["clip_count"] / max(s["steps"], 1) for s in b])))
                rec("custom/motor_sat_frac", float(np.mean([np.mean(s["motor_saturation_frac"]) for s in b])))
            self.buf = []
            self.f.flush()
            if self.num_timesteps >= self.next_ckpt:
                self.model.save(run_dir / "checkpoints" / f"step_{self.num_timesteps:09d}")
                self.next_ckpt += checkpoint_every
            it = self.num_timesteps // (self.model.n_steps * n_envs)
            if it % 10 == 0:
                print(f"[train] it={it} steps={self.num_timesteps} elapsed={time.time()-t0:.0f}s "
                      f"success={self.logger.name_to_value.get('custom/rate_success', float('nan')):.2f}", flush=True)

        def _on_training_end(self) -> None:
            self.f.close()

    return EpisodeStats()


def train(config_path: str | Path, seed: int, steps: int, run_id: str, runs_dir: str | Path = "runs",
          n_envs: int | None = None, overwrite: bool = False, extra_overrides: dict | None = None) -> Path:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    cfg = load_yaml(config_path)
    if extra_overrides:
        from roborobust.config import deep_merge
        cfg = deep_merge(cfg, extra_overrides)
    n_envs = n_envs or cfg["n_envs"]
    env_cfg, env_raw = load_env_config(cfg["env_config"])
    validate_train_config(cfg, env_cfg, n_envs)

    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; refusing to overwrite silently (pass --overwrite to replace it)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    _limit_threads(1)
    torch.set_num_threads(cfg.get("torch_threads", 1))
    ppo = cfg["ppo"]
    rollout = ppo["n_steps"] * n_envs
    n_iters = math.ceil(steps / rollout)
    actual_steps = n_iters * rollout

    manifest = str(resolve_path(cfg["train_manifest"]))
    rand = RandomizationConfig.from_dict(cfg.get("randomization")).to_dict()
    venv = make_vec_env(env_cfg.to_dict(), manifest, rand, n_envs)

    resolved = {**cfg, "n_envs": n_envs, "seed": seed, "requested_steps": steps, "actual_steps": actual_steps, "run_id": run_id}
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    model = PPO(
        ppo["policy"], venv, learning_rate=ppo["learning_rate"], n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
        ent_coef=ppo["ent_coef"], policy_kwargs=dict(net_arch=dict(pi=list(ppo["net_arch"]), vf=list(ppo["net_arch"]))),
        seed=seed, device=cfg.get("device", "cpu"), verbose=0,
    )
    model.set_logger(configure(str(run_dir), ["csv"]))
    t0 = time.time()
    cb = _callbacks(run_dir, cfg["checkpoint_every"], n_envs, t0)
    try:
        model.learn(total_timesteps=actual_steps, callback=cb, progress_bar=False)
    finally:
        venv.close()
    elapsed = time.time() - t0

    final = run_dir / "final.zip"
    model.save(final)
    reload_ok, max_diff = _reload_check(final, model, env_cfg, manifest)

    import pandas as pd
    prog = pd.read_csv(run_dir / "progress.csv")
    # SB3 dumps the first row before its first gradient update, so train/* is legitimately empty there;
    # every later row must be finite (NaN/inf in any loss would mean divergence).
    loss_cols = [c for c in ("train/loss", "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss") if c in prog]
    finite = bool(len(prog) > 1 and np.isfinite(prog[loss_cols].iloc[1:].to_numpy(dtype=float)).all()) if loss_cols else False

    meta = {
        "run_id": run_id, "name": cfg["name"], "training_seed": seed,
        "requested_steps": steps, "actual_transitions": int(model.num_timesteps), "n_envs": n_envs,
        "rollout_size": rollout, "n_iterations": n_iters, "elapsed_s": elapsed,
        "steps_per_s": model.num_timesteps / elapsed,
        "randomization": rand, "ppo": ppo,
        "config_hash": config_hash({k: v for k, v in resolved.items() if k not in ("run_id", "seed")}),
        "env_cfg_hash": config_hash(env_raw),
        "train_manifest": cfg["train_manifest"], "train_manifest_sha256": sha256_file(manifest),
        "checkpoint": "final.zip", "checkpoint_sha256": sha256_file(final),
        "reload_check": {"identical_deterministic_actions": reload_ok, "max_abs_diff": max_diff},
        "finite_losses": finite,
        **provenance(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] done {run_id}: {model.num_timesteps} transitions in {elapsed:.0f}s "
          f"({meta['steps_per_s']:.0f}/s); reload_ok={reload_ok} finite_losses={finite}", flush=True)
    return run_dir


def _reload_check(path: Path, model, env_cfg: EnvConfig, manifest: str) -> tuple[bool, float]:
    """Save/reload gate: the reloaded policy must give the same deterministic actions on fixed observations."""
    from stable_baselines3 import PPO

    from roborobust.env import RoboRobustEnv

    env = RoboRobustEnv(env_cfg, manifest)
    rng = np.random.default_rng(0)
    obs = []
    o, _ = env.reset(seed=0)
    for _ in range(64):
        obs.append(o)
        o, _, te, tr, _ = env.step(rng.uniform(-1, 1, 2))
        if te or tr:
            o, _ = env.reset()
    obs = np.stack(obs)
    reloaded = PPO.load(path, device="cpu")
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = reloaded.predict(obs, deterministic=True)
    diff = float(np.abs(a1 - a2).max())
    return diff == 0.0, diff
