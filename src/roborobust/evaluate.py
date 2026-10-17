"""Paired, deterministic evaluation of trained policies and baselines.

Pairing: every policy sees the same scenarios and, per (scenario, condition), the same measurement-noise
stream (seeded from eval_seed + scenario_id + condition, independent of the policy). The streams are
*common exogenous randomness*, not identical sensor readings: two policies following different
trajectories see different true ranges to which the same noise sequence is added.

No learning happens here: PPO inference is deterministic (`predict(deterministic=True)`), the policy is
in eval mode, no normalisation statistics exist, and test_experiment.py checks that weights are
bit-identical before and after evaluation.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from roborobust.baselines import Policy, RandomPolicy, ScriptedReactive
from roborobust.config import EnvConfig, config_hash, load_env_config, load_yaml, resolve_path, sha256_file
from roborobust.randomization import DomainParams
from roborobust.runtime import provenance
from roborobust.scenarios import Scenario, load_manifest


def stable_int(*parts, bits: int = 32) -> int:
    """Process-independent hash -> int (Python's hash() is salted per process, so never use it for seeds)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[: bits // 8], "little")


class PPOPolicy(Policy):
    def __init__(self, run_dir: Path, checkpoint: str = "final.zip"):
        from stable_baselines3 import PPO

        self.run_dir = Path(run_dir)
        self.meta = json.loads((self.run_dir / "meta.json").read_text())
        self.ckpt_path = self.run_dir / checkpoint
        self.model = PPO.load(self.ckpt_path, device="cpu")
        self.model.policy.set_training_mode(False)
        self.label = self.meta["name"]
        self.training_seed = self.meta["training_seed"]
        self.run_id = self.meta["run_id"]
        self.checkpoint_sha256 = sha256_file(self.ckpt_path)
        # transitions seen by THIS checkpoint (periodic checkpoints are named step_<transitions>.zip)
        m = re.search(r"step_(\d+)\.zip$", checkpoint)
        self.train_transitions = int(m.group(1)) if m else self.meta["actual_transitions"]
        self.train_elapsed_s = None if m else self.meta["elapsed_s"]

    def act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

    def weights_hash(self) -> str:
        import torch

        h = hashlib.sha256()
        for k, v in sorted(self.model.policy.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()


def build_policy(spec: dict, env_cfg: EnvConfig) -> tuple[Policy, dict]:
    kind = spec["kind"]
    if kind == "ppo":
        p = PPOPolicy(Path(spec["run_dir"]), spec.get("checkpoint", "final.zip"))
        return p, {"run_id": p.run_id, "policy": p.label, "training_seed": p.training_seed,
                   "checkpoint_sha256": p.checkpoint_sha256, "train_transitions": p.train_transitions,
                   "train_elapsed_s": p.train_elapsed_s, "checkpoint": spec.get("checkpoint", "final.zip"),
                   "train_config_hash": p.meta.get("config_hash")}
    if kind == "scripted":
        return ScriptedReactive(env_cfg), {"run_id": "baseline:scripted", "policy": "scripted", "training_seed": None,
                                           "checkpoint_sha256": None, "train_transitions": 0, "train_elapsed_s": 0.0}
    if kind == "random":
        return RandomPolicy(), {"run_id": "baseline:random", "policy": "random", "training_seed": None,
                                "checkpoint_sha256": None, "train_transitions": 0, "train_elapsed_s": 0.0}
    raise ValueError(kind)


def episode_id(ident: dict, condition: str, scenario_id: str) -> str:
    tag = ident["policy"] if ident["training_seed"] is None else f"{ident['policy']}_s{ident['training_seed']}"
    return f"{tag}__{condition}__{scenario_id}"


# --------------------------------------------------------------------------------------
# One task = one policy x one condition x all scenarios (runs inside a worker process)
# --------------------------------------------------------------------------------------
def run_task(task: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from roborobust.env import RoboRobustEnv

    suite = task["suite"]
    env_cfg = EnvConfig.from_dict(task["env_cfg"])
    scenarios = [Scenario.from_dict(s) for s in task["scenarios"]]
    policy, ident = build_policy(task["policy"], env_cfg)
    cond_name = task["condition"]
    cond = suite["conditions"][cond_name]
    domain = DomainParams(left_motor_mult=cond["left"], right_motor_mult=cond["right"], range_noise_std=cond["noise_std"])
    env = RoboRobustEnv(env_cfg, scenarios, record=task["record"] != "none")
    out_dir = Path(task["output"])

    records, trajs = [], {}
    t0 = time.time()
    for idx, sc in enumerate(scenarios):
        noise_seed = stable_int(suite["eval_seed"], sc.scenario_id, cond_name, "noise")
        obs, _ = env.reset(seed=suite["eval_seed"], options={"scenario": sc, "domain": domain, "noise_seed": noise_seed})
        policy.reset(stable_int(suite["random_baseline_seed"], sc.scenario_id, cond_name, "policy"))
        done = False
        while not done:
            obs, _, te, tr, info = env.step(policy.act(obs))
            done = te or tr
        s = info["episode_summary"]
        eid = episode_id(ident, cond_name, sc.scenario_id)
        rec = {
            "episode_id": eid,
            "run_id": ident["run_id"], "policy": ident["policy"], "training_seed": ident["training_seed"],
            "scenario_id": sc.scenario_id, "layout_id": sc.layout_id, "scenario_index": idx, "condition": cond_name,
            "manifest_sha256": task["manifest_sha256"], "env_cfg_hash": env.cfg_hash,
            "checkpoint_sha256": ident["checkpoint_sha256"], "eval_seed": suite["eval_seed"], "noise_seed": noise_seed,
            "terminal_reason": s["terminal_reason"], "success": s["success"], "collision": s["collision"],
            "terminated": s["terminated"], "truncated": s["truncated"],
            "steps": s["steps"], "duration_s": s["duration_s"], "time_to_goal_s": s["time_to_goal_s"],
            "first_goal_entry_s": s["first_goal_entry_s"], "path_length_m": s["path_length_m"],
            "return": s["return"], "reward_components": s["components"],
            "sampled_params": s["domain"], "clip_count": s["clip_count"],
            "motor_saturation_frac": s["motor_saturation_frac"], "collision_event": s["collision_event"],
            "features": s["features"], "trajectory_file": None,
        }
        records.append(rec)
        if task["record"] != "none":
            trajs[eid] = env.get_trajectory()

    saved = _select_and_save(records, trajs, task, out_dir)
    return {"policy": ident, "condition": cond_name, "records": records, "saved_trajectories": saved,
            "elapsed_s": time.time() - t0, "weights_hash": policy.weights_hash() if isinstance(policy, PPOPolicy) else None}


def _select_and_save(records, trajs, task, out_dir: Path) -> list[str]:
    """Trajectory subset rule (documented, fixed before results): pinned scenarios for every task,
    plus the first success and the first two failures in scenario order. mode 'all' saves everything."""
    if task["record"] == "none":
        return []
    mode = task["record"]
    chosen: set[str] = set()
    if mode == "all":
        chosen = {r["episode_id"] for r in records}
    else:
        chosen |= {r["episode_id"] for r in records[: task["suite"]["trajectories"]["pinned_scenarios"]]}
        succ = [r for r in records if r["success"]]
        fail = [r for r in records if not r["success"]]
        chosen |= {r["episode_id"] for r in succ[:1]}
        chosen |= {r["episode_id"] for r in fail[:2]}
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    by_id = {r["episode_id"]: r for r in records}
    for eid in sorted(chosen):
        t = trajs[eid]
        meta = {k: t[k] for k in ("scenario", "domain", "events", "cfg_hash", "env_config", "reward_columns")}
        meta["episode_record"] = by_id[eid]
        arrays = {k: t[k] for k in ("t", "qpos", "qvel", "ctrl", "obs", "action_raw", "action", "reward")}
        np.savez_compressed(out_dir / "trajectories" / f"{eid}.npz", meta_json=json.dumps(meta, default=str), **arrays)
        by_id[eid]["trajectory_file"] = f"trajectories/{eid}.npz"
    return sorted(chosen)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def evaluate(
    suite_path: str | Path,
    runs: list[str],
    baselines: list[str],
    output: str | Path,
    workers: int = 6,
    save_trajectories: str = "subset",
    overwrite: bool = False,
    checkpoint: str = "final.zip",
) -> Path:
    suite = load_yaml(suite_path)
    suite_file = resolve_path(suite_path)
    env_cfg, env_raw = load_env_config(suite["env_config"])
    manifest = resolve_path(suite["manifest"])
    scenarios = load_manifest(manifest)
    out = Path(output)
    if (out / "episode_results.jsonl").exists() and not overwrite:
        raise FileExistsError(f"{out}/episode_results.jsonl exists; refusing to overwrite (pass --overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    splits = {s.split for s in scenarios}
    if "test" in splits:  # audit trail of every touch of the held-out split
        log = Path("results") / "test_access_log.jsonl"
        log.parent.mkdir(exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "suite": str(suite_path),
                                "suite_sha256": sha256_file(suite_file), "runs": runs, "baselines": baselines,
                                "output": str(out)}) + "\n")

    policies = [{"kind": "ppo", "run_dir": r, "checkpoint": checkpoint} for r in runs] + [{"kind": b} for b in baselines]
    tasks = [
        {"suite": suite, "env_cfg": env_cfg.to_dict(), "scenarios": [s.to_dict() for s in scenarios], "policy": p,
         "condition": c, "record": save_trajectories, "output": str(out), "manifest_sha256": sha256_file(manifest)}
        for p in policies for c in suite["conditions"]
    ]
    t0 = time.time()
    results, incomplete = [], []
    if workers <= 1:
        for t in tasks:
            try:
                results.append(run_task(t))
            except Exception:
                incomplete.append({"policy": t["policy"], "condition": t["condition"], "error": traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t): t for t in tasks}
            for fu in as_completed(futs):
                t = futs[fu]
                try:
                    results.append(fu.result())
                    print(f"[eval] done {t['policy'].get('run_dir', t['policy']['kind'])} / {t['condition']}", flush=True)
                except Exception:
                    incomplete.append({"policy": t["policy"], "condition": t["condition"], "error": traceback.format_exc()})
                    print(f"[eval] FAILED {t['policy']} / {t['condition']}", flush=True)

    records = [r for res in results for r in res["records"]]
    records.sort(key=lambda r: (r["policy"], r["training_seed"] if r["training_seed"] is not None else -1,
                                list(suite["conditions"]).index(r["condition"]), r["scenario_index"]))
    with open(out / "episode_results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    expected = len(policies) * len(suite["conditions"]) * len(scenarios)
    meta = {
        "suite": suite, "suite_sha256": sha256_file(suite_file), "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest), "env_cfg_hash": config_hash(env_raw),
        "policies": [r["policy"] for r in results], "expected_episodes": expected, "recorded_episodes": len(records),
        "complete": len(records) == expected and not incomplete, "incomplete_tasks": incomplete,
        "task_weights_hash": sorted({(r["policy"]["run_id"], r["weights_hash"]) for r in results if r["weights_hash"]}),
        "elapsed_s": time.time() - t0, "checkpoint_name": checkpoint, **provenance(),
    }
    meta["policies"] = [dict(p) for p in {json.dumps(r["policy"], sort_keys=True): r["policy"] for r in results}.values()]
    meta["task_weights_hash"] = [list(x) for x in meta["task_weights_hash"]]
    (out / "eval_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    status = "COMPLETE" if meta["complete"] else f"INCOMPLETE ({len(records)}/{expected} episodes, {len(incomplete)} failed tasks)"
    print(f"[eval] {status}; wrote {out}/episode_results.jsonl in {meta['elapsed_s']:.0f}s", flush=True)
    if incomplete:
        raise RuntimeError(f"evaluation incomplete: {len(incomplete)} task(s) failed; see {out}/eval_meta.json")
    return out
