"""Command-line interface: ``roborobust <command>`` == ``python -m roborobust <command>``."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def _doctor(args) -> int:
    ok = True

    def line(status, name, detail=""):
        print(f"  [{status:^4}] {name}{(': ' + detail) if detail else ''}")

    print("roborobust doctor")
    py = sys.version_info
    line("OK" if py[:2] == (3, 11) else "WARN", "python", f"{platform.python_version()} ({platform.machine()}, {platform.platform()})")
    if py[:2] != (3, 11):
        print("        the spec targets Python 3.11; other versions are untested")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        line("WARN", "architecture", "not arm64: Apple-silicon-native wheels expected (are you under Rosetta?)")
    for mod in ("mujoco", "gymnasium", "stable_baselines3", "torch", "numpy", "pandas", "matplotlib", "yaml"):
        try:
            m = __import__(mod)
            line("OK", f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            line("FAIL", f"import {mod}", str(e))
            ok = False
    try:
        import numpy as np
        import mujoco

        from roborobust.env import build_model

        m = build_model([])
        d = mujoco.MjData(m)
        for _ in range(200):
            mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all()
        line("OK", "MuJoCo stepping on robot.xml", f"200 steps, chassis z={d.qpos[2]:.4f} m")
    except Exception as e:  # noqa: BLE001
        line("FAIL", "MuJoCo stepping", repr(e))
        ok = False
    try:
        import torch

        line("OK", "torch CPU", f"threads={torch.get_num_threads()}, MPS available={torch.backends.mps.is_available()} (unused: CPU training)")
    except Exception as e:  # noqa: BLE001
        line("FAIL", "torch", repr(e))
        ok = False
    try:  # optional: rendering must never block headless training
        import mujoco

        from roborobust.env import build_model

        m = build_model([])
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 64, 64)
        r.update_scene(d)
        img = r.render()
        r.close()
        line("OK", "offscreen renderer (optional)", f"frame {img.shape}")
    except Exception as e:  # noqa: BLE001
        line("SKIP", "offscreen renderer (optional)", f"unavailable ({type(e).__name__}); headless training is unaffected")
    line("OK" if shutil.which("mjpython") else "SKIP", "mjpython (needed only for the interactive viewer)", shutil.which("mjpython") or "not on PATH")
    try:
        from roborobust.scenarios import load_scenario_config, verify_manifests

        problems = verify_manifests(Path("data/scenarios"), load_scenario_config("configs/scenarios.yaml"))
        line("OK" if not problems else "FAIL", "scenario manifests", "verified" if not problems else f"{len(problems)} problem(s)")
        ok &= not problems
    except Exception as e:  # noqa: BLE001
        line("WARN", "scenario manifests", f"not checked ({e}); run `roborobust generate-scenarios`")
    print("doctor:", "all required checks passed" if ok else "REQUIRED CHECKS FAILED")
    return 0 if ok else 1


def _generate(args) -> int:
    from roborobust.scenarios import generate_all, load_scenario_config, verify_manifests, write_manifests

    out = Path(args.output)
    if (out / "manifest_hashes.json").exists() and not args.force:
        print(f"{out}/manifest_hashes.json already exists. Manifests are frozen once experiments start; "
              "pass --force only if you intend to invalidate every result that used them.", file=sys.stderr)
        return 2
    cfg = load_scenario_config(args.config)
    manifests, stats = generate_all(cfg)
    summary = write_manifests(manifests, out, cfg, stats)
    problems = verify_manifests(out, cfg)
    for name, info in summary["manifests"].items():
        print(f"  {name}: n={info['n']} sha256={info['sha256'][:16]}...")
    if problems:
        print("VERIFICATION PROBLEMS:\n  " + "\n  ".join(problems[:20]))
        return 1
    print("manifests verified: disjoint layouts, valid starts/goals, connected on inflated grid")
    return 0


def _validate_env(args) -> int:
    from roborobust.config import load_env_config
    from roborobust.scenarios import load_scenario_config, verify_manifests
    from roborobust.validation import run_all

    cfg = load_env_config(args.env_config)[0]
    results = run_all(cfg)
    problems = verify_manifests(Path("data/scenarios"), load_scenario_config("configs/scenarios.yaml"))
    results.append({"name": "scenario_manifests", "passed": not problems, "measured": problems[:10], "criterion": "disjoint, valid, connected, hashed"})
    for r in results:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['name']}: {r['criterion']}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"measurements written to {out}")
    return 0 if all(r["passed"] for r in results) else 1


def _benchmark(args) -> int:
    from roborobust.benchmark import benchmark

    benchmark(args.workers, config=args.config, ppo_steps=args.ppo_steps)
    return 0


def _train(args) -> int:
    from roborobust.train import train

    train(args.config, args.seed, args.steps, args.run_id, runs_dir=args.runs_dir, n_envs=args.n_envs, overwrite=args.overwrite)
    return 0


def _evaluate(args) -> int:
    from roborobust.evaluate import evaluate

    evaluate(args.suite, args.runs, args.baselines, args.output, workers=args.workers,
             save_trajectories=args.save_trajectories, overwrite=args.overwrite, checkpoint=args.checkpoint)
    return 0


def _report(args) -> int:
    from roborobust.analyze import report

    report(args.input, args.output, runs_dir=args.runs_dir, manual_labels=args.manual_labels, secondary=args.secondary)
    return 0


def _replay(args) -> int:
    from roborobust.replay import replay

    if args.resimulate_check:
        from roborobust.replay import check_resimulation

        r = check_resimulation(args.episode)
        print(json.dumps(r, indent=2))
        return 0 if r["within_tolerance"] else 1
    replay(args.episode, speed=args.speed, png=args.png, gif=args.gif, no_viewer=args.no_viewer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roborobust", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="check Python/arch, imports, MuJoCo stepping, optional renderer")
    s.set_defaults(fn=_doctor)

    s = sub.add_parser("generate-scenarios", help="generate + verify frozen scenario manifests")
    s.add_argument("--config", default="configs/scenarios.yaml")
    s.add_argument("--output", default="data/scenarios")
    s.add_argument("--force", action="store_true", help="regenerate even though manifests exist (invalidates results)")
    s.set_defaults(fn=_generate)

    s = sub.add_parser("validate-env", help="Gate A/B physics + sensor + manifest checks; writes measurements to JSON")
    s.add_argument("--env-config", default="configs/env.yaml")
    s.add_argument("--output", default="results/validation/physics_checks.json")
    s.set_defaults(fn=_validate_env)

    s = sub.add_parser("benchmark", help="measure training throughput for N workers")
    s.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    s.add_argument("--config", default="configs/randomized.yaml")
    s.add_argument("--ppo-steps", type=int, default=32768)
    s.set_defaults(fn=_benchmark)

    s = sub.add_parser("train", help="train PPO")
    s.add_argument("--config", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--steps", type=int, required=True, help="requested transitions (rounded up to a whole rollout)")
    s.add_argument("--run-id", required=True)
    s.add_argument("--n-envs", type=int, default=None, help="override config n_envs")
    s.add_argument("--runs-dir", default="runs")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=_train)

    s = sub.add_parser("evaluate", help="paired deterministic evaluation")
    s.add_argument("--suite", required=True)
    s.add_argument("--runs", nargs="*", default=[])
    s.add_argument("--baselines", nargs="*", default=[], choices=["scripted", "random"])
    s.add_argument("--output", required=True)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--save-trajectories", choices=["none", "subset", "all"], default="subset")
    s.add_argument("--checkpoint", default="final.zip")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=_evaluate)

    s = sub.add_parser("report", help="aggregate raw episode results into results.csv, plots and REPORT.md")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--runs-dir", default="runs")
    s.add_argument("--manual-labels", default=None, help="optional CSV: episode_id,label")
    s.add_argument("--secondary", default=None, help="a previously generated report directory (e.g. the ~500k-checkpoint analysis) to summarise inside this report")
    s.set_defaults(fn=_report)

    s = sub.add_parser("replay", help="visualise a saved trajectory (use mjpython for the interactive viewer)")
    s.add_argument("--episode", required=True)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--png", default=None, help="write a top-down trajectory PNG (headless)")
    s.add_argument("--gif", default=None, help="write an animated GIF using the offscreen renderer (headless)")
    s.add_argument("--no-viewer", action="store_true")
    s.add_argument("--resimulate-check", action="store_true",
                   help="re-run the recorded actions and report state divergence from the recording (replay itself restores stored states)")
    s.set_defaults(fn=_replay)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
