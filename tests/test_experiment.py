"""Gate C/D: experiment integrity (paired design, equal budgets, no test-time learning, honest bookkeeping)."""
import json
import math

import numpy as np
import pytest
import yaml

from roborobust.config import load_env_config, load_yaml
from roborobust.scenarios import load_manifest
from conftest import DATA, ROOT


def cfgs():
    return {n: load_yaml(ROOT / "configs" / f"{n}.yaml") for n in ("nominal", "randomized")}


def test_conditions_differ_only_in_randomization():
    c = cfgs()
    strip = lambda d: {k: v for k, v in d.items() if k not in ("name", "randomization")}  # noqa: E731
    assert strip(c["nominal"]) == strip(c["randomized"])
    assert c["nominal"]["randomization"]["enabled"] is False
    r = c["randomized"]["randomization"]
    assert r["enabled"] is True and r["motor_mult_range"] == [0.8, 1.2] and r["range_noise_std_range"] == [0.0, 0.03]


def test_ppo_settings_and_budget_constraints():
    c = cfgs()["nominal"]
    p = c["ppo"]
    assert (p["gamma"], p["gae_lambda"], p["clip_range"], p["n_epochs"], p["batch_size"], p["n_steps"], p["ent_coef"],
            p["learning_rate"]) == (0.999, 0.95, 0.2, 10, 256, 1024, 0.0, 3e-4)
    assert p["net_arch"] == [64, 64] and p["policy"] == "MlpPolicy" and c["device"] == "cpu"
    assert (p["n_steps"] * c["n_envs"]) % p["batch_size"] == 0
    assert p["gamma"] == load_env_config(ROOT / "configs" / "env.yaml")[0].reward.gamma


def test_gamma_mismatch_is_rejected(tmp_path):
    from roborobust.train import train
    with pytest.raises(ValueError, match="gamma"):
        train(ROOT / "configs" / "nominal.yaml", 0, 1024, "bad", runs_dir=tmp_path, n_envs=1, extra_overrides={"ppo": {"gamma": 0.9}})


def test_indivisible_rollout_is_rejected(tmp_path):
    from roborobust.train import train
    with pytest.raises(ValueError, match="divisible"):
        train(ROOT / "configs" / "nominal.yaml", 0, 1000, "bad", runs_dir=tmp_path, n_envs=1, extra_overrides={"ppo": {"batch_size": 300}})


def test_test_suite_is_frozen_to_the_specified_conditions():
    s = load_yaml(ROOT / "configs" / "test_suite.yaml")
    want = {"nominal": (1.0, 1.0, 0.0), "within_support": (0.85, 1.10, 0.02), "weak_left": (0.60, 1.0, 0.0),
            "weak_right": (1.0, 0.60, 0.0), "noisy_sensing": (1.0, 1.0, 0.06), "combined_stress": (0.60, 1.0, 0.06)}
    assert {k: (v["left"], v["right"], v["noise_std"]) for k, v in s["conditions"].items()} == want
    assert s["primary_condition"] == "combined_stress" and s["manifest"].endswith("main_test.jsonl")
    assert len(load_manifest(ROOT / s["manifest"])) == 50


# ---------------------------------------------------------------- tiny end-to-end pipeline
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Two tiny paired runs (nominal / randomized), a tiny 3-scenario suite, and evaluation results."""
    from roborobust.evaluate import evaluate
    from roborobust.train import train

    root = tmp_path_factory.mktemp("tiny")
    runs = root / "runs"
    over = {"checkpoint_every": 10 ** 9}
    train(ROOT / "configs" / "nominal.yaml", 0, 2048, "nominal_s0", runs_dir=runs, n_envs=1, extra_overrides=over)
    train(ROOT / "configs" / "randomized.yaml", 0, 2048, "randomized_s0", runs_dir=runs, n_envs=1, extra_overrides=over)
    scs = load_manifest(DATA / "empty_val.jsonl")[:3]
    man = root / "mini.jsonl"
    man.write_text("\n".join(json.dumps(s.to_dict()) for s in scs) + "\n")
    suite = {"suite_name": "mini", "manifest": str(man), "env_config": str(ROOT / "configs" / "env.yaml"), "eval_seed": 5,
             "random_baseline_seed": 6, "primary_condition": "combined_stress",
             "conditions": {"nominal": {"left": 1.0, "right": 1.0, "noise_std": 0.0},
                            "combined_stress": {"left": 0.6, "right": 1.0, "noise_std": 0.06}},
             "trajectories": {"pinned_scenarios": 1}}
    sp = root / "suite.yaml"
    sp.write_text(yaml.safe_dump(suite))
    return {"root": root, "runs": runs, "suite": sp, "manifest": man, "scenarios": scs}


@pytest.mark.slow
def test_equal_budgets_and_reload_gate(tiny):
    m = {n: json.loads((tiny["runs"] / n / "meta.json").read_text()) for n in ("nominal_s0", "randomized_s0")}
    assert m["nominal_s0"]["actual_transitions"] == m["randomized_s0"]["actual_transitions"] == 2048
    for v in m.values():
        assert v["reload_check"]["identical_deterministic_actions"] is True and v["finite_losses"] is True
        assert len(v["checkpoint_sha256"]) == 64 and v["packages"]["mujoco"] and v["platform"]["machine"]
    assert m["nominal_s0"]["randomization"]["enabled"] is False and m["randomized_s0"]["randomization"]["enabled"] is True
    assert m["nominal_s0"]["train_manifest_sha256"] == m["randomized_s0"]["train_manifest_sha256"]


@pytest.mark.slow
def test_train_refuses_silent_overwrite(tiny):
    from roborobust.train import train
    with pytest.raises(FileExistsError):
        train(ROOT / "configs" / "nominal.yaml", 0, 1024, "nominal_s0", runs_dir=tiny["runs"], n_envs=1)


@pytest.mark.slow
def test_evaluation_is_complete_paired_and_does_not_touch_weights(tiny):
    from roborobust.analyze import integrity_checks, load_results
    from roborobust.evaluate import PPOPolicy, evaluate

    before = {n: PPOPolicy(tiny["runs"] / n).weights_hash() for n in ("nominal_s0", "randomized_s0")}
    out = tiny["root"] / "out"
    evaluate(tiny["suite"], [str(tiny["runs"] / "nominal_s0"), str(tiny["runs"] / "randomized_s0")], ["scripted", "random"], out, workers=1)
    after = {n: PPOPolicy(tiny["runs"] / n).weights_hash() for n in ("nominal_s0", "randomized_s0")}
    assert before == after  # no test-time learning, no normalisation statistics updated

    df, meta = load_results(out)
    assert len(df) == meta["expected_episodes"] == 4 * 2 * 3 and meta["complete"]
    assert integrity_checks(df, meta)["ok"]
    # pairing: every policy saw the same noise seed for each (scenario, condition)
    assert (df.groupby(["scenario_id", "condition"]).noise_seed.nunique() == 1).all()
    # the condition's domain parameters are what was actually applied and logged
    comb = df[df.condition == "combined_stress"]
    assert (comb.p_left_motor_mult == 0.6).all() and (comb.p_right_motor_mult == 1.0).all() and (comb.p_range_noise_std == 0.06).all()
    nom = df[df.condition == "nominal"]
    assert (nom.p_left_motor_mult == 1.0).all() and (nom.p_range_noise_std == 0.0).all()
    # every record carries the hashes the spec requires
    for col in ("manifest_sha256", "env_cfg_hash", "eval_seed", "terminal_reason", "path_length_m", "return", "reward_components"):
        assert df[col].notna().all()
    assert df[df.policy.isin(["nominal", "randomized"])].checkpoint_sha256.notna().all()
    # deterministic: re-running yields identical outcomes
    out2 = tiny["root"] / "out2"
    evaluate(tiny["suite"], [str(tiny["runs"] / "nominal_s0")], [], out2, workers=1)
    d2, _ = load_results(out2)
    a = df[df.policy == "nominal"].sort_values("episode_id")[["episode_id", "terminal_reason", "steps", "path_length_m"]].reset_index(drop=True)
    b = d2.sort_values("episode_id")[["episode_id", "terminal_reason", "steps", "path_length_m"]].reset_index(drop=True)
    assert a.equals(b)
    with pytest.raises(FileExistsError):
        evaluate(tiny["suite"], [], ["scripted"], out, workers=1)


@pytest.mark.slow
def test_report_pipeline_end_to_end_and_reconciles(tiny):
    from roborobust.analyze import report
    out = tiny["root"] / "out"
    rep = tiny["root"] / "rep"
    report(out, rep, runs_dir=tiny["runs"])
    for f in ("REPORT.md", "results.csv", "comparison.png", "learning_curves.png", "episode_results.jsonl", "failure_examples.png"):
        assert (rep / f).exists(), f
    text = (rep / "REPORT.md").read_text()
    assert "PRELIMINARY" in text and "Simulation only" in text and "Reproducibility" in text
    import pandas as pd
    res = pd.read_csv(rep / "results.csv")
    raw = pd.read_json(out / "episode_results.jsonl", lines=True)
    for _, r in res.iterrows():
        sel = raw[(raw.policy == r.policy) & (raw.condition == r.condition)]
        if not math.isnan(r.training_seed):
            sel = sel[sel.training_seed == r.training_seed]
        assert r.successes == sel.success.sum() and r.n_episodes == len(sel)
        assert r.successes + r.collisions + r.timeouts + r.other_failures == r.n_episodes


@pytest.mark.slow
def test_report_refuses_incomplete_or_tampered_results(tiny):
    from roborobust.analyze import report
    src = tiny["root"] / "out"
    bad = tiny["root"] / "bad"
    bad.mkdir()
    lines = (src / "episode_results.jsonl").read_text().splitlines()
    (bad / "episode_results.jsonl").write_text("\n".join(lines[:-3]) + "\n")  # drop episodes
    (bad / "eval_meta.json").write_text((src / "eval_meta.json").read_text())
    with pytest.raises(RuntimeError, match="integrity"):
        report(bad, tiny["root"] / "rep_bad", runs_dir=tiny["runs"])


@pytest.mark.slow
def test_test_split_access_is_logged(tiny, tmp_path, monkeypatch):
    from roborobust.evaluate import evaluate
    monkeypatch.chdir(tmp_path)
    scs = load_manifest(DATA / "main_test.jsonl")[:2]
    man = tmp_path / "t.jsonl"
    man.write_text("\n".join(json.dumps(s.to_dict()) for s in scs) + "\n")
    suite = yaml.safe_load(open(tiny["suite"]))
    suite["manifest"] = str(man)
    sp = tmp_path / "s.yaml"
    sp.write_text(yaml.safe_dump(suite))
    evaluate(sp, [], ["scripted"], tmp_path / "o", workers=1)
    log = (tmp_path / "results" / "test_access_log.jsonl").read_text().splitlines()
    assert len(log) == 1 and json.loads(log[0])["baselines"] == ["scripted"]
    # a validation-split evaluation does not touch the log
    evaluate(tiny["suite"], [], ["scripted"], tmp_path / "o2", workers=1)
    assert len((tmp_path / "results" / "test_access_log.jsonl").read_text().splitlines()) == 1
