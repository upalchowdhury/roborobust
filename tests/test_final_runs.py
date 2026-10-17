"""Gate D checks on the ACTUAL final artifacts (skipped when runs/ and results/ are not present, e.g. on a fresh clone)."""
import json

import pytest

from roborobust.config import load_yaml, sha256_file
from conftest import DATA, ROOT

RUNS = [ROOT / "runs" / f"{c}_s{s}" for c in ("nominal", "randomized") for s in range(5)]
HAVE_RUNS = all((r / "meta.json").exists() and (r / "final.zip").exists() for r in RUNS)
HAVE_EVAL = (ROOT / "results" / "weekend" / "eval_meta.json").exists()


@pytest.mark.skipif(not HAVE_RUNS, reason="final runs not present")
def test_final_runs_have_equal_budgets_and_verified_checkpoints():
    metas = {r.name: json.loads((r / "meta.json").read_text()) for r in RUNS}
    assert {m["actual_transitions"] for m in metas.values()} == {1_003_520}
    assert {m["rollout_size"] for m in metas.values()} == {4096}
    for r in RUNS:
        m = metas[r.name]
        assert sha256_file(r / "final.zip") == m["checkpoint_sha256"]          # checkpoint unchanged since training
        assert m["reload_check"]["identical_deterministic_actions"] and m["finite_losses"]
        assert m["train_manifest_sha256"] == sha256_file(DATA / "main_train.jsonl")
        assert m["ppo"]["gamma"] == 0.999
    assert len({m["env_cfg_hash"] for m in metas.values()}) == 1
    for cond in ("nominal", "randomized"):
        assert sorted(m["training_seed"] for n, m in metas.items() if n.startswith(cond)) == [0, 1, 2, 3, 4]
    # the two conditions share every training setting except randomization
    strip = lambda m: {k: v for k, v in m["ppo"].items()}  # noqa: E731
    assert strip(metas["nominal_s0"]) == strip(metas["randomized_s0"])
    assert not metas["nominal_s0"]["randomization"]["enabled"] and metas["randomized_s0"]["randomization"]["enabled"]


@pytest.mark.skipif(not HAVE_EVAL, reason="final evaluation not present")
def test_final_evaluation_is_complete_frozen_and_unmodified():
    meta = json.loads((ROOT / "results" / "weekend" / "eval_meta.json").read_text())
    assert meta["complete"] and meta["expected_episodes"] == meta["recorded_episodes"]
    assert meta["recorded_episodes"] == (10 + 2) * 6 * 50 == 3600
    assert meta["suite_sha256"] == sha256_file(ROOT / "configs" / "test_suite.yaml")   # suite not edited after evaluation
    assert meta["manifest_sha256"] == sha256_file(DATA / "main_test.jsonl")           # held-out manifest not edited
    rows = [json.loads(l) for l in open(ROOT / "results" / "weekend" / "episode_results.jsonl")]
    assert len(rows) == 3600 and len({r["episode_id"] for r in rows}) == 3600
    for pol in ("nominal", "randomized"):
        for seed in range(5):
            assert sum(1 for r in rows if r["policy"] == pol and r["training_seed"] == seed) == 6 * 50
    log = (ROOT / "results" / "test_access_log.jsonl")
    assert log.exists() and len(log.read_text().splitlines()) >= 1


def test_ablation_configs_differ_from_randomized_only_in_the_ablated_factor():
    base = load_yaml(ROOT / "configs" / "randomized.yaml")
    motor, noise = load_yaml(ROOT / "configs" / "motor_only.yaml"), load_yaml(ROOT / "configs" / "noise_only.yaml")
    strip = lambda d: {k: v for k, v in d.items() if k not in ("name", "randomization")}  # noqa: E731
    assert strip(motor) == strip(base) == strip(noise)
    assert motor["randomization"]["motor_mult_range"] == base["randomization"]["motor_mult_range"] and motor["randomization"]["range_noise_std_range"] == [0.0, 0.0]
    assert noise["randomization"]["range_noise_std_range"] == base["randomization"]["range_noise_std_range"] and noise["randomization"]["motor_mult_range"] == [1.0, 1.0]
