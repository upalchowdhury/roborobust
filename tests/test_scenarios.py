"""Gate B: manifests, disjointness, validity, connectivity, bounded deterministic generation."""
import copy
import itertools
import json

import pytest

from roborobust.config import sha256_file
from roborobust.scenarios import (
    GenerationError, Obstacle, OccupancyGrid, canonical_layout, generate_all, layout_distance,
    load_manifest, load_scenario_config, verify_manifests,
)
from conftest import DATA, ROOT


@pytest.fixture(scope="module")
def scfg():
    return load_scenario_config(ROOT / "configs" / "scenarios.yaml")


def test_committed_manifests_verify(scfg):
    assert verify_manifests(DATA, scfg) == []


def test_manifest_sizes_and_hashes(scfg):
    rec = json.loads((DATA / "manifest_hashes.json").read_text())["manifests"]
    assert {k: v["n"] for k, v in rec.items()} == {
        "main_train.jsonl": 200, "main_val.jsonl": 40, "main_test.jsonl": 50,
        "empty_train.jsonl": 200, "empty_val.jsonl": 20}
    for name, info in rec.items():
        assert sha256_file(DATA / name) == info["sha256"]


def test_main_layouts_disjoint_across_splits_even_near_duplicates(scfg):
    splits = {s: load_manifest(DATA / f"main_{s}.jsonl") for s in ("train", "val", "test")}
    for a, b in itertools.combinations(splits, 2):
        keys_a = {canonical_layout(s.obstacles) for s in splits[a]}
        keys_b = {canonical_layout(s.obstacles) for s in splits[b]}
        assert not keys_a & keys_b
        d = min(layout_distance(x.obstacles, y.obstacles) for x in splits[a] for y in splits[b])
        assert d >= scfg["min_layout_separation"]


def test_every_main_scenario_has_its_own_layout():
    for s in ("train", "val", "test"):
        scs = load_manifest(DATA / f"main_{s}.jsonl")
        assert len({x.layout_id for x in scs}) == len(scs)


def test_empty_room_family_is_labelled_separately():
    for s in ("train", "val"):
        scs = load_manifest(DATA / f"empty_{s}.jsonl")
        assert all(x.family == "empty" and x.layout_id == "empty" and not x.obstacles for x in scs)


def test_starts_goals_valid_and_far_enough(scfg):
    from roborobust.scenarios import clearance_to_obstacles
    import math
    r = scfg["robot_radius"] + scfg["inflate_margin"]
    for name in ("main_test", "main_val", "main_train", "empty_val"):
        for s in load_manifest(DATA / f"{name}.jsonl"):
            for p in (s.start, s.goal):
                assert clearance_to_obstacles(p["x"], p["y"], s.obstacles) >= r - 1e-9
                assert max(abs(p["x"]), abs(p["y"])) <= scfg["room_half_size"] - r + 1e-9
            assert math.hypot(s.goal["x"] - s.start["x"], s.goal["y"] - s.start["y"]) >= 1.0
            assert -math.pi <= s.start["yaw"] <= math.pi


def test_headings_are_randomised():
    import numpy as np
    yaws = np.array([s.start["yaw"] for s in load_manifest(DATA / "main_train.jsonl")])
    assert yaws.std() > 1.5 and yaws.min() < -2.5 and yaws.max() > 2.5


def test_connectivity_check_rejects_disconnected_rooms():
    wall = [Obstacle(0.0, 0.0, 3.0, 0.3)]  # spans wall to wall
    g = OccupancyGrid(wall, 3.0, 0.30, 0.05)
    assert not g.connected((0.0, -2.0), (0.0, 2.0))
    assert g.shortest_path_length((0.0, -2.0), (0.0, 2.0)) == float("inf")
    g2 = OccupancyGrid([Obstacle(0.0, 0.0, 1.0, 0.3)], 3.0, 0.30, 0.05)
    assert g2.connected((0.0, -2.0), (0.0, 2.0))
    assert 4.0 < g2.shortest_path_length((0.0, -2.0), (0.0, 2.0)) < 6.5  # must go around the block


def test_generation_is_deterministic_and_seed_dependent(scfg):
    small = copy.deepcopy(scfg)
    small["splits"] = {"train": 6, "val": 3, "test": 3}
    small["empty_splits"] = {"train": 3, "val": 2}
    a, _ = generate_all(small)
    b, _ = generate_all(small)
    assert {k: [s.to_dict() for s in v] for k, v in a.items()} == {k: [s.to_dict() for s in v] for k, v in b.items()}
    small["split_seed"] += 1
    c, _ = generate_all(small)
    assert [s.to_dict() for s in a["main_test"]] != [s.to_dict() for s in c["main_test"]]


def test_rejection_sampling_is_bounded(scfg):
    impossible = copy.deepcopy(scfg)
    impossible["splits"] = {"train": 2, "val": 1, "test": 1}
    impossible["empty_splits"] = {"train": 1, "val": 1}
    impossible["families"]["main"]["min_gap"] = 2.9      # no room for any obstacle
    impossible["max_attempts_per_scenario"] = 20
    with pytest.raises(GenerationError):
        generate_all(impossible)


def test_corrupted_manifest_is_detected(scfg, tmp_path):
    import shutil
    d = tmp_path / "m"; shutil.copytree(DATA, d)
    p = d / "main_test.jsonl"
    p.write_text(p.read_text().replace('"yaw":', '"yaw": 0.0, "_":', 1))
    assert any("hash differs" in x for x in verify_manifests(d, scfg))
