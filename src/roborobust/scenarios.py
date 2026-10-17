"""Scenario generation, validation and manifests.

A *scenario* is one concrete task instance: room layout (axis-aligned rectangular obstacles),
start pose (including heading) and goal. Scenarios are generated once, before any training, from a
dedicated split-generation seed, and stored as JSONL manifests with SHA-256 hashes.

Design points (see DECISIONS.md):
  * Each main-family scenario owns a distinct layout, so "50 held-out geometries" means 50 layouts.
  * Layouts are canonicalised (sorted, rounded to 5 cm) and must be disjoint across train/val/test.
    On top of exact-hash disjointness we require a minimum parameter-space separation between
    layouts of different splits, so near-duplicates cannot leak from train into test.
  * Feasibility = connectivity of start and goal on an occupancy grid inflated by robot radius plus
    margin. This is an approximate check, NOT a guarantee that the dynamics can solve the task.
  * Rejection sampling is bounded and deterministic given the seed.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roborobust import ENV_VERSION
from roborobust.config import canonical_json, load_yaml, sha256_file


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Obstacle:
    cx: float
    cy: float
    hx: float  # half extent along x (m)
    hy: float  # half extent along y (m)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.hx, self.hy)


@dataclass
class Scenario:
    scenario_id: str
    split: str
    family: str  # "main" | "empty"
    layout_id: str
    start: dict  # {"x","y","yaw"}  (m, m, rad)
    goal: dict  # {"x","y"}
    obstacles: list[Obstacle]
    env_version: str = ENV_VERSION
    meta: dict = field(default_factory=dict)  # analysis metadata, never shown to the policy

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "family": self.family,
            "layout_id": self.layout_id,
            "start": self.start,
            "goal": self.goal,
            "obstacles": [dict(cx=o.cx, cy=o.cy, hx=o.hx, hy=o.hy) for o in self.obstacles],
            "env_version": self.env_version,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            scenario_id=d["scenario_id"],
            split=d["split"],
            family=d.get("family", "main"),
            layout_id=d["layout_id"],
            start=dict(d["start"]),
            goal=dict(d["goal"]),
            obstacles=[Obstacle(**o) for o in d["obstacles"]],
            env_version=d.get("env_version", ENV_VERSION),
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def rect_distance(px, py, o: Obstacle):
    """Euclidean distance from point(s) to an axis-aligned rectangle (0 inside)."""
    dx = np.maximum(np.abs(np.asarray(px) - o.cx) - o.hx, 0.0)
    dy = np.maximum(np.abs(np.asarray(py) - o.cy) - o.hy, 0.0)
    return np.hypot(dx, dy)


def clearance_to_obstacles(px, py, obstacles: list[Obstacle]):
    if not obstacles:
        return np.full(np.shape(px), np.inf)
    return np.min([rect_distance(px, py, o) for o in obstacles], axis=0)


def rect_gap(a: Obstacle, b: Obstacle) -> float:
    dx = max(abs(a.cx - b.cx) - a.hx - b.hx, 0.0)
    dy = max(abs(a.cy - b.cy) - a.hy - b.hy, 0.0)
    return math.hypot(dx, dy)


def segment_hits_expanded_rect(p0, p1, o: Obstacle, inflate: float) -> bool:
    """Slab test: does segment p0->p1 intersect the rectangle expanded by `inflate` on all sides?"""
    lo = (o.cx - o.hx - inflate, o.cy - o.hy - inflate)
    hi = (o.cx + o.hx + inflate, o.cy + o.hy + inflate)
    t0, t1 = 0.0, 1.0
    for axis in (0, 1):
        d = p1[axis] - p0[axis]
        if abs(d) < 1e-12:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False
        else:
            ta, tb = (lo[axis] - p0[axis]) / d, (hi[axis] - p0[axis]) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


# --------------------------------------------------------------------------------------
# Layout canonicalisation
# --------------------------------------------------------------------------------------
def canonical_layout(obstacles: list[Obstacle], quantum: float = 0.05) -> tuple:
    """Order-independent, 5 cm-quantised key. Empty layout -> empty tuple."""
    q = lambda v: int(round(v / quantum))  # noqa: E731
    return tuple(sorted((q(o.cx), q(o.cy), q(o.hx), q(o.hy)) for o in obstacles))


def layout_id_from_key(key: tuple) -> str:
    if not key:
        return "empty"
    return "L" + hashlib.sha1(json.dumps(key).encode()).hexdigest()[:10]


def snap_obstacles(obstacles: list[Obstacle], quantum: float = 0.05) -> list[Obstacle]:
    """Snap geometry onto the canonical grid so the stored geometry IS the canonical geometry."""
    s = lambda v: round(round(v / quantum) * quantum, 4)  # noqa: E731
    return sorted((Obstacle(s(o.cx), s(o.cy), s(o.hx), s(o.hy)) for o in obstacles), key=Obstacle.as_tuple)


def layout_distance(a: list[Obstacle], b: list[Obstacle]) -> float:
    """Min over obstacle matchings of the max absolute parameter difference (m).

    Layouts with different obstacle counts are different task families -> infinite distance.
    """
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.0
    best = math.inf
    for perm in itertools.permutations(range(len(b))):
        worst = max(
            max(abs(x - y) for x, y in zip(a[i].as_tuple(), b[j].as_tuple()))
            for i, j in enumerate(perm)
        )
        best = min(best, worst)
    return best


# --------------------------------------------------------------------------------------
# Occupancy grid + connectivity
# --------------------------------------------------------------------------------------
class OccupancyGrid:
    """Boolean free-space grid inflated by `inflate` (= robot radius + margin) around obstacles
    and inside the walls."""

    def __init__(self, obstacles: list[Obstacle], half_size: float, inflate: float, res: float):
        self.res, self.half, self.inflate = res, half_size, inflate
        self.n = int(round(2 * half_size / res))
        c = -half_size + res * (np.arange(self.n) + 0.5)
        self.xs, self.ys = np.meshgrid(c, c, indexing="ij")  # [ix, iy]
        free = (np.abs(self.xs) <= half_size - inflate) & (np.abs(self.ys) <= half_size - inflate)
        if obstacles:
            free &= clearance_to_obstacles(self.xs, self.ys, obstacles) >= inflate
        self.free = free

    def cell(self, x: float, y: float) -> tuple[int, int]:
        ix = min(max(int((x + self.half) / self.res), 0), self.n - 1)
        iy = min(max(int((y + self.half) / self.res), 0), self.n - 1)
        return ix, iy

    def connected(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        ca, cb = self.cell(*a), self.cell(*b)
        if not (self.free[ca] and self.free[cb]):
            return False
        seen = np.zeros_like(self.free)
        seen[ca] = True
        q = deque([ca])
        while q:
            ix, iy = q.popleft()
            if (ix, iy) == cb:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < self.n and 0 <= jy < self.n and self.free[jx, jy] and not seen[jx, jy]:
                    seen[jx, jy] = True
                    q.append((jx, jy))
        return False

    def shortest_path_length(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """8-connected Dijkstra (no corner cutting). Returns metres, inf if disconnected."""
        ca, cb = self.cell(*a), self.cell(*b)
        if not (self.free[ca] and self.free[cb]):
            return math.inf
        dist = {ca: 0.0}
        heap = [(0.0, ca)]
        diag = math.sqrt(2.0) * self.res
        while heap:
            d, (ix, iy) = heapq.heappop(heap)
            if (ix, iy) == cb:
                return d
            if d > dist.get((ix, iy), math.inf):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < self.n and 0 <= jy < self.n) or not self.free[jx, jy]:
                    continue
                if dx and dy and not (self.free[ix + dx, iy] and self.free[ix, iy + dy]):
                    continue
                nd = d + (diag if dx and dy else self.res)
                if nd < dist.get((jx, jy), math.inf):
                    dist[(jx, jy)] = nd
                    heapq.heappush(heap, (nd, (jx, jy)))
        return math.inf


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
class GenerationError(RuntimeError):
    pass


def _sample_layout(rng: np.random.Generator, fam: dict, half: float) -> list[Obstacle]:
    n_choices = fam["n_obstacles"]
    n = int(rng.choice(n_choices))
    obs: list[Obstacle] = []
    lo, hi = fam["half_size_range"]
    gap = fam["min_gap"]
    for _ in range(50):  # bounded placement attempts per obstacle
        if len(obs) == n:
            break
        hx, hy = rng.uniform(lo, hi, size=2)
        mx, my = half - gap - hx, half - gap - hy  # keep >= min_gap corridor to the walls
        if mx <= 0 or my <= 0:
            continue
        cand = Obstacle(float(rng.uniform(-mx, mx)), float(rng.uniform(-my, my)), float(hx), float(hy))
        cand = snap_obstacles([cand])[0]
        if all(rect_gap(cand, o) >= gap for o in obs):
            obs.append(cand)
    if len(obs) != n:
        raise GenerationError("could not place obstacles")
    return snap_obstacles(obs)


def _sample_free_point(rng, obstacles, half, inflate):
    lim = half - inflate
    for _ in range(100):
        x, y = rng.uniform(-lim, lim, size=2)
        if clearance_to_obstacles(x, y, obstacles) >= inflate:
            return float(x), float(y)
    raise GenerationError("no free point")


def generate_split(
    cfg: dict,
    family: str,
    split: str,
    n: int,
    rng: np.random.Generator,
    other_layouts: list[list[Obstacle]],
    stats: dict,
) -> list[Scenario]:
    """Generate `n` scenarios of `family` for `split`; rejects layouts too close to `other_layouts`."""
    half = cfg["room_half_size"]
    inflate = cfg["robot_radius"] + cfg["inflate_margin"]
    res = cfg["grid_resolution"]
    min_sg = cfg["min_start_goal_dist"]
    fam = cfg["families"][family] if family != "empty" else {}
    sep = cfg["min_layout_separation"]
    max_tries = cfg["max_attempts_per_scenario"]

    scenarios: list[Scenario] = []
    own_layouts: list[list[Obstacle]] = []
    own_keys: set = set()
    rejects = {"layout": 0, "layout_dup": 0, "startgoal": 0, "blocking": 0, "disconnected": 0}

    for i in range(n):
        for attempt in range(max_tries):
            try:
                obstacles = [] if family == "empty" else _sample_layout(rng, fam, half)
            except GenerationError:
                rejects["layout"] += 1
                continue
            key = canonical_layout(obstacles)
            if family != "empty":
                if key in own_keys or any(layout_distance(obstacles, o) < sep for o in own_layouts):
                    rejects["layout_dup"] += 1
                    continue
                if any(layout_distance(obstacles, o) < sep for o in other_layouts):
                    rejects["layout_dup"] += 1
                    continue
            want_blocking = family != "empty" and rng.random() < fam["blocking_fraction"]
            try:
                sx, sy = _sample_free_point(rng, obstacles, half, inflate)
                gx, gy = _sample_free_point(rng, obstacles, half, inflate)
            except GenerationError:
                rejects["startgoal"] += 1
                continue
            if math.hypot(gx - sx, gy - sy) < min_sg:
                rejects["startgoal"] += 1
                continue
            blocked = any(segment_hits_expanded_rect((sx, sy), (gx, gy), o, inflate) for o in obstacles)
            if want_blocking and not blocked:
                rejects["blocking"] += 1
                continue
            grid = OccupancyGrid(obstacles, half, inflate, res)
            if not grid.connected((sx, sy), (gx, gy)):
                rejects["disconnected"] += 1
                continue
            path_len = grid.shortest_path_length((sx, sy), (gx, gy))
            yaw = float(rng.uniform(-math.pi, math.pi))
            sc = Scenario(
                scenario_id=f"{family}_{split}_{i:04d}",
                split=split,
                family=family,
                layout_id=layout_id_from_key(key),
                start={"x": round(sx, 4), "y": round(sy, 4), "yaw": round(yaw, 4)},
                goal={"x": round(gx, 4), "y": round(gy, 4)},
                obstacles=obstacles,
                meta={
                    "n_obstacles": len(obstacles),
                    "straight_line_blocked": bool(blocked),
                    "start_goal_distance": round(math.hypot(gx - sx, gy - sy), 4),
                    "grid_path_length": round(path_len, 3),
                },
            )
            scenarios.append(sc)
            own_layouts.append(obstacles)
            own_keys.add(key)
            break
        else:
            raise GenerationError(
                f"{family}/{split}: could not generate scenario {i} in {max_tries} attempts "
                f"(rejections so far: {rejects}). Relax the family config; do not change the seed to hide this."
            )
    stats[f"{family}_{split}"] = {"n": len(scenarios), "rejections": rejects}
    return scenarios


def generate_all(cfg: dict) -> tuple[dict[str, list[Scenario]], dict]:
    """Generate every manifest. Order test -> val -> train so later splits are checked against
    already-fixed layouts of the earlier ones (train is the largest, hence the most constrained)."""
    stats: dict = {}
    out: dict[str, list[Scenario]] = {}
    seed = cfg["split_seed"]
    layouts_by_split: dict[str, list[list[Obstacle]]] = {}
    for idx, split in enumerate(("test", "val", "train")):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, idx]))
        others = [o for s, ls in layouts_by_split.items() for o in ls if s != split]
        scs = generate_split(cfg, "main", split, cfg["splits"][split], rng, others, stats)
        out[f"main_{split}"] = scs
        layouts_by_split[split] = [s.obstacles for s in scs]
    for idx, split in enumerate(("val", "train")):
        n = cfg["empty_splits"][split]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, idx]))
        out[f"empty_{split}"] = generate_split(cfg, "empty", split, n, rng, [], stats)
    return out, stats


# --------------------------------------------------------------------------------------
# I/O + verification
# --------------------------------------------------------------------------------------
def write_manifests(manifests: dict[str, list[Scenario]], out_dir: Path, cfg: dict, stats: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, scs in manifests.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for s in scs:
                f.write(canonical_json(s.to_dict()) + "\n")
        hashes[path.name] = {"sha256": sha256_file(path), "n": len(scs)}
    summary = {
        "env_version": ENV_VERSION,
        "generator_config": cfg,
        "generation_stats": stats,
        "manifests": hashes,
    }
    with open(out_dir / "manifest_hashes.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def load_manifest(path: str | Path) -> list[Scenario]:
    with open(path) as f:
        return [Scenario.from_dict(json.loads(line)) for line in f if line.strip()]


def manifest_hash(path: str | Path) -> str:
    return sha256_file(path)


def verify_manifests(manifest_dir: Path, cfg: dict) -> list[str]:
    """Independent re-check of everything the spec requires of the stored manifests.
    Returns a list of problems (empty == OK)."""
    problems: list[str] = []
    half = cfg["room_half_size"]
    inflate = cfg["robot_radius"] + cfg["inflate_margin"]
    res = cfg["grid_resolution"]
    recorded = json.loads((manifest_dir / "manifest_hashes.json").read_text())["manifests"]

    sets: dict[str, list[Scenario]] = {}
    for p in sorted(manifest_dir.glob("*.jsonl")):
        if sha256_file(p) != recorded.get(p.name, {}).get("sha256"):
            problems.append(f"{p.name}: hash differs from manifest_hashes.json")
        sets[p.stem] = load_manifest(p)

    ids = [s.scenario_id for scs in sets.values() for s in scs]
    if len(ids) != len(set(ids)):
        problems.append("duplicate scenario_id across manifests")

    for name, scs in sets.items():
        for s in scs:
            if s.env_version != ENV_VERSION:
                problems.append(f"{s.scenario_id}: env_version {s.env_version}")
            sx, sy, gx, gy = s.start["x"], s.start["y"], s.goal["x"], s.goal["y"]
            if clearance_to_obstacles(sx, sy, s.obstacles) < inflate - 1e-9 or max(abs(sx), abs(sy)) > half - inflate:
                problems.append(f"{s.scenario_id}: start inside inflated geometry")
            if clearance_to_obstacles(gx, gy, s.obstacles) < inflate - 1e-9 or max(abs(gx), abs(gy)) > half - inflate:
                problems.append(f"{s.scenario_id}: goal inside inflated geometry")
            if math.hypot(gx - sx, gy - sy) < cfg["min_start_goal_dist"]:
                problems.append(f"{s.scenario_id}: start-goal distance < {cfg['min_start_goal_dist']}")
            grid = OccupancyGrid(s.obstacles, half, inflate, res)
            if not grid.connected((sx, sy), (gx, gy)):
                problems.append(f"{s.scenario_id}: start/goal disconnected on inflated grid")
            if layout_id_from_key(canonical_layout(s.obstacles)) != s.layout_id:
                problems.append(f"{s.scenario_id}: layout_id does not match canonical layout")

    # Layout disjointness across main splits (empty room excluded by construction).
    main = {k: v for k, v in sets.items() if k.startswith("main_")}
    for (ka, a), (kb, b) in itertools.combinations(main.items(), 2):
        ids_a, ids_b = {s.layout_id for s in a}, {s.layout_id for s in b}
        if ids_a & ids_b:
            problems.append(f"{ka} and {kb} share layouts: {sorted(ids_a & ids_b)[:3]}")
        sep = cfg["min_layout_separation"]
        for sa in a:
            for sb in b:
                if layout_distance(sa.obstacles, sb.obstacles) < sep:
                    problems.append(f"near-duplicate layouts {sa.scenario_id} vs {sb.scenario_id}")
    return problems


def load_scenario_config(path: str | Path = "configs/scenarios.yaml") -> dict:
    return load_yaml(path)
