"""Provenance helpers: package versions, platform, git state, source-tree hash."""
from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from roborobust.config import PACKAGE_DIR, REPO_ROOT

TRACKED_PACKAGES = [
    "mujoco", "gymnasium", "stable-baselines3", "torch", "numpy", "pandas", "matplotlib",
    "pyyaml", "pytest",
]


def package_versions() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def platform_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "platform": platform.platform(),
    }


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    # `git rev-parse HEAD` prints the literal word "HEAD" (and fails) in a repo with no commits, so verify it.
    commit = run("rev-parse", "--verify", "-q", "HEAD^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return {"commit": None, "dirty": None, "note": "no git commit available (repository has no commits)"}
    return {"commit": commit, "dirty": bool(run("status", "--porcelain"))}


def code_hash() -> str:
    """sha256 over every .py/.xml file in the package (stable order). Works without a commit."""
    h = hashlib.sha256()
    for p in sorted(list(PACKAGE_DIR.glob("*.py")) + list(PACKAGE_DIR.glob("*.xml"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def provenance() -> dict:
    return {
        "packages": package_versions(),
        "platform": platform_info(),
        "git": git_info(),
        "code_hash": code_hash(),
    }
