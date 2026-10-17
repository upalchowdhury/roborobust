"""Gate A: physics. Thin wrappers over roborobust.validation so CLI and tests share one definition."""
import pytest

from roborobust import validation as V


@pytest.mark.parametrize("check", V.ALL_PHYSICS_CHECKS, ids=lambda f: f.__name__)
def test_physics_check(cfg, check):
    res = check(cfg)
    assert res["passed"], f"{res['name']} failed: {res['measured']}"


def test_weak_motor_effect_is_strongest_under_demanding_command(cfg):
    """Documents WHERE the intervention bites: negligible at half command, large at full command."""
    m = V.check_weak_motor_effect(cfg)["measured"]
    full = abs(m["demanding_full_command"]["0.6"]["lateral_m"])
    half = abs(m["light_half_command"]["0.6"]["lateral_m"])
    assert full > 5 * half
