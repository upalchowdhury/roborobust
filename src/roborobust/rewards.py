"""Training reward: terminal bonus/penalty + step cost + potential-based progress shaping.

    r = +10 [success] - 10 [collision | tip | out-of-bounds] - 0.01 + alpha * (gamma * Phi(s') - Phi(s))
    Phi(s) = -goal_distance / room_diagonal

Policy-invariance assumptions of potential-based shaping (Ng, Harada, Russell 1999) -- these are
assumptions, not guarantees against reward hacking:
  * gamma in the shaping term equals the learner's discount (train.py asserts it);
  * Phi of a genuine terminal next state is 0 (applied here for success AND failures);
  * on time-limit truncation the TRUE next potential is kept (the state is not terminal, the
    value function bootstraps through it);
  * the potential depends on state only. Here it uses true goal distance, while the policy sees a
    partially observed state, so invariance is argued for the underlying MDP, not the POMDP.
The reward never decides success: verifier.py does.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from roborobust.config import RewardCfg
from roborobust.verifier import Outcome

COMPONENT_NAMES = ("step_cost", "shaping", "success_bonus", "failure_penalty", "total")


@dataclass(frozen=True)
class RewardComponents:
    step_cost: float
    shaping: float
    success_bonus: float
    failure_penalty: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self)


class RewardFn:
    def __init__(self, cfg: RewardCfg, room_diagonal: float):
        self.cfg = cfg
        self.diag = room_diagonal

    def potential(self, goal_distance: float) -> float:
        return -goal_distance / self.diag

    def compute(self, prev_distance: float, next_distance: float, outcome: Outcome) -> RewardComponents:
        c = self.cfg
        phi = self.potential(prev_distance)
        phi_next = 0.0 if outcome.terminated else self.potential(next_distance)
        shaping = c.shaping_alpha * (c.gamma * phi_next - phi)
        step_cost = -c.step_cost
        bonus = c.success_bonus if outcome.success else 0.0
        penalty = -c.failure_penalty if (outcome.terminated and not outcome.success) else 0.0
        return RewardComponents(step_cost, shaping, bonus, penalty, step_cost + shaping + bonus + penalty)
