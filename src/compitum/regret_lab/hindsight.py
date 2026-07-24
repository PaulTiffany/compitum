"""Hindsight constrained sequence optimum (tranche 3).

Computes, with perfect foresight of realized consumption, the exact
cumulative-utility-maximizing choice sequence subject to the cumulative
resource ledger -- the comparator against which every online policy's
regret is measured. Exact via memoized search over
``(step, discretized remaining-budget state)``: budgets are generated on a
fixed rational grid (``environment.GRID_UNIT``) and represented internally
as scaled integers, so the search has no floating-point drift and no
approximation while the state space stays within ``max_states``. If a
sequence is long/branchy enough to exceed that bound, this falls back to a
documented greedy policy and reports a conservative optimality gap against
the trivial (always achievable in principle, ignoring every resource
constraint) per-step unconstrained-utility upper bound -- never silently
returning a loose number as if it were exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .environment import GRID_UNIT, DynamicSequence

_SCALE = round(1.0 / GRID_UNIT)


class _StateBudgetExceeded(Exception):
    pass


@dataclass
class HindsightResult:
    value: float
    choices: List[str]
    exact: bool
    optimality_gap: float
    state_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "choices": list(self.choices),
            "exact": self.exact,
            "optimality_gap": self.optimality_gap,
            "state_count": self.state_count,
        }


def _scaled(values: Dict[str, float], resource_names: Tuple[str, ...]) -> Tuple[int, ...]:
    return tuple(int(round(values[r] * _SCALE)) for r in resource_names)


def _greedy_fallback(seq: DynamicSequence) -> Tuple[float, List[str]]:
    remaining = dict(seq.initial_budget)
    total = 0.0
    choices: List[str] = []
    for case in seq.cases:
        ranked = sorted(case.base_utility, key=lambda m: case.base_utility[m], reverse=True)
        picked = None
        for m in ranked:
            consumption = case.realized_consumption[m]
            if all(remaining[r] - consumption[r] >= -1e-9 for r in seq.resource_names):
                picked = m
                break
        if picked is None:
            choices.append("defer")
        else:
            for r in seq.resource_names:
                remaining[r] -= case.realized_consumption[picked][r]
            total += case.base_utility[picked]
            choices.append(picked)
        for r in seq.resource_names:
            remaining[r] += case.replenishment[r]
    return total, choices


def compute_hindsight_optimum(seq: DynamicSequence, max_states: int = 200_000) -> HindsightResult:
    resource_names = seq.resource_names
    steps = len(seq.cases)
    initial_state = _scaled(seq.initial_budget, resource_names)

    memo: Dict[Tuple[int, Tuple[int, ...]], Tuple[float, List[str]]] = {}
    state_count = 0

    def recurse(t: int, state: Tuple[int, ...]) -> Tuple[float, List[str]]:
        nonlocal state_count
        if t == steps:
            return 0.0, []
        key = (t, state)
        if key in memo:
            return memo[key]
        state_count += 1
        if state_count > max_states:
            raise _StateBudgetExceeded()

        case = seq.cases[t]
        replen = _scaled(case.replenishment, resource_names)

        defer_next = tuple(state[i] + replen[i] for i in range(len(resource_names)))
        rest_value, rest_choices = recurse(t + 1, defer_next)
        best_value, best_choice, best_rest = rest_value, "defer", rest_choices

        for m in case.base_utility:
            cons = _scaled(case.realized_consumption[m], resource_names)
            if any(state[i] - cons[i] < 0 for i in range(len(resource_names))):
                continue
            next_state = tuple(state[i] - cons[i] + replen[i] for i in range(len(resource_names)))
            value, choices = recurse(t + 1, next_state)
            total = case.base_utility[m] + value
            if total > best_value:
                best_value, best_choice, best_rest = total, m, choices

        result = (best_value, [best_choice] + best_rest)
        memo[key] = result
        return result

    try:
        value, choices = recurse(0, initial_state)
        return HindsightResult(
            value=value, choices=choices, exact=True, optimality_gap=0.0, state_count=state_count
        )
    except _StateBudgetExceeded:
        greedy_value, greedy_choices = _greedy_fallback(seq)
        upper_bound = sum(max(case.base_utility.values()) for case in seq.cases)
        gap = (upper_bound - greedy_value) / upper_bound if upper_bound > 0 else 0.0
        return HindsightResult(
            value=greedy_value,
            choices=greedy_choices,
            exact=False,
            optimality_gap=gap,
            state_count=state_count,
        )
