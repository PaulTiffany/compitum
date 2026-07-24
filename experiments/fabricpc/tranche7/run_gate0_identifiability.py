"""Tranche 7, Gate 0: benchmark identifiability.

Runs under plain ``.venv`` (no FabricPC/JAX needed -- pure regret_lab).
Per docs/adr/0009-belief-sensitive-shadow-charge-validation.md: before
training or evaluating FabricPC, prove the belief-sensitive environment
actually has reachable states where the Bellman-optimal action depends
on belief. Tranche 6.5's environment did not (confirmed both by a direct
scan and by every arm tying at zero regret); this script is the
analogous scan for the corrected environment, run BEFORE any learned
model.

First pass (payoff/budget tuning only, 8 configs): found genuine
belief-dependent decision boundaries (56-73 reachable states out of
77-92 had them) but the exact-belief policy tied EXACTLY with
fixed-prior and shuffled controls in almost every config. Diagnosed
directly (inspecting real belief trajectories): tranche 6's fixed
observation informativeness (``P_OPPORTUNITY[HIGH]=0.35`` vs
``[NORMAL]=0.05``) makes a single "opportunity available" observation so
decisive that a naive one-shot read already lands on the same side of
the decision boundary as several steps of accumulated exact belief,
regardless of history. This second pass fixes payoff/budget at
reasonable first-pass values and instead varies observation
informativeness and regime persistence -- exactly the dimensions the
authorizing brief named alongside payoff/budget -- selected purely by
belief-boundary occupancy and the exact-belief policy's margin over
belief-blind controls, never by FabricPC/ridge/backprop performance
(none of which are even trained here).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compitum.regret_lab.belief_action_pricing_v2 import (  # noqa: E402
    ExactBeliefEstimatorV2,
    run_shadow_charge_policy_v2,
)
from compitum.regret_lab.belief_bellman_v2 import BeliefSensitiveBellmanOracle  # noqa: E402
from compitum.regret_lab.belief_online_optimum_v2 import (  # noqa: E402
    online_optimum_as_hindsight_result_v2,
)
from compitum.regret_lab.belief_pricing import LookupBeliefEstimator  # noqa: E402
from compitum.regret_lab.belief_regime import INITIAL_BELIEF, STEPS  # noqa: E402
from compitum.regret_lab.belief_regime_v2 import generate_belief_dataset_v2  # noqa: E402
from compitum.regret_lab.metrics import (  # noqa: E402
    bootstrap_ci,
    paired_regret_deltas,
    regret_metrics,
)

ARTIFACTS = REPO_ROOT / "experiments" / "fabricpc" / "tranche7" / "artifacts"

# Payoff/budget frozen at reasonable first-pass values (budget=6.0 showed
# far higher boundary occupancy than 8.0 in every first-pass config);
# this pass's tiny grid varies only observation informativeness and
# regime persistence, the two dimensions the first pass's direct
# diagnostic showed were the actual bottleneck.
U_NORMAL = 1.0
U_HIGH = 8.0
INITIAL_BUDGET = 6.0
P_OPPORTUNITY_GRID: Tuple[Tuple[float, float], ...] = (
    (0.05, 0.35),  # tranche 6's original gap (0.30) -- first-pass baseline
    (0.15, 0.25),  # narrow gap (0.10): a single observation is far less decisive alone
)
TRANSITION_GRID: Tuple[Tuple[float, float], ...] = (
    (0.2, 0.6),  # tranche 6's original persistence
    (0.3, 0.85),  # stickier: regime (and therefore belief) persists longer
)
DIAGNOSTIC_SEED = 9001  # separate from the eventual pilot's train/val/test seeds
N_DIAGNOSTIC_SEQUENCES = 30
BELIEF_GRID = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 41))
BOUNDARY_DISTANCE = 0.1
MIN_OCCUPANCY_FRACTION = 0.10


def enumerate_reachable_states(
    oracle: BeliefSensitiveBellmanOracle, steps: int, initial_budget: float, initial_belief: float
) -> List[Tuple[int, float]]:
    """Populates the oracle's memo by computing the value function once
    from the true initial state -- since that recursion branches over
    every feasible action at every step, the memo then contains every
    (remaining_steps, budget) pair reachable under ANY action sequence,
    not just one specific policy's path."""
    oracle.value(steps, initial_budget, initial_belief)
    seen = {(remaining_steps, budget) for (remaining_steps, budget, _belief) in oracle._value_memo}
    return sorted(seen)


def scan_belief_boundaries(
    oracle: BeliefSensitiveBellmanOracle, remaining_steps: int, budget: float
) -> Dict[str, Any]:
    """For one reachable (remaining_steps, budget) state, the exact
    Bellman-optimal action at every belief in ``BELIEF_GRID``, for both
    observation branches, and the interior belief thresholds where it
    changes."""
    result: Dict[str, Any] = {}
    for observed in (False, True):
        actions = [
            oracle.best_action_given_observation(remaining_steps, budget, b, observed)[0]
            for b in BELIEF_GRID
        ]
        transitions = [
            BELIEF_GRID[i] for i in range(len(actions) - 1) if actions[i] != actions[i + 1]
        ]
        result[str(observed)] = {"actions": actions, "transition_beliefs": transitions}
    return result


def build_boundary_map(
    oracle: BeliefSensitiveBellmanOracle, reachable_states: List[Tuple[int, float]]
) -> Dict[Tuple[int, float], Dict[str, Any]]:
    return {state: scan_belief_boundaries(oracle, *state) for state in reachable_states}


def _nearest_boundary_distance(belief: float, transition_beliefs: List[float]) -> float:
    if not transition_beliefs:
        return float("inf")
    return min(abs(belief - t) for t in transition_beliefs)


def evaluate_boundary_occupancy(
    oracle: BeliefSensitiveBellmanOracle,
    boundary_map: Dict[Tuple[int, float], Dict[str, Any]],
    dataset: List[Tuple[Any, List[int], List[float], List[float]]],
    exact_kwargs: Dict[str, float],
) -> Dict[str, Any]:
    near_boundary = 0
    total = 0
    for seq, _, _, _ in dataset:
        estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **exact_kwargs["transition"])
        _, _, traces = run_shadow_charge_policy_v2(
            seq, oracle, estimator, u_normal=U_NORMAL, u_high=U_HIGH, **exact_kwargs["scoring"]
        )
        total_steps = len(seq.cases)
        for t, trace in enumerate(traces):
            remaining_steps = total_steps - t
            budget_key = round(trace.remaining_budget_before / 0.5) * 0.5
            state_key = (remaining_steps, budget_key)
            entry = boundary_map.get(state_key)
            total += 1
            if entry is None:
                continue
            transitions = entry[str(trace.observation)]["transition_beliefs"]
            distance = _nearest_boundary_distance(trace.filtered_belief_value, transitions)
            if distance <= BOUNDARY_DISTANCE:
                near_boundary += 1
    return {
        "near_boundary_steps": near_boundary,
        "total_steps": total,
        "occupancy_fraction": near_boundary / total if total else 0.0,
    }


def _belief_blind_controls(
    exact_beliefs: List[List[float]], seed: int
) -> Dict[str, List[List[float]]]:
    fixed_prior = [[0.5] * len(b) for b in exact_beliefs]
    inverted = [[1.0 - x for x in b] for b in exact_beliefs]
    rng = np.random.default_rng(seed)
    shuffled = []
    for b in exact_beliefs:
        arr = list(b)
        rng.shuffle(arr)
        shuffled.append(arr)
    return {"fixed_prior": fixed_prior, "inverted": inverted, "shuffled": shuffled}


def evaluate_config(
    p_opportunity_normal: float,
    p_opportunity_high: float,
    transition_normal_to_high: float,
    transition_high_to_high: float,
) -> Dict[str, Any]:
    oracle = BeliefSensitiveBellmanOracle(
        u_normal_opportunity=U_NORMAL,
        u_high_opportunity=U_HIGH,
        p_opportunity_normal=p_opportunity_normal,
        p_opportunity_high=p_opportunity_high,
        transition_normal_to_high=transition_normal_to_high,
        transition_high_to_high=transition_high_to_high,
    )
    reachable = enumerate_reachable_states(oracle, STEPS, INITIAL_BUDGET, INITIAL_BELIEF)
    boundary_map = build_boundary_map(oracle, reachable)
    n_boundary_states = sum(
        1
        for entry in boundary_map.values()
        if entry["False"]["transition_beliefs"] or entry["True"]["transition_beliefs"]
    )

    dataset = generate_belief_dataset_v2(
        seed=DIAGNOSTIC_SEED,
        n_sequences=N_DIAGNOSTIC_SEQUENCES,
        initial_budget=INITIAL_BUDGET,
        u_normal=U_NORMAL,
        u_high=U_HIGH,
        p_opportunity_normal=p_opportunity_normal,
        p_opportunity_high=p_opportunity_high,
        transition_normal_to_high=transition_normal_to_high,
        transition_high_to_high=transition_high_to_high,
        id_prefix="gate0",
    )
    sequences = [d[0] for d in dataset]

    transition_kwargs = {
        "p_opportunity_normal": p_opportunity_normal,
        "p_opportunity_high": p_opportunity_high,
        "transition_normal_to_high": transition_normal_to_high,
        "transition_high_to_high": transition_high_to_high,
    }
    exact_kwargs = {"transition": transition_kwargs, "scoring": transition_kwargs}

    occupancy = evaluate_boundary_occupancy(oracle, boundary_map, dataset, exact_kwargs)

    online_optimum = {
        seq.sequence_id: online_optimum_as_hindsight_result_v2(
            seq, oracle, INITIAL_BELIEF, **transition_kwargs
        )
        for seq in sequences
    }

    exact_results = {}
    exact_beliefs_per_seq = []
    for seq in sequences:
        estimator = ExactBeliefEstimatorV2(belief=INITIAL_BELIEF, **transition_kwargs)
        result, _, traces = run_shadow_charge_policy_v2(
            seq, oracle, estimator, u_normal=U_NORMAL, u_high=U_HIGH, **transition_kwargs
        )
        exact_results[seq.sequence_id] = result
        exact_beliefs_per_seq.append([t.filtered_belief_value for t in traces])

    controls = _belief_blind_controls(exact_beliefs_per_seq, seed=DIAGNOSTIC_SEED + 1)
    control_results: Dict[str, Dict[str, Any]] = {}
    for name, beliefs_per_seq in controls.items():
        results = {}
        for seq, beliefs in zip(sequences, beliefs_per_seq):
            initial = beliefs[0] if beliefs else INITIAL_BELIEF
            lookup = LookupBeliefEstimator(beliefs=beliefs, initial_belief=initial)
            result, _, _ = run_shadow_charge_policy_v2(
                seq, oracle, lookup, u_normal=U_NORMAL, u_high=U_HIGH, **transition_kwargs
            )
            results[seq.sequence_id] = result
        control_results[name] = results

    exact_vs_controls = {}
    for name, results in control_results.items():
        delta = paired_regret_deltas(
            list(exact_results.values()), list(results.values()), online_optimum
        )
        ci = bootstrap_ci(delta)
        # Gate 0 checks direction ("the exact-belief policy has lower
        # regret than..."), not full statistical significance -- that
        # rigor belongs to the actual pilot's larger held-out test set
        # (task 7.3), not this bounded, 30-sequence feasibility screen.
        exact_vs_controls[name] = {"delta": ci, "exact_beats_control": ci["mean"] < 0.0}

    metrics_exact = regret_metrics(list(exact_results.values()), online_optimum)

    config_result = {
        "p_opportunity_normal": p_opportunity_normal,
        "p_opportunity_high": p_opportunity_high,
        "transition_normal_to_high": transition_normal_to_high,
        "transition_high_to_high": transition_high_to_high,
        "u_normal": U_NORMAL,
        "u_high": U_HIGH,
        "initial_budget": INITIAL_BUDGET,
        "n_reachable_states": len(reachable),
        "n_boundary_states": n_boundary_states,
        "boundary_state_fraction": n_boundary_states / len(reachable) if reachable else 0.0,
        "occupancy": occupancy,
        "exact_mean_regret_vs_online_optimum": metrics_exact["mean_regret"],
        "exact_vs_belief_blind_controls": exact_vs_controls,
        "meets_min_occupancy": occupancy["occupancy_fraction"] >= MIN_OCCUPANCY_FRACTION,
        "exact_beats_all_controls": all(
            v["exact_beats_control"] for v in exact_vs_controls.values()
        ),
    }
    config_result["passes_gate0"] = bool(
        n_boundary_states > 0
        and config_result["meets_min_occupancy"]
        and config_result["exact_beats_all_controls"]
    )
    return config_result


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    grid_results = []
    for p_normal, p_high in P_OPPORTUNITY_GRID:
        for t_n2h, t_h2h in TRANSITION_GRID:
            print(
                f"evaluating p_opportunity=({p_normal},{p_high}) "
                f"transition=({t_n2h},{t_h2h})..."
            )
            grid_results.append(evaluate_config(p_normal, p_high, t_n2h, t_h2h))

    passing = [r for r in grid_results if r["passes_gate0"]]
    report: Dict[str, Any] = {
        "schema": "compitum.fabricpc-tranche7-gate0-report/v1",
        "pass_number": 2,
        "first_pass_finding": (
            "8 configs varying only u_normal/u_high/initial_budget found genuine "
            "boundary states (56-73 of 77-92 reachable states) but the exact-belief "
            "policy tied EXACTLY with fixed-prior/shuffled controls in nearly every "
            "config -- diagnosed as tranche 6's fixed observation informativeness "
            "(P_OPPORTUNITY gap 0.30) making a single observation swamp any prior."
        ),
        "grid": grid_results,
        "n_configs_evaluated": len(grid_results),
        "n_configs_passing": len(passing),
        "total_elapsed_seconds": time.perf_counter() - started,
    }

    if not passing:
        report["outcome"] = (
            "STOPPED: no configuration in this second development grid passed Gate 0 "
            "either. The environment does not have identifiable belief-sensitive "
            "decisions at any tried parameterization; do not train FabricPC."
        )
        report["selected_config"] = None
    else:
        selected = max(passing, key=lambda r: r["occupancy"]["occupancy_fraction"])
        report["outcome"] = (
            f"Gate 0 PASSED. Selected config: p_opportunity=("
            f"{selected['p_opportunity_normal']},{selected['p_opportunity_high']}), "
            f"transition=({selected['transition_normal_to_high']},"
            f"{selected['transition_high_to_high']}), u_normal={selected['u_normal']}, "
            f"u_high={selected['u_high']}, initial_budget={selected['initial_budget']} "
            f"(occupancy={selected['occupancy']['occupancy_fraction']:.3f}), chosen solely "
            "by maximum belief-boundary occupancy among configs that also beat all "
            "belief-blind controls -- never by FabricPC/ridge/backprop performance."
        )
        report["selected_config"] = selected

    out_path = ARTIFACTS / "gate0_report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(rendered, encoding="utf-8", newline="")
    print(f"\n{report['outcome']}")
    print(f"report -> {out_path}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
