"""Minimal frozen submission demonstration.

Deterministic, no training, no FabricPC/JAX dependency for the core
demo. Runs under plain ``.venv``. Demonstrates, in order:

1. scalar-price action selection (tranche 6's linear translation, the
   corrected-away baseline);
2. Bellman action-shadow-charge selection (tranche 6.5's correction);
3. exact online-optimum selection (the principal regret comparator);
4. that (2) and (3) agree exactly, at every step of a full sequence --
   the Gate A-prime identity this program's central claim rests on;
5. one belief-sensitive state (tranche 7's environment) where the
   Bellman-optimal action genuinely differs on opposite sides of a
   verified belief boundary.

See SUBMISSION.md and docs/adr/0008-*.md / 0009-*.md for the full
argument these five facts support.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compitum.regret_lab import (  # noqa: E402
    BeliefPricingController,
    BeliefSensitiveBellmanOracle,
    BellmanOracle,
    ExactBeliefEstimator,
    generate_belief_sequence,
    run_online_optimal_policy,
    run_shadow_charge_policy,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF  # noqa: E402
from compitum.regret_lab.simulator import simulate_policy  # noqa: E402

DEMO_SEED = 42
DEMO_SEQUENCE_ID = "submission-demo"

# A concrete, verified belief-decision-boundary state in tranche 7's
# frozen Gate-0-selected environment configuration -- not illustrative,
# directly computed from BeliefSensitiveBellmanOracle.
BOUNDARY_CONFIG = dict(
    u_normal_opportunity=1.0,
    u_high_opportunity=8.0,
    p_opportunity_normal=0.15,
    p_opportunity_high=0.25,
    transition_normal_to_high=0.2,
    transition_high_to_high=0.6,
)
BOUNDARY_REMAINING_STEPS = 1
BOUNDARY_BUDGET = 4.5
BOUNDARY_OBSERVED = True
BOUNDARY_LOW_BELIEF = 0.05
BOUNDARY_HIGH_BELIEF = 0.2


def main() -> int:
    import numpy as np

    rng = np.random.default_rng(DEMO_SEED)
    seq, _, _, _ = generate_belief_sequence(rng, DEMO_SEQUENCE_ID)
    oracle = BellmanOracle()

    # 1. Scalar-price action selection (tranche 6's linear translation).
    scalar_controller = BeliefPricingController(
        oracle=oracle,
        belief_estimator=ExactBeliefEstimator(belief=INITIAL_BELIEF),
        total_steps=len(seq.cases),
        initial_budget=seq.initial_budget["budget"],
    )
    scalar_result, _ = simulate_policy(seq, pricing_controller=scalar_controller)

    # 2. Bellman action-shadow-charge selection (tranche 6.5's correction).
    shadow_result, _, _ = run_shadow_charge_policy(
        seq, oracle, ExactBeliefEstimator(belief=INITIAL_BELIEF)
    )

    # 3. Exact online-optimum selection (the principal regret comparator).
    online_result, _ = run_online_optimal_policy(seq, oracle, INITIAL_BELIEF)

    # 4. Equality between shadow charge and the online optimum.
    shadow_equals_online = shadow_result.choices == online_result.choices

    # 5. One belief-sensitive state showing different actions on opposite
    #    sides of a verified belief boundary.
    boundary_oracle = BeliefSensitiveBellmanOracle(**BOUNDARY_CONFIG)
    low_action, _ = boundary_oracle.best_action_given_observation(
        BOUNDARY_REMAINING_STEPS, BOUNDARY_BUDGET, BOUNDARY_LOW_BELIEF, BOUNDARY_OBSERVED
    )
    high_action, _ = boundary_oracle.best_action_given_observation(
        BOUNDARY_REMAINING_STEPS, BOUNDARY_BUDGET, BOUNDARY_HIGH_BELIEF, BOUNDARY_OBSERVED
    )

    print(f"scalar choice (first 5 steps):        {scalar_result.choices[:5]}")
    print(f"shadow-charge choice (first 5 steps):  {shadow_result.choices[:5]}")
    print(f"online-optimal choice (first 5 steps): {online_result.choices[:5]}")
    print(f"shadow-charge equality: {'PASS' if shadow_equals_online else 'FAIL'}")
    print(
        f"belief boundary: remaining_steps={BOUNDARY_REMAINING_STEPS} "
        f"budget={BOUNDARY_BUDGET} observed={BOUNDARY_OBSERVED}"
    )
    print(f"low-belief ({BOUNDARY_LOW_BELIEF}) action:  {low_action}")
    print(f"high-belief ({BOUNDARY_HIGH_BELIEF}) action: {high_action}")

    ok = shadow_equals_online and low_action != high_action
    print(f"\noverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
