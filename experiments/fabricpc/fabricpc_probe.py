"""Read-only FabricPC trajectory probe (runs only under .venv-fabricpc).

This is the JAX-side exporter. It imports FabricPC read-only against the
pinned external checkout, verifies the Compitum-owned installation receipt
before any run, and emits raw observation payloads in the dependency-free
``compitum.fabricpc-observation-raw/v1`` shape consumed by
``compitum.trajectory.evidence.build_evidence``. FabricPC is never patched or
forked; trajectories come from its own public utilities
(``run_inference_with_history`` for ordinary runs,
``run_inference_with_full_history`` only for bounded paired audits).

Graph construction is adapted from Sketched's
``verification/tools/fabricpc_orientation_adapter.py`` (C:/src/sketched),
with provenance retained. The Sketched Principia/Lean authority model is not
imported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO_ROOT / "experiments" / "fabricpc" / "fabricpc_install_receipt.json"
DEFAULT_CHECKOUT = Path("C:/src/FabricPC")

RAW_SCHEMA = "compitum.fabricpc-observation-raw/v1"
ORIENTATION_INPUT_SCHEMA = "compitum.trajectory-orientation-input/v1"


def _require_pinned_fabricpc(checkout: Path) -> Dict[str, str]:
    """Verify the checkout against the receipt; return provenance fields."""
    from compitum.trajectory.capability import verify_receipt

    drift = verify_receipt(RECEIPT_PATH, checkout)
    if drift is not None:
        raise RuntimeError(f"refusing to observe: {drift}")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    return {
        "dependency_repository": receipt["source"]["repository"],
        "dependency_commit": receipt["source"]["commit"],
    }


def _build_structure(nonlinear: bool, eta: float, infer_steps: int, supervised: bool):
    """source -> hidden -> latent chain.

    With ``supervised=True`` the terminal node is declared as the task target
    so both ends can be clamped. This matters for observation: with only the
    input clamped, feedforward initialization already sits at the predictive
    coding equilibrium (all node energies exactly zero, latents static), so
    the inference trajectory is trivially constant -- verified empirically on
    this pin before this flag existed. Clamping input AND target forces the
    hidden node to reconcile, producing a genuine settling trajectory.
    """
    from fabricpc.core.activations import SigmoidActivation
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.core.topology import Edge
    from fabricpc.graph_assembly import TaskMap, graph
    from fabricpc.nodes import IdentityNode, Linear

    source = IdentityNode(shape=(2,), name="source")
    activation = SigmoidActivation() if nonlinear else None
    hidden = Linear(shape=(2,), name="hidden", **({"activation": activation} if activation else {}))
    latent = Linear(shape=(2,), name="latent", **({"activation": activation} if activation else {}))
    task_map = TaskMap(x=source, y=latent) if supervised else TaskMap(x=source)
    return graph(
        nodes=[source, hidden, latent],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=latent.slot("in")),
        ],
        task_map=task_map,
        inference=InferenceSGD(eta_infer=eta, infer_steps=infer_steps),
    )


def observe_case(
    case_id: str,
    clamp: Tuple[float, float],
    parameter_seed: int,
    state_seed: int,
    target: Tuple[float, float] = (0.6, 0.4),
    eta: float = 0.05,
    infer_steps: int = 12,
    nonlinear: bool = True,
    checkout: Path = DEFAULT_CHECKOUT,
) -> Dict[str, Any]:
    """One lightweight-history observation -> raw observation payload."""
    provenance = _require_pinned_fabricpc(checkout)
    started = time.perf_counter()

    import jax
    import jax.numpy as jnp
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.utils.dashboarding.inference_tracking import (
        run_inference_with_history,
        unstack_inference_history,
    )

    structure = _build_structure(nonlinear, eta, infer_steps, supervised=True)
    params = initialize_params(structure, jax.random.PRNGKey(parameter_seed))
    clamps = {
        "source": jnp.asarray([list(clamp)], dtype=jnp.float32),
        "latent": jnp.asarray([list(target)], dtype=jnp.float32),
    }
    initial = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=jax.random.PRNGKey(state_seed),
        clamps=clamps,
        params=params,
    )
    _, stacked = run_inference_with_history(params, initial, clamps, structure)
    steps_raw = unstack_inference_history(stacked)
    node_order = ["source", "hidden", "latent"]
    steps: List[Dict[str, Dict[str, float]]] = [
        {
            node: {metric: float(value) for metric, value in step[node].items()}
            for node in node_order
        }
        for step in steps_raw
    ]
    runtime = time.perf_counter() - started
    return {
        "schema": RAW_SCHEMA,
        "run_id": (
            f"{case_id}-pseed{parameter_seed}-sseed{state_seed}"
            f"-eta{eta:g}-steps{infer_steps}-nonlinear{int(nonlinear)}"
        ),
        "case_id": case_id,
        "seeds": {"parameter_seed": parameter_seed, "state_seed": state_seed},
        "config": {
            "clamp": list(clamp),
            "target": list(target),
            "eta_infer": eta,
            "infer_steps": infer_steps,
            "nonlinear": nonlinear,
            "graph": "source->hidden->latent (2,2,2), InferenceSGD, x and y clamped",
            "history": "lightweight (run_inference_with_history)",
        },
        "node_order": node_order,
        "steps": steps,
        "terminal": {},
        "runtime_seconds": runtime,
        **provenance,
    }


def paired_orientation_payload(
    perturbation: float,
    direction: Tuple[float, float],
    parameter_seed: int,
    state_seed: int,
    clamp: Tuple[float, float] = (0.25, -0.5),
    eta: float = 0.05,
    infer_steps: int = 12,
    nonlinear: bool = False,
    checkout: Path = DEFAULT_CHECKOUT,
) -> Dict[str, Any]:
    """Bounded full-history paired audit -> orientation-sensor input payload.

    Full state history is used here only because the orientation/blockwise
    instruments need intermediate vectors; ordinary observations use the
    lightweight history above.
    """
    provenance = _require_pinned_fabricpc(checkout)

    import jax
    import jax.numpy as jnp
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.utils.dashboarding.inference_tracking import (
        run_inference_with_full_history,
    )

    if perturbation == 0:
        raise ValueError("perturbation must be nonzero")
    # Unsupervised on purpose: the paired audit measures how an explicit
    # hidden-node perturbation is transported from the feedforward
    # equilibrium, exactly matching the Sketched orientation experiment.
    structure = _build_structure(nonlinear, eta, infer_steps, supervised=False)
    params = initialize_params(structure, jax.random.PRNGKey(parameter_seed))
    clamps = {"source": jnp.asarray([list(clamp)], dtype=jnp.float32)}
    base_initial = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=jax.random.PRNGKey(state_seed),
        clamps=clamps,
        params=params,
    )
    hidden_state = base_initial.nodes["hidden"]
    direction_array = jnp.asarray(direction, dtype=hidden_state.z_latent.dtype)
    direction_norm = jnp.linalg.norm(direction_array)
    if float(direction_norm) == 0.0:
        raise ValueError("perturbation direction must be nonzero")
    delta = (perturbation * direction_array / direction_norm).reshape(1, 2)
    probe_initial = base_initial._replace(
        nodes={
            **base_initial.nodes,
            "hidden": hidden_state._replace(z_latent=hidden_state.z_latent + delta),
        }
    )

    _, base_history = run_inference_with_full_history(params, base_initial, clamps, structure)
    _, probe_history = run_inference_with_full_history(params, probe_initial, clamps, structure)
    observed_nodes = ["hidden", "latent"]

    def _vector(state) -> List[float]:
        values: List[float] = []
        for name in observed_nodes:
            values.extend(float(x) for x in jnp.ravel(state.nodes[name].z_latent))
        return values

    base_states = [_vector(s) for s in [base_initial, *base_history]]
    probe_states = [_vector(s) for s in [probe_initial, *probe_history]]
    return {
        "schema": ORIENTATION_INPUT_SCHEMA,
        "run_id": (
            f"paired-pseed{parameter_seed}-sseed{state_seed}-eta{eta:g}"
            f"-eps{perturbation:g}-dir{direction[0]:g},{direction[1]:g}"
            f"-nonlinear{int(nonlinear)}"
        ),
        "thresholds": {
            "directional_gain": 1.0,
            "orientation_cosine": 0.0,
            "zero_tolerance": 1e-12,
        },
        "base_states": base_states,
        "probe_states": probe_states,
        "observed_nodes": observed_nodes,
        "blocks": {"hidden": [0, 2], "latent": [2, 4]},
        **provenance,
    }


def shuffled_control_payload(payload: Dict[str, Any], shuffle_seed: int) -> Dict[str, Any]:
    """Negative control: destroy temporal structure by shuffling step order.

    The per-step values are preserved exactly; only their order changes, so a
    predictor that only uses terminal-state or order-free aggregates is
    unaffected while genuine trajectory-order information is destroyed.
    """
    import random

    control = json.loads(json.dumps(payload))
    rng = random.Random(shuffle_seed)
    rng.shuffle(control["steps"])
    control["run_id"] = f"{payload['run_id']}-shuffled{shuffle_seed}"
    control["config"] = {**control["config"], "negative_control": "step-order shuffle"}
    return control


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="deterministic smoke run")
    args = parser.parse_args()
    if args.smoke:
        first = observe_case("smoke", (0.25, -0.5), parameter_seed=17, state_seed=23)
        second = observe_case("smoke", (0.25, -0.5), parameter_seed=17, state_seed=23)
        first.pop("runtime_seconds")
        second.pop("runtime_seconds")
        identical = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        print(
            json.dumps(
                {
                    "steps": len(first["steps"]),
                    "terminal_hidden_energy": first["steps"][-1]["hidden"]["energy"],
                    "deterministic_repeat_identical": identical,
                },
                indent=2,
            )
        )
        raise SystemExit(0 if identical else 1)
