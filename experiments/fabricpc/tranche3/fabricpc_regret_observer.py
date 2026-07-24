"""Tranche 3 FabricPC observer: single-step observation of the declared
15-channel dynamic-regret state (``compitum.regret_lab.channels``).

Runs only under ``.venv-fabricpc``. Imports FabricPC read-only against the
same pinned checkout/receipt as tranches 1-2
(``experiments/fabricpc/fabricpc_install_receipt.json``); never patches or
forks it. Graph: source (IdentityNode, dim=15) -> hidden (Linear, dim=8) ->
latent (Linear, dim=4), latent clamped to zero to force genuine settling
(per tranche 1's finding that an unclamped-target feedforward graph sits at
its equilibrium trivially). Network parameters are seeded from the
sequence id alone (fixed for a whole sequence); only the clamped source
channel vector evolves step to step -- consistent with tranches 1-2.

"Trajectory" here means the PC graph's own settling dynamics within one
inference call (as in tranches 1-2), not a multi-environment-step window;
this pilot does not test whether windowing multiple past environment steps
into FabricPC's own input would do better -- see the tranche 3 report's
open items.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / "experiments" / "fabricpc" / "fabricpc_install_receipt.json"
DEFAULT_CHECKOUT = Path("C:/src/FabricPC")

sys.path.insert(0, str(REPO_ROOT / "src"))

from compitum.regret_lab.channels import CHANNEL_DIMENSION  # noqa: E402
from compitum.trajectory.capability import verify_receipt  # noqa: E402

RAW_SCHEMA = (
    "compitum.fabricpc-observation-raw/v1"  # compitum.trajectory.evidence's required schema
)
NODE_ORDER = ["source", "hidden", "latent"]
HIDDEN_DIM = 8
LATENT_DIM = 4


def _require_pinned_fabricpc(checkout: Path) -> Dict[str, str]:
    drift = verify_receipt(RECEIPT_PATH, checkout)
    if drift is not None:
        raise RuntimeError(f"refusing to observe: {drift}")
    import json

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    return {
        "dependency_repository": receipt["source"]["repository"],
        "dependency_commit": receipt["source"]["commit"],
    }


def _seed_from_id(sequence_id: str) -> int:
    digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def _build_structure(eta: float, infer_steps: int):
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.core.topology import Edge
    from fabricpc.graph_assembly import TaskMap, graph
    from fabricpc.nodes import IdentityNode, Linear

    source = IdentityNode(shape=(CHANNEL_DIMENSION,), name="source")
    hidden = Linear(shape=(HIDDEN_DIM,), name="hidden")
    latent = Linear(shape=(LATENT_DIM,), name="latent")
    return graph(
        nodes=[source, hidden, latent],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=latent.slot("in")),
        ],
        task_map=TaskMap(x=source, y=latent),
        inference=InferenceSGD(eta_infer=eta, infer_steps=infer_steps),
    )


def observe_channel_vector(
    sequence_id: str,
    step: int,
    channel_vector: np.ndarray,
    eta: float = 0.05,
    infer_steps: int = 12,
    checkout: Path = DEFAULT_CHECKOUT,
) -> Dict[str, Any]:
    """One lightweight-history observation of one decision step's channel
    vector. The network's parameters are seeded from ``sequence_id`` alone
    (fixed for the whole sequence); ``step`` only labels the output."""
    if channel_vector.shape != (CHANNEL_DIMENSION,):
        raise ValueError(f"channel_vector must have shape ({CHANNEL_DIMENSION},)")
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

    seed = _seed_from_id(sequence_id)
    structure = _build_structure(eta, infer_steps)
    params = initialize_params(structure, jax.random.PRNGKey(seed))
    clamps = {
        "source": jnp.asarray([channel_vector], dtype=jnp.float32),
        "latent": jnp.zeros((1, LATENT_DIM), dtype=jnp.float32),
    }
    initial = initialize_graph_state(
        structure,
        batch_size=1,
        rng_key=jax.random.PRNGKey(seed + 1),
        clamps=clamps,
        params=params,
    )
    _, stacked = run_inference_with_history(params, initial, clamps, structure)
    steps_raw = unstack_inference_history(stacked)
    steps: List[Dict[str, Dict[str, float]]] = [
        {
            node: {metric: float(value) for metric, value in step_data[node].items()}
            for node in NODE_ORDER
        }
        for step_data in steps_raw
    ]
    runtime = time.perf_counter() - started
    return {
        "schema": RAW_SCHEMA,
        "run_id": f"{sequence_id}-step{step}",
        "case_id": f"{sequence_id}-step{step}",
        "seeds": {"parameter_seed": seed, "state_seed": seed + 1},
        "config": {
            "eta_infer": eta,
            "infer_steps": infer_steps,
            "graph": "source(15)->hidden(8)->latent(4), InferenceSGD, x and y clamped",
            "history": "lightweight (run_inference_with_history)",
            "channel_layout": "compitum.regret_lab.channels (15-dim, see its docstring)",
        },
        "node_order": NODE_ORDER,
        "steps": steps,
        "terminal": {},
        "runtime_seconds": runtime,
        **provenance,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        import json

        vector = np.linspace(-1.0, 1.0, CHANNEL_DIMENSION)
        first = observe_channel_vector("smoke-sequence", 0, vector)
        second = observe_channel_vector("smoke-sequence", 0, vector)
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
