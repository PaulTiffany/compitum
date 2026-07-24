"""Tranche 6 FabricPC belief model: genuine trained-weight predictive
modeling, not frozen/random inference (unlike tranche 5). Runs only under
``.venv-fabricpc``. Imports FabricPC read-only against the same pinned
checkout/receipt as tranches 1-5
(``experiments/fabricpc/fabricpc_install_receipt.json``); never patches or
forks it. See docs/adr/0007-belief-state-fabricpc-bellman-pricing.md.

Task: regress the next-step belief prior (``P(regime=HIGH)`` entering step
``t + 1``, in ``[0, 1]``) from a flattened window of the last
``MAX_WINDOW`` declared belief-estimation channel vectors
(``compitum.regret_lab.belief_channels``). Never given the target itself,
future realizations, hindsight choices, or an oracle continuation value --
only previous route/outcome history and this window's own observed
signals, exactly like ``RidgeBeliefEstimator`` (arm 5).

One graph topology (source(55) -> hidden(16, sigmoid) -> belief(1,
sigmoid, Gaussian energy)) is used, UNCHANGED, for both training
algorithms: ``train_pcn`` (arm 7, genuine local predictive-coding
learning) and ``train_backprop`` (arm 6, ordinary end-to-end
backpropagation on the same graph) -- both start from the exact same
initialized parameters (same seed), so the only thing that differs
between arms 6 and 7 is the learning rule, not the architecture or
initialization. The graph's default ``FeedforwardStateInit`` (unchanged
from ``graph()``'s own default) is compatible with both training paths,
so no separate topology is needed for the backprop control.

Early-stopping rule (declared, not dynamic): trains for a fixed budget of
``NUM_EPOCHS`` epochs; validation MSE is computed after every epoch via a
genuine forward/inference pass (not FabricPC's built-in "accuracy", which
is meaningless here -- argmax over a width-1 output axis is a no-op); the
snapshot of parameters with the lowest validation MSE across all epochs is
kept as the trained checkpoint, discarding later, possibly-overfit epochs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / "experiments" / "fabricpc" / "fabricpc_install_receipt.json"
DEFAULT_CHECKOUT = Path("C:/src/FabricPC")

sys.path.insert(0, str(REPO_ROOT / "src"))

from compitum.regret_lab.belief_channels import (  # noqa: E402
    CHANNEL_DIMENSION,
    BeliefChannelHistory,
    advance_belief_history,
    compute_belief_channel_vector,
)
from compitum.regret_lab.belief_regime import INITIAL_BELIEF  # noqa: E402
from compitum.regret_lab.windowed_predictor import flatten_window  # noqa: E402
from compitum.trajectory.capability import verify_receipt  # noqa: E402

MAX_WINDOW = 5
SOURCE_DIM = MAX_WINDOW * CHANNEL_DIMENSION
HIDDEN_DIM = 16
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INFER_STEPS = 20
ETA_INFER = 0.05
TRAIN_SEEDS: Tuple[int, ...] = (0, 1, 2)
TOPOLOGY_DESCRIPTION = (
    f"source({SOURCE_DIM}, IdentityNode) -> hidden({HIDDEN_DIM}, Linear, "
    "SigmoidActivation) -> belief(1, Linear, SigmoidActivation, "
    "GaussianEnergy); InferenceSGD(eta_infer="
    f"{ETA_INFER}, infer_steps={INFER_STEPS}); default FeedforwardStateInit "
    "(shared by both train_pcn and train_backprop)"
)


def _require_pinned_fabricpc(checkout: Path) -> Dict[str, str]:
    drift = verify_receipt(RECEIPT_PATH, checkout)
    if drift is not None:
        raise RuntimeError(f"refusing to train: {drift}")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    return {
        "dependency_repository": receipt["source"]["repository"],
        "dependency_commit": receipt["source"]["commit"],
    }


class ArrayLoader:
    """Minimal in-memory loader matching FabricPC's generic ``(x, y)``
    tuple-yielding contract (see ``fabricpc.utils.data.dataloader.FewShotLoader``),
    over plain numpy feature/target arrays rather than an image dataset."""

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.num_samples = len(self.features)
        self._num_batches = max(1, (self.num_samples + batch_size - 1) // batch_size)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + 10000 + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1
        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield self.features[batch_idx], self.targets[batch_idx]

    def __len__(self) -> int:
        return self._num_batches


def _build_structure() -> Any:
    from fabricpc.core.activations import SigmoidActivation
    from fabricpc.core.energy import GaussianEnergy
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.core.topology import Edge
    from fabricpc.graph_assembly import TaskMap, graph
    from fabricpc.nodes import IdentityNode, Linear

    source = IdentityNode(shape=(SOURCE_DIM,), name="source")
    hidden = Linear(shape=(HIDDEN_DIM,), activation=SigmoidActivation(), name="hidden")
    belief = Linear(
        shape=(1,), activation=SigmoidActivation(), energy=GaussianEnergy(), name="belief"
    )
    return graph(
        nodes=[source, hidden, belief],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=belief.slot("in")),
        ],
        task_map=TaskMap(x=source, y=belief),
        inference=InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS),
    )


def predict_belief_batch(
    params: Any, structure: Any, features: Sequence[Sequence[float]], method: str, rng_key: Any
) -> np.ndarray:
    """Reads a belief prediction for every row of ``features``. PC-trained
    params (``method="pcn"``) are evaluated via their native iterative
    inference (clamp input only, settle, read the converged ``z_latent``);
    backprop-trained params (``method="backprop"``) via a single
    feedforward pass (``compute_forward_pass``), consistent with how each
    was trained."""
    import jax.numpy as jnp
    from fabricpc.core.inference import run_inference
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.training.train_backprop import compute_forward_pass

    x = jnp.asarray(np.asarray(features, dtype=np.float32))
    batch_size = x.shape[0]
    if method == "backprop":
        state = compute_forward_pass(params, structure, {"x": x}, rng_key)
    elif method == "pcn":
        clamps = {"source": x}
        initial = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        state = run_inference(params, initial, clamps, structure)
    else:
        raise ValueError(f"unknown method: {method}")
    return np.asarray(state.nodes["belief"].z_latent).reshape(-1)


def train_belief_model(
    method: str,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[float],
    val_features: Sequence[Sequence[float]],
    val_targets: Sequence[float],
    seed: int,
    checkout: Path = DEFAULT_CHECKOUT,
) -> Dict[str, Any]:
    """Trains one belief-regression model via ``method in {"pcn", "backprop"}``.
    Returns a dict carrying the trained (best-validation-epoch) params and
    structure alongside every declared training record: topology,
    initialization seed, objective, epoch/update count, train/validation
    split sizes, early-stopping rule outcome, optimizer, and a checkpoint
    hash of the frozen params actually used downstream."""
    if method not in ("pcn", "backprop"):
        raise ValueError(f"unknown method: {method}")
    provenance = _require_pinned_fabricpc(checkout)

    import jax
    import optax
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.training import train_pcn
    from fabricpc.training.train_backprop import train_backprop

    structure = _build_structure()
    master_key = jax.random.PRNGKey(seed)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)
    params = initialize_params(structure, graph_key)
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = ArrayLoader(train_features, train_targets, BATCH_SIZE, shuffle=True, seed=seed)
    val_targets_arr = np.asarray(val_targets, dtype=np.float64)

    history: List[Dict[str, float]] = []
    best: Dict[str, Any] = {"epoch": -1, "val_mse": float("inf"), "params": params}

    def epoch_callback(
        epoch_idx: int, epoch_params: Any, structure_: Any, config_: Any, rng_key: Any
    ) -> None:
        predictions = predict_belief_batch(
            epoch_params, structure_, val_features, method, eval_key
        )
        val_mse = float(np.mean((predictions - val_targets_arr) ** 2))
        history.append({"epoch": float(epoch_idx), "val_mse": val_mse})
        if val_mse < best["val_mse"]:
            best["val_mse"] = val_mse
            best["epoch"] = epoch_idx
            best["params"] = epoch_params

    started = time.perf_counter()
    if method == "pcn":
        config = {"num_epochs": NUM_EPOCHS}
        _, energy_history, _ = train_pcn(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
            rng_key=train_key,
            verbose=False,
            use_tqdm=False,
            epoch_callback=epoch_callback,
        )
    else:
        config = {"num_epochs": NUM_EPOCHS, "loss_type": "mse"}
        _, energy_history, _ = train_backprop(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
            rng_key=train_key,
            verbose=False,
            epoch_callback=epoch_callback,
        )
    elapsed = time.perf_counter() - started

    checkpoint_bytes = pickle.dumps(jax.tree_util.tree_map(lambda a: np.asarray(a), best["params"]))
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()

    return {
        "method": method,
        "seed": seed,
        "topology": TOPOLOGY_DESCRIPTION,
        "objective": (
            "mean squared error (GaussianEnergy free energy for PC; MSE loss for "
            "backprop) against the next-step belief_prior target"
        ),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": f"optax.adamw(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})",
        "train_examples": len(train_features),
        "val_examples": len(val_features),
        "early_stopping_rule": "fixed epoch budget; best-validation-MSE epoch snapshot kept",
        "best_epoch": best["epoch"],
        "best_val_mse": best["val_mse"],
        "training_history": history,
        "checkpoint_hash": checkpoint_hash,
        "runtime_seconds": elapsed,
        "params": best["params"],
        "structure": structure,
        "eval_key": eval_key,
        **provenance,
    }


class FabricPCBeliefEstimator:
    """Live per-step belief estimator wrapping one trained (frozen) FabricPC
    model -- structurally satisfies ``compitum.regret_lab.belief_pricing.
    BeliefEstimator`` (duck-typed; this module is JAX-side and not imported
    by the dependency-free ``regret_lab`` package). Builds its window
    online using the SAME convention as ``RidgeBeliefEstimator`` and
    ``build_belief_training_pairs``. Records every predicted belief in
    ``predicted_beliefs`` so a shuffled negative control (arm 8) can be
    built from this arm's own already-computed predictions with no
    further FabricPC calls."""

    def __init__(
        self,
        params: Any,
        structure: Any,
        method: str,
        rng_key: Any,
        max_window: int = MAX_WINDOW,
    ) -> None:
        self.params = params
        self.structure = structure
        self.method = method
        self.rng_key = rng_key
        self.max_window = max_window
        self._belief = INITIAL_BELIEF
        self._history = BeliefChannelHistory()
        self._window: Deque[np.ndarray] = deque()
        self.predicted_beliefs: List[float] = []

    def current_belief(self) -> float:
        return self._belief

    def advance(self, context: Any) -> None:
        remaining = context.remaining_before["budget"]
        steps_left = context.total_steps - context.step
        vector = compute_belief_channel_vector(
            remaining, context.case, self._history, steps_left, context.total_steps
        )
        self._window.append(vector)
        while len(self._window) > self.max_window:
            self._window.popleft()
        flattened = flatten_window(list(self._window), self.max_window, CHANNEL_DIMENSION)
        prediction = predict_belief_batch(
            self.params, self.structure, flattened.reshape(1, -1), self.method, self.rng_key
        )[0]
        self._belief = float(min(1.0, max(0.0, prediction)))
        self.predicted_beliefs.append(self._belief)
        self._history = advance_belief_history(self._history, context.chosen, context.case)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        rng = np.random.default_rng(0)
        n = 40
        features = rng.uniform(-1.0, 1.0, size=(n, SOURCE_DIM))
        targets = rng.uniform(0.0, 1.0, size=n)
        result_pcn = train_belief_model(
            "pcn", features[:30], targets[:30], features[30:], targets[30:], seed=0
        )
        result_bp = train_belief_model(
            "backprop", features[:30], targets[:30], features[30:], targets[30:], seed=0
        )
        print(
            json.dumps(
                {
                    "pcn_best_val_mse": result_pcn["best_val_mse"],
                    "pcn_best_epoch": result_pcn["best_epoch"],
                    "pcn_checkpoint_hash": result_pcn["checkpoint_hash"][:12],
                    "backprop_best_val_mse": result_bp["best_val_mse"],
                    "backprop_best_epoch": result_bp["best_epoch"],
                    "backprop_checkpoint_hash": result_bp["checkpoint_hash"][:12],
                },
                indent=2,
            )
        )
