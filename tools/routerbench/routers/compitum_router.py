import time
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from compitum.boundary import BoundaryAnalyzer
from compitum.capabilities import Capabilities
from compitum.coherence import CoherenceFunctional
from compitum.constraints import ReflectiveConstraintSolver
from compitum.control import SRMFController
from compitum.energy import SymbolicFreeEnergy
from compitum.metric import SymbolicManifoldMetric
from compitum.models import Model
from compitum.pgd import RegexPromptExtractor
from compitum.predictors import CalibratedPredictor
from compitum.router import CompitumRouter
from routerbench.embedding.cache import EmbeddingCache


def _load_models(D: int, data_path: str) -> List[Model]:
    df = pd.read_pickle(data_path)  # nosec B301 # project-generated source
    model_names = [
        col
        for col in df.columns
        if "|" not in col
        and col
        not in [
            "sample_id",
            "prompt",
            "eval_name",
            "oracle_model_to_route_to",
        ]
    ]

    rng = np.random.default_rng(7)
    caps = Capabilities(regions={"US", "CA", "EU"}, tools_allowed={"none"})

    models = []
    for name in model_names:
        center = rng.normal(0.0, 1.0, size=D)
        cost = df[f"{name}|total_cost"].mean()
        models.append(Model(name=name, center=center, capabilities=caps, cost=cost))

    return models


class CompitumRouterAdapter:
    def __init__(
        self,
        router_defaults_path: str = "configs/router_defaults.yaml",
        constraints_path: str = "configs/constraints_routerbench_default.yaml",
        data_path: str = "src/routerbench/routerbench_5shot.pkl",
        pretrained_predictors_path: Optional[Path] = None,
        **kwargs,
    ) -> None:
        init_start_time = time.time()

        t = time.time()
        project_root = next(
            (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
            Path(__file__).resolve().parents[3],
        )
        router_defaults_path = project_root / router_defaults_path
        constraints_path = project_root / constraints_path
        data_path = project_root / data_path
        print(f"Path setup took {time.time() - t:.2f} seconds")

        t = time.time()
        dcfg = yaml.safe_load(router_defaults_path.read_text())
        D = 384
        rank = int(dcfg["metric"]["rank"])
        delta = float(dcfg["metric"]["delta"])
        print(f"Config loading took {time.time() - t:.2f} seconds")

        t = time.time()
        models = _load_models(D, str(data_path))
        print(f"_load_models took {time.time() - t:.2f} seconds")

        t = time.time()
        self._embed = EmbeddingCache(local_mode=True)
        print(f"EmbeddingCache initialization took {time.time() - t:.2f} seconds")

        if pretrained_predictors_path and pretrained_predictors_path.exists():
            print(f"Loading pre-trained predictors from {pretrained_predictors_path}")
            predictors = joblib.load(pretrained_predictors_path)
            print("Predictors loaded successfully.")
        else:
            t = time.time()
            df = pd.read_pickle(data_path)  # nosec B301
            prompts = df["prompt"].tolist()
            embeddings = self._embed.batch_get_embedding(
                tuple(prompts), embedding_model="all-MiniLM-L12-v2"
            )
            print(f"Embedding creation took {time.time() - t:.2f} seconds")

            t = time.time()
            predictors = {
                m.name: {
                    "quality": CalibratedPredictor(),
                    "latency": CalibratedPredictor(),
                    "cost": CalibratedPredictor(),
                }
                for m in models
            }
            print(f"Predictor initialization took {time.time() - t:.2f} seconds")

            t = time.time()
            for m in models:
                quality_data = df[m.name].astype(float).fillna(df[m.name].mean()).values
                cost_data = (
                    df[f"{m.name}|total_cost"].astype(float).fillna(df[f"{m.name}|total_cost"].mean()).values
                )
                latency_data = np.zeros(len(quality_data))

                predictors[m.name]["quality"].fit(embeddings, quality_data)
                predictors[m.name]["cost"].fit(embeddings, cost_data)
                predictors[m.name]["latency"].fit(embeddings, latency_data)
            print(f"Predictor fitting took {time.time() - t:.2f} seconds")

        t = time.time()
        metrics = {m.name: SymbolicManifoldMetric(D, rank, delta) for m in models}
        coherence = CoherenceFunctional(k=500)
        A, b = _load_constraints(Path(constraints_path))
        solver = ReflectiveConstraintSolver(A, b)
        boundary = BoundaryAnalyzer()
        srmf = SRMFController()
        energy = SymbolicFreeEnergy(
            dcfg["alpha"],
            dcfg["beta_t"],
            dcfg["beta_c"],
            dcfg["beta_d"],
            dcfg["beta_s"],
        )
        pgd = RegexPromptExtractor()
        print(f"Dependency creation took {time.time() - t:.2f} seconds")

        t = time.time()
        self.router = CompitumRouter(
            models,
            predictors,
            solver,
            coherence,
            boundary,
            srmf,
            pgd,
            metrics,
            energy,
            update_stride=int(dcfg["update_stride"]),
        )
        print(f"CompitumRouter initialization took {time.time() - t:.2f} seconds")

        print(f"AdapterOK D={D} rank={rank} delta={delta} cache_local={self._embed.local_mode}")
        print(f"CompitumRouterAdapter.__init__ took {time.time() - init_start_time:.2f} seconds")

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        start_time = time.time()
        if self.router is None:
            raise Exception("Router not initialized")

        embs = self._embed.batch_get_embedding(tuple(prompts), embedding_model="all-MiniLM-L12-v2")
        certificates = self.router.batch_route(embeddings=embs, prompts=prompts)
        model_names = [cert.model for cert in certificates]

        if os.environ.get("COMPITUM_DEBUG_ROUTER") == "1":
            if len(model_names) > 1:
                print(f"Selected models (first 2): {model_names[:2]}")
            elif len(model_names) == 1:
                print(f"Selected models (first 1): {model_names[0]}")

            print(
                f"CompitumRouterAdapter.batch_route_prompts took {time.time() - start_time:.2f} seconds"
            )
        return np.array(model_names)


def _load_constraints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Constraints file not found at {path}")

    constraints = yaml.safe_load(path.read_text())
    A = np.array(constraints.get("A", []))
    b = np.array(constraints.get("b", []))

    if A.size == 0 or b.size == 0:
        raise ValueError(f"Constraints file {path} is empty or malformed.")

    return A, b
