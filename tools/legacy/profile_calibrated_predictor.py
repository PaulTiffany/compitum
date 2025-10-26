import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "tools" / "routerbench"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from embedding.cache import EmbeddingCache
from tools.routerbench.routers.compitum_router import _load_models

from src.compitum.predictors import CalibratedPredictor


def profile_predictor_fitting():
    data_path = "routerbench_5shot.pkl"
    D = 384  # Dimension of embeddings

    # Load a small subset of data for profiling
    df = pd.read_pickle(data_path).head(100)  # Limit to 100 rows
    prompts = df["prompt"].tolist()

    # Initialize EmbeddingCache
    embed_cache = EmbeddingCache(local_mode=True)

    # Load models (using the helper function)
    models = _load_models(D, data_path)

    # Get embeddings for the subset of prompts
    embeddings = embed_cache.batch_get_embedding(
        tuple(prompts), embedding_model="all-MiniLM-L12-v2"
    )

    # Initialize and fit predictors
    predictors = {
        m.name: {
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
        for m in models
    }

    for m in models:
        quality_data = df[m.name].fillna(df[m.name].mean()).values
        cost_data = df[f"{m.name}|total_cost"].fillna(df[f"{m.name}|total_cost"].mean()).values
        latency_data = np.zeros(len(quality_data))

        predictors[m.name]["quality"].fit(embeddings, quality_data)
        predictors[m.name]["cost"].fit(embeddings, cost_data)
        predictors[m.name]["latency"].fit(embeddings, latency_data)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_predictor_fitting()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
