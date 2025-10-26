import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from embedding.cache import EmbeddingCache
from utils import build_train_eval_dataset, get_models_to_route

from src.compitum.predictors import CalibratedPredictor

# Detect project root at runtime
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parents[2],
)
sys.path.append(str(project_root))
sys.path.append(str(project_root / "tools" / "routerbench"))


def pretrain_and_save_predictors(
    data_path: str, embedding_model: str, output_dir: Path, train_fraction: float = 0.7
):
    print(f"Loading data from {data_path}")
    df = pd.read_pickle(data_path)

    MODELS_TO_ROUTE = get_models_to_route(df)

    # Assuming 'grade-school-math' for simplicity, similar to evaluate_routers.py
    wanted_eval_name = "grade-school-math"
    other_eval_names = df.eval_name.unique().tolist()
    dataset_df_train, _ = build_train_eval_dataset(
        wanted_eval_name, other_eval_names, df, fraction=train_fraction
    )

    print(f"Initializing EmbeddingCache for {embedding_model}")
    embed_cache = EmbeddingCache(local_mode=True)

    prompts = dataset_df_train["prompt"].tolist()
    print(f"Getting embeddings for {len(prompts)} prompts")
    embeddings = embed_cache.batch_get_embedding(tuple(prompts), embedding_model=embedding_model)

    predictors = {
        m: {  # Changed m.name to m
            "quality": CalibratedPredictor(),
            "latency": CalibratedPredictor(),
            "cost": CalibratedPredictor(),
        }
        for m in MODELS_TO_ROUTE
    }

    print("Fitting predictors...")
    for m in MODELS_TO_ROUTE:
        quality_data = dataset_df_train[m].fillna(dataset_df_train[m].mean()).values
        cost_data = (
            dataset_df_train[f"{m}|total_cost"]
            .fillna(dataset_df_train[f"{m}|total_cost"].mean())
            .values
        )
        latency_data = np.zeros(len(quality_data))  # Assuming latency data is all zeros for now

        predictors[m]["quality"].fit(embeddings, quality_data)
        predictors[m]["cost"].fit(embeddings, cost_data)
        predictors[m]["latency"].fit(embeddings, latency_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"predictors_{embedding_model}_{train_fraction}.joblib"
    print(f"Saving predictors to {output_file}")
    joblib.dump(predictors, output_file)
    print("Predictors saved successfully.")


if __name__ == "__main__":
    # Example usage (these should ideally come from command line arguments)
    data_path = "routerbench_5shot.pkl"
    embedding_model = "all-MiniLM-L12-v2"
    output_dir = Path("data/pretrain_predictors")
    train_fraction = 0.1  # Use the reduced fraction for pre-training

    pretrain_and_save_predictors(data_path, embedding_model, output_dir, train_fraction)
