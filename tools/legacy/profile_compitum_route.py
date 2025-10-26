import cProfile
import pstats
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "tools" / "routerbench"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from tools.routerbench.routers.partially_real_compitum_router import PartiallyRealCompitumRouter


def profile_compitum_route():
    data_path = "routerbench_5shot.pkl"

    # Load a small subset of data for profiling
    df = pd.read_pickle(data_path).head(20)  # Limit to 20 rows
    prompts = df["prompt"].tolist()
    models_to_route = [
        col.replace("|model_response", "") for col in df.columns if "|model_response" in col
    ]

    # Instantiate the router
    router = PartiallyRealCompitumRouter(models_to_route=models_to_route, data_path=data_path)

    # Run batch_route_prompts
    router.batch_route_prompts(prompts)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_compitum_route()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
