import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "tools" / "routerbench"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from tools.routerbench.routers.compitum_router import CompitumRouterAdapter

start_time = time.time()
adapter = CompitumRouterAdapter(
    pretrained_predictors_path=(
        Path("data/pretrain_predictors") / "predictors_all-MiniLM-L12-v2_0.1.joblib"
    )
)
end_time = time.time()

print(f"Total time to instantiate CompitumRouterAdapter: {end_time - start_time:.2f} seconds")
