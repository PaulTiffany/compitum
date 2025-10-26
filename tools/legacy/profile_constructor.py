import cProfile
import pstats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "tools" / "routerbench"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from tools.routerbench.routers.compitum_router import CompitumRouterAdapter

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    adapter = CompitumRouterAdapter()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
