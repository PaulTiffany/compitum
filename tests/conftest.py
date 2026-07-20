"""
Pytest configuration file.
"""

from typing import Any
import os

try:
    from hypothesis import settings
except Exception:  # pragma: no cover - optional hypothesis support
    if os.environ.get("HYPOTHESIS_OPTIONAL", "1") == "1":

        class _DummySettings:
            @staticmethod
            def register_profile(*args, **kwargs) -> None:
                pass

            @staticmethod
            def load_profile(*args, **kwargs) -> None:
                pass

        settings = _DummySettings()  # type: ignore
    else:
        raise


def pytest_configure(config: Any) -> None:
    """Pytest hook to configure settings and profiles."""
    config.addinivalue_line(
        "markers", "invariants: property-based tests for core system invariants"
    )
    config.addinivalue_line("markers", "routerbench: RouterBench integration tests (optional)")
    # Ensure artifacts directory exists for Hypothesis DB and reports.
    try:
        os.makedirs("artifacts", exist_ok=True)
    except Exception:
        pass


# Register deterministic profiles for different testing scenarios.
# Keep dev relatively fast; CI deterministic; mutation a bit higher; stress is heavy.
settings.register_profile("dev", max_examples=25, deadline=200)
settings.register_profile("ci", max_examples=50, derandomize=True, deadline=500)
settings.register_profile("mutation", max_examples=100, derandomize=True, deadline=750)
settings.register_profile("mutation_ci", max_examples=30, derandomize=True, deadline=400)
settings.register_profile("stress", max_examples=400, derandomize=True, deadline=None)

# Select profile via env var if present; otherwise prefer CI in CI environments, else dev.
_env_profile = os.getenv("HYPOTHESIS_PROFILE", "").strip()
if _env_profile:
    settings.load_profile(_env_profile)
else:
    settings.load_profile("ci" if os.getenv("CI") else "dev")
