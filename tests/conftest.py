"""
Pytest configuration file.
"""
from typing import Any

from hypothesis import settings


def pytest_configure(config: Any) -> None:
    """Pytest hook to configure settings and profiles."""
    config.addinivalue_line(
        "markers", "invariants: property-based tests for core system invariants"
    )

# Register deterministic profiles for different testing scenarios.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci", max_examples=100, derandomize=True, deadline=None
)
settings.register_profile(
    "mutation", max_examples=200, derandomize=True, deadline=None
)
settings.register_profile(
    "stress", max_examples=800, derandomize=True, deadline=None
)

# Load the "ci" profile by default for all test runs.
settings.load_profile("ci")
