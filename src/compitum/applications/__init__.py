"""
Domain-specific adapters that wrap Compitum's core math for familiar use in
applied settings (e.g., fusion, power grids). Keeping these under the
``compitum.applications`` namespace preserves a clean separation between the
generic core and domain-facing shims.
"""

# Shallow re-export to avoid deep nesting in imports
from .fusion import PlasmaMonitor  # noqa: F401

__all__ = ["PlasmaMonitor"]
