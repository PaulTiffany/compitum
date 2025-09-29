from dataclasses import dataclass
from typing import Any, Dict, Set


@dataclass
class Capabilities:
    regions: Set[str]
    tools_allowed: Set[str]
    deterministic: bool = False

    def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
        # Hook for model-specific gates; extend as needed.
        # Example: block if context["region"] not in self.regions
        return True
