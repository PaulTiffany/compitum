from dataclasses import dataclass
from typing import Any, Dict, Set


@dataclass
class Capabilities:
    regions: Set[str]
    tools_allowed: Set[str]
    deterministic: bool = False

    def supports(self, pgd_vector: Any, context: Dict[str, Any] | None = None) -> bool:
        if context and "region" in context:
            return context["region"] in self.regions
        return True
