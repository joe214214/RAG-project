from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyDecision:
    top_k_super: int = 4
    top_k_fine: int = 8
    hybrid_alpha: float = 0.3
    rerank_depth: int = 20
    use_hierarchical: bool = True
    activate_graph: bool = False
    max_context_tokens: int = 1024


class PolicyEngine:
    """
    Adaptive policy controller that accounts for query type and system load.
    """

    def __init__(
        self,
        *,
        default_super_k: int = 4,
        default_fine_k: int = 8,
    ) -> None:
        self.default_super_k = default_super_k
        self.default_fine_k = default_fine_k

    def decide_policy(self, query_type: str, system_load: float) -> PolicyDecision:
        """
        Determine retrieval knobs based on query type and load factor.

        Args:
            query_type: "oltp" or "olap".
            system_load: float in [0, 1], where 1.0 indicates saturation.
        """
        decision = PolicyDecision()
        high_load = system_load > 0.7

        if query_type.lower() == "oltp":
            decision.top_k_super = max(2, self.default_super_k - 1)
            decision.top_k_fine = max(4, self.default_fine_k - 2)
            decision.use_hierarchical = False
            decision.activate_graph = False
            decision.hybrid_alpha = 0.5
        else:
            decision.top_k_super = self.default_super_k + 2
            decision.top_k_fine = self.default_fine_k + 4
            decision.use_hierarchical = True
            decision.activate_graph = True
            decision.hybrid_alpha = 0.2

        if high_load:
            decision.top_k_super = max(1, decision.top_k_super - 1)
            decision.top_k_fine = max(2, decision.top_k_fine - 2)
            decision.rerank_depth = max(10, decision.rerank_depth - 5)
            decision.max_context_tokens = 512

        return decision

