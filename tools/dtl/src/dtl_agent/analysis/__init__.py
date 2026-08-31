"""Read-only analysis helpers (cost-savings estimator, presentation aggregates).

Does not modify recommendation, simulation, policy, or ML scoring.
"""

from dtl_agent.analysis.cost_savings import (
    CostSavingsAssumptions,
    estimate_cost_savings,
)

__all__ = [
    "CostSavingsAssumptions",
    "estimate_cost_savings",
]
