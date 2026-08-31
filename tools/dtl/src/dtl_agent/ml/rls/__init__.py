"""Experimental Recursive Least Squares (RLS) candidate scorer.

Offline / shadow only. Does not modify the production GRU recommendation path.
"""

from dtl_agent.ml.rls.regressor import RLSRegressor

__all__ = ["RLSRegressor"]
