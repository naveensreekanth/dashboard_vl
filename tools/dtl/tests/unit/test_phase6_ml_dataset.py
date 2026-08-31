"""Phase 6 unit tests."""

from __future__ import annotations

import pandas as pd

from dtl_agent.ml_dataset.pipeline import _det_example_id, _pairwise_contract


def test_deterministic_example_id() -> None:
    a = _det_example_id(["train", "LOT_A", "D1", "ir_drop", "25.0"])
    b = _det_example_id(["train", "LOT_A", "D1", "ir_drop", "25.0"])
    c = _det_example_id(["train", "LOT_A", "D1", "ir_drop", "26.0"])
    assert a == b
    assert a != c


def test_pairwise_contract_shape() -> None:
    df = pd.DataFrame({"target_score": [0.1, 0.2], "lot_id": ["L1", "L1"]})
    contract = _pairwise_contract(df, tolerance=1e-4)
    assert contract["enabled"] is True
    assert "score_A > score_B" in contract["preference_rule"]
    assert "abs(score_A - score_B)" in contract["tie_rule"]
