"""Phase 10.11 measurement / distribution / condition API tests."""

from __future__ import annotations

import csv
from unittest.mock import patch

from fastapi.testclient import TestClient

from dtl_agent.api.measurement_data import (
    SOURCE_CLASSIFICATION,
    _core_die_features,
    _feature_prefix,
    _param_measurements_index,
    get_distribution,
    get_selected_measurement,
)
from dtl_agent.features.stats import compute_dist_stats
from tests.api.conftest import ROOT


def _pick_core_fixture(client: TestClient) -> tuple[str, str, str]:
    lots = client.get("/api/v1/lots").json()["lots"]
    for lot_id in lots:
        dies = client.get(f"/api/v1/lots/{lot_id}/dies").json()["dies"]
        for die_id in dies:
            params = client.get(f"/api/v1/lots/{lot_id}/dies/{die_id}/parameters").json()[
                "parameters"
            ]
            if "ir_drop" in params:
                return lot_id, die_id, "ir_drop"
    raise AssertionError("No core ir_drop fixture found")


def _pick_param_fixture(client: TestClient) -> tuple[str, str, str]:
    lots = client.get("/api/v1/lots").json()["lots"]
    for lot_id in lots:
        dies = client.get(f"/api/v1/lots/{lot_id}/dies").json()["dies"]
        for die_id in dies:
            params = client.get(f"/api/v1/lots/{lot_id}/dies/{die_id}/parameters").json()[
                "parameters"
            ]
            if "VMIN" in params:
                return lot_id, die_id, "VMIN"
    raise AssertionError("No parametric VMIN fixture found")


def test_core_selected_die_measurement(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["domain"] == "core"
    assert data["lot_id"] == lot_id
    assert data["die_id"] == die_id
    assert data["parameter"] == parameter
    assert data["observed_value_rule"] == "median_over_patterns"
    assert isinstance(data["observed_value"], float)
    assert data["source_classification"] == SOURCE_CLASSIFICATION
    assert data["dataset_version"]
    assert "Synthetic" in data["disclaimer"]

    # Matches Phase 3 die_features median
    row = _core_die_features(str(ROOT))[(lot_id, die_id)]
    prefix = _feature_prefix("core", parameter)
    assert float(row[f"{prefix}_median"]) == data["observed_value"]


def test_core_distribution_die_scope(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter, "scope": "die"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["scope"] == "die"
    assert data["n"] == 200
    assert data["stats_method"] == "phase3_compute_dist_stats"
    assert data["source_classification"] == SOURCE_CLASSIFICATION
    assert data["dataset_version"]
    row = _core_die_features(str(ROOT))[(lot_id, die_id)]
    prefix = _feature_prefix("core", parameter)
    assert data["min"] == float(row[f"{prefix}_min"])
    assert data["median"] == float(row[f"{prefix}_median"])
    assert data["p95"] == float(row[f"{prefix}_p95"])
    assert data["max"] == float(row[f"{prefix}_max"])


def test_parametric_selected_die_measurement(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["domain"] == "parametric"
    assert data["condition_id"] == "COND_RT_NOM"
    assert data["observed_value_rule"] == "selected_condition"
    assert data["source_classification"] == SOURCE_CLASSIFICATION
    assert data["dataset_version"]
    expected = _param_measurements_index(str(ROOT))[
        (lot_id, die_id, parameter, "COND_RT_NOM")
    ]["measurement_value"]
    assert data["observed_value"] == expected


def test_parametric_distribution(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter, "scope": "die"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["scope"] == "die"
    assert data["n"] == 4  # four conditions
    assert data["stats_method"] == "phase3_compute_dist_stats"
    assert data["source_classification"] == SOURCE_CLASSIFICATION


def test_parametric_conditions_endpoint(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/conditions",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert len(data["conditions"]) == 4
    for row in data["conditions"]:
        assert row["condition_id"]
        assert row["measurement_value"] is not None
        assert row["unit"]
        assert "temperature_c" in row
        assert "vdd_applied" in row
        assert "test_mode" in row
        assert "pass_fail_condition" in row


def test_core_conditions_not_condition_aware(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/conditions",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False
    assert data["reason"] == "not_condition_aware"
    assert data["conditions"] == []


def test_missing_die(session_client: TestClient) -> None:
    lot_id, _, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        "/api/v1/dies/INVALID_DIE/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.status_code == 404
    assert "Die not found" in resp.json()["error"]["message"]


def test_missing_parameter(session_client: TestClient) -> None:
    lot_id, die_id, _ = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": "NOT_A_REAL_PARAM"},
    )
    assert resp.status_code == 404
    assert "Parameter not found" in resp.json()["error"]["message"]


def test_wrong_lot_die_combination(session_client: TestClient) -> None:
    lots = session_client.get("/api/v1/lots").json()["lots"]
    assert len(lots) >= 2
    lot_a, lot_b = lots[0], lots[1]
    die_b = session_client.get(f"/api/v1/lots/{lot_b}/dies").json()["dies"][0]
    params = session_client.get(f"/api/v1/lots/{lot_b}/dies/{die_b}/parameters").json()[
        "parameters"
    ]
    resp = session_client.get(
        f"/api/v1/dies/{die_b}/measurements",
        params={"lot_id": lot_a, "parameter": params[0]},
    )
    assert resp.status_code == 404
    assert "Die not found for lot" in resp.json()["error"]["message"]


def test_missing_measurement_found_false(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    with patch(
        "dtl_agent.api.measurement_data._core_die_features",
        return_value={(lot_id, die_id): {}},
    ), patch(
        "dtl_agent.api.measurement_data._stream_core_values",
        return_value=[],
    ):
        data = get_selected_measurement(
            str(ROOT),
            lot_id=lot_id,
            die_id=die_id,
            parameter=parameter,
        )
    assert data["found"] is False
    assert data["observed_value"] is None


def test_units_from_backend_catalog(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    catalog = ROOT / "data" / "core" / "test_catalog.csv"
    expected_unit = None
    with catalog.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["parameter"] == parameter:
                expected_unit = row["unit"]
                break
    assert expected_unit
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.json()["unit"] == expected_unit


def test_source_classification_synthetic(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    for path in (
        f"/api/v1/dies/{die_id}/measurements",
        f"/api/v1/dies/{die_id}/distribution",
        f"/api/v1/dies/{die_id}/conditions",
    ):
        resp = session_client.get(path, params={"lot_id": lot_id, "parameter": parameter})
        assert resp.status_code == 200
        assert resp.json()["source_classification"] == "SYNTHETIC"


def test_dataset_version_present(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.json()["dataset_version"] == "DTL_DATASET_V1"

    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.json()["dataset_version"] == "DTL_PARAMETRIC_DATASET_V1"


def test_no_recommendation_engine_invocation(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    with (
        patch("dtl_agent.recommendation.pipeline.recommend") as mock_recommend,
        patch("dtl_agent.recommendation.ranking.rank_candidates") as mock_rank,
        patch("dtl_agent.recommendation.safety.evaluate_safety") as mock_safety,
    ):
        for path in (
            f"/api/v1/dies/{die_id}/measurements",
            f"/api/v1/dies/{die_id}/distribution",
            f"/api/v1/dies/{die_id}/conditions",
        ):
            resp = session_client.get(
                path, params={"lot_id": lot_id, "parameter": parameter}
            )
            assert resp.status_code == 200
        mock_recommend.assert_not_called()
        mock_rank.assert_not_called()
        mock_safety.assert_not_called()


def test_distribution_matches_compute_dist_stats_for_parametric(
    session_client: TestClient,
) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    values = [
        float(rec["measurement_value"])
        for (lot, die, param, _cid), rec in _param_measurements_index(str(ROOT)).items()
        if lot == lot_id and die == die_id and param == parameter
        and rec["measurement_value"] is not None
    ]
    expected = compute_dist_stats(values)
    assert expected is not None
    data = get_distribution(
        str(ROOT),
        lot_id=lot_id,
        die_id=die_id,
        parameter=parameter,
        scope="die",
    )
    assert data["n"] == expected.count
    assert abs(data["min"] - expected.min) < 1e-9
    assert abs(data["median"] - expected.median) < 1e-9
    assert abs(data["p95"] - expected.p95) < 1e-9
    assert abs(data["max"] - expected.max) < 1e-9


def test_n_correct_core_and_parametric(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    data = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter},
    ).json()
    assert data["n"] == 200

    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    data = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter},
    ).json()
    assert data["n"] == 4


def test_scope_die_default(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter},
    )
    assert resp.json()["scope"] == "die"


def test_scope_lot(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter, "scope": "lot"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scope"] == "lot"
    assert data["found"] is True
    assert data["n"] > 200  # many dies × 200 patterns
    assert data["source_classification"] == "SYNTHETIC"


def test_condition_filtering(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={
            "lot_id": lot_id,
            "parameter": parameter,
            "scope": "die",
            "condition_id": "COND_RT_NOM",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["n"] == 1
    assert data["condition_id"] == "COND_RT_NOM"
    assert data["min"] == data["median"] == data["p95"] == data["max"]

    meas = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={
            "lot_id": lot_id,
            "parameter": parameter,
            "condition_id": "COND_HOT_NOM",
        },
    ).json()
    assert meas["condition_id"] == "COND_HOT_NOM"
    assert meas["observed_value"] is not None


def test_no_fabricated_values_match_canonical(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_param_fixture(session_client)
    api_val = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": lot_id, "parameter": parameter, "condition_id": "COND_RT_NOM"},
    ).json()["observed_value"]
    path = ROOT / "data" / "parametric" / "measurements.csv"
    canonical = None
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if (
                row["lot_id"] == lot_id
                and row["die_id"] == die_id
                and row["parameter"] == parameter
                and row["condition_id"] == "COND_RT_NOM"
            ):
                canonical = float(row["measurement_value"])
                break
    assert canonical is not None
    assert api_val == canonical


def test_unsupported_condition_core(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={
            "lot_id": lot_id,
            "parameter": parameter,
            "condition_id": "COND_RT_NOM",
        },
    )
    assert resp.status_code == 404
    assert "Condition not supported" in resp.json()["error"]["message"]


def test_invalid_scope(session_client: TestClient) -> None:
    lot_id, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/distribution",
        params={"lot_id": lot_id, "parameter": parameter, "scope": "global"},
    )
    assert resp.status_code == 404
    assert "Invalid scope" in resp.json()["error"]["message"]


def test_missing_lot(session_client: TestClient) -> None:
    _, die_id, parameter = _pick_core_fixture(session_client)
    resp = session_client.get(
        f"/api/v1/dies/{die_id}/measurements",
        params={"lot_id": "NO_SUCH_LOT", "parameter": parameter},
    )
    assert resp.status_code == 404
    assert "Lot not found" in resp.json()["error"]["message"]
