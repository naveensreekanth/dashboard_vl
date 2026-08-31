"""Shared fixture helpers for Phase 1 tests (temporary dirs; never mutate real data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def minimal_core_fixture(root: Path) -> Path:
    core = root / "data" / "core"
    (core / "rules").mkdir(parents=True, exist_ok=True)
    write_csv(
        core / "lots_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "tester_id": "TESTER_A",
                "total_dies": "1",
                "production_sequence": "1",
                "generation_seed": "1",
                "generator_version": "test",
            }
        ],
    )
    write_csv(
        core / "parts_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "status": "PASS",
                "tester_id": "TESTER_A",
                "generation_seed": "1",
                "generator_version": "test",
                "production_sequence": "1",
            }
        ],
    )
    write_csv(
        core / "test_catalog.csv",
        [
            {
                "test_id": "T_IR_DROP_MV",
                "test_name": "IR_DROP_MV",
                "parameter": "ir_drop",
                "unit": "mV",
                "direction": "UPPER",
                "source_status": "SOURCE_CONFIRMED",
                "dtl_eligible": "True",
                "optimization_priority": "PRIMARY",
            },
            {
                "test_id": "T_THERMAL_C",
                "test_name": "THERMAL_C",
                "parameter": "thermal",
                "unit": "°C",
                "direction": "UPPER",
                "source_status": "SOURCE_CONFIRMED",
                "dtl_eligible": "True",
                "optimization_priority": "PRIMARY",
            },
            {
                "test_id": "T_SETUP_SLACK_PS",
                "test_name": "SETUP_SLACK_PS",
                "parameter": "setup_slack",
                "unit": "ps",
                "direction": "LOWER",
                "source_status": "SOURCE_PRESENT_LIMIT_NOT_CONFIRMED",
                "dtl_eligible": "True",
                "optimization_priority": "SECONDARY",
            },
            {
                "test_id": "T_HOLD_SLACK_PS",
                "test_name": "HOLD_SLACK_PS",
                "parameter": "hold_slack",
                "unit": "ps",
                "direction": "LOWER",
                "source_status": "SOURCE_PRESENT_LIMIT_NOT_CONFIRMED",
                "dtl_eligible": "True",
                "optimization_priority": "SECONDARY",
            },
            {
                "test_id": "T_TEST_TIME_MS",
                "test_name": "TEST_TIME_MS",
                "parameter": "test_time",
                "unit": "ms",
                "direction": "UNKNOWN",
                "source_status": "SOURCE_PRESENT_LIMIT_NOT_CONFIRMED",
                "dtl_eligible": "False",
                "optimization_priority": "FUTURE",
            },
        ],
    )
    write_csv(
        core / "current_limits.csv",
        [
            {
                "test_id": "T_IR_DROP_MV",
                "test_name": "IR_DROP_MV",
                "parameter": "ir_drop",
                "unit": "mV",
                "lower_limit": "",
                "upper_limit": "25.0",
                "nominal_value": "",
                "limit_direction": "UPPER",
                "limit_type": "THRESHOLD",
                "source_status": "SOURCE_CONFIRMED",
                "active": "True",
            },
            {
                "test_id": "T_THERMAL_C",
                "test_name": "THERMAL_C",
                "parameter": "thermal",
                "unit": "°C",
                "lower_limit": "",
                "upper_limit": "60.0",
                "nominal_value": "",
                "limit_direction": "UPPER",
                "limit_type": "THRESHOLD",
                "source_status": "SOURCE_CONFIRMED",
                "active": "True",
            },
        ],
    )
    write_csv(
        core / "scenario_manifest_public.csv",
        [
            {
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "lot_id": "LOT_A",
                "production_sequence": "1",
                "tester_id": "TESTER_A",
                "target_parameters": "IR_DROP_MV,THERMAL_C",
                "generation_seed": "1",
                "generator_version": "test",
            }
        ],
    )
    write_csv(
        core / "measurements.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "pattern_id": "1",
                "test_id": "T_IR_DROP_MV",
                "test_name": "IR_DROP_MV",
                "parameter": "ir_drop",
                "measurement_value": "20.0",
                "unit": "mV",
                "scenario_id": "SCEN_NORMAL",
                "scenario_family": "normal",
                "tester_id": "TESTER_A",
                "site_id": "1",
                "pass_fail_pattern": "PASS",
                "die_status": "PASS",
                "generation_seed": "1",
                "generator_version": "test",
                "production_sequence": "1",
            }
        ],
    )
    write_json(
        core / "DATASET_VERSION.json",
        {
            "dataset_version": "DTL_DATASET_V1",
            "lot_count": 1,
            "die_count": 1,
            "measurement_row_count": 1,
        },
    )
    write_json(core / "rules" / "disposition_rules.json", {"version": "test"})
    write_json(core / "rules" / "limit_simulation_config.json", {"version": "test"})
    (core / "README_DATA_CONTRACT.md").write_text(
        "Do not use latent_quality or true_optimal_limit in agent input.\n",
        encoding="utf-8",
    )
    return core


def minimal_parametric_fixture(root: Path) -> Path:
    par = root / "data" / "parametric"
    (par / "rules").mkdir(parents=True, exist_ok=True)
    write_csv(
        par / "lots_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "production_sequence": "1",
                "tester_id": "TESTER_A",
                "v1_link": "True",
                "total_dies": "1",
                "generation_seed": "1",
                "generator_version": "test",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
            },
            {
                "lot_id": "LOT_P_ONLY",
                "scenario_id": "SCEN_P_RES",
                "scenario_family": "resistance_degradation",
                "production_sequence": "10000",
                "tester_id": "TESTER_B",
                "v1_link": "False",
                "total_dies": "1",
                "generation_seed": "1",
                "generator_version": "test",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
            },
        ],
    )
    write_csv(
        par / "parts_dim.csv",
        [
            {
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "tester_id": "TESTER_A",
                "site_id": "SITE_1",
                "v1_link": "True",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "generation_seed": "1",
                "generator_version": "test",
            },
            {
                "lot_id": "LOT_P_ONLY",
                "die_id": "LOT_P_ONLY_D001",
                "scenario_id": "SCEN_P_RES",
                "scenario_family": "resistance_degradation",
                "tester_id": "TESTER_B",
                "site_id": "SITE_2",
                "v1_link": "False",
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "generation_seed": "1",
                "generator_version": "test",
            },
        ],
    )
    conditions = [
        ("COND_RT_NOM", "25.0", "1.0", "NOMINAL"),
        ("COND_HOT_NOM", "85.0", "1.0", "HOT"),
        ("COND_RT_LOWV", "25.0", "0.9", "LOW_VDD"),
        ("COND_HOT_HIGHV", "85.0", "1.1", "HIGH_VDD"),
    ]
    write_csv(
        par / "conditions_dim.csv",
        [
            {
                "condition_id": cid,
                "temperature_c": t,
                "vdd_applied": v,
                "test_mode": m,
                "description": cid,
            }
            for cid, t, v, m in conditions
        ],
    )
    catalog_rows = []
    for tid, param, lim in [
        ("T_VMIN", "VMIN", "UPPER"),
        ("T_VMAX", "VMAX", "LOWER"),
        ("T_IDDQ", "IDDQ", "UPPER"),
        ("T_SUPPLY_CURRENT", "SUPPLY_CURRENT", "UPPER"),
        ("T_CONTACT_R", "CONTACT_RESISTANCE", "UPPER"),
        ("T_INTERCONNECT_R", "INTERCONNECT_RESISTANCE", "UPPER"),
        ("T_ON_R", "ON_RESISTANCE", "UPPER"),
    ]:
        catalog_rows.append(
            {
                "test_id": tid,
                "parameter": param,
                "test_name": param,
                "unit": "V" if "V" in param else ("uA" if param == "IDDQ" else ("mA" if param == "SUPPLY_CURRENT" else "ohm")),
                "limit_type": lim,
                "dtl_eligible": "True",
                "priority": "PRIMARY",
                "condition_dependent": "True",
                "synthetic_source": "SYNTHETIC_ASSUMED",
                "role": "DTL_TARGET",
            }
        )
    catalog_rows.append(
        {
            "test_id": "COND_VDD",
            "parameter": "VDD",
            "test_name": "VDD_APPLIED",
            "unit": "V",
            "limit_type": "NONE",
            "dtl_eligible": "False",
            "priority": "CONTEXT",
            "condition_dependent": "False",
            "synthetic_source": "TEST_CONDITION",
            "role": "TEST_CONDITION",
        }
    )
    write_csv(par / "test_catalog.csv", catalog_rows)
    write_csv(
        par / "current_limits.csv",
        [
            {
                "test_id": "T_VMIN",
                "parameter": "VMIN",
                "limit_type": "UPPER",
                "limit_value": "0.85",
                "unit": "V",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_VMAX",
                "parameter": "VMAX",
                "limit_type": "LOWER",
                "limit_value": "1.15",
                "unit": "V",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_IDDQ",
                "parameter": "IDDQ",
                "limit_type": "UPPER",
                "limit_value": "50.0",
                "unit": "uA",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_SUPPLY_CURRENT",
                "parameter": "SUPPLY_CURRENT",
                "limit_type": "UPPER",
                "limit_value": "120.0",
                "unit": "mA",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_CONTACT_R",
                "parameter": "CONTACT_RESISTANCE",
                "limit_type": "UPPER",
                "limit_value": "5.0",
                "unit": "ohm",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_INTERCONNECT_R",
                "parameter": "INTERCONNECT_RESISTANCE",
                "limit_type": "UPPER",
                "limit_value": "15.0",
                "unit": "ohm",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
            {
                "test_id": "T_ON_R",
                "parameter": "ON_RESISTANCE",
                "limit_type": "UPPER",
                "limit_value": "25.0",
                "unit": "ohm",
                "source": "SYNTHETIC_ASSUMED",
                "note": "synthetic",
            },
        ],
    )
    write_csv(
        par / "scenario_manifest_public.csv",
        [
            {
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "lot_count": "1",
                "mechanism": "baseline",
            }
        ],
    )
    write_csv(
        par / "measurements.csv",
        [
            {
                "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
                "scenario_id": "SCEN_P_NORMAL",
                "scenario_family": "normal",
                "lot_id": "LOT_A",
                "die_id": "LOT_A_D001",
                "condition_id": "COND_RT_NOM",
                "tester_id": "TESTER_A",
                "site_id": "SITE_1",
                "temperature_c": "25.0",
                "vdd_applied": "1.0",
                "test_mode": "NOMINAL",
                "test_id": "T_IDDQ",
                "parameter": "IDDQ",
                "measurement_value": "40.0",
                "unit": "uA",
                "limit_type": "UPPER",
                "generation_seed": "1",
                "generator_version": "test",
                "pass_fail_condition": "P",
            }
        ],
    )
    write_json(
        par / "PARAMETRIC_DATASET_VERSION.json",
        {
            "dataset_version": "DTL_PARAMETRIC_DATASET_V1",
            "lot_count": 2,
            "die_count": 2,
            "row_count": 1,
            "condition_count": 4,
        },
    )
    write_json(par / "rules" / "disposition_rules.json", {"version": "test"})
    write_json(par / "rules" / "limit_simulation_config.json", {"version": "test"})
    (par / "README_DATA_CONTRACT.md").write_text(
        "Forbidden: latent_quality, true_optimal_*\n",
        encoding="utf-8",
    )
    return par
