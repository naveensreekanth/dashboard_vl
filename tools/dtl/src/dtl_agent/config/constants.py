"""Domain constants derived from inspected catalogs/limits (not invented)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedLimit:
    test_id: str
    parameter: str
    direction: str
    value: float
    unit: str
    source: str


CORE_DATASET_VERSION = "DTL_DATASET_V1"
PARAMETRIC_DATASET_VERSION = "DTL_PARAMETRIC_DATASET_V1"

CORE_PRIMARY_TEST_IDS = frozenset({"T_IR_DROP_MV", "T_THERMAL_C"})
CORE_SECONDARY_TEST_IDS = frozenset({"T_SETUP_SLACK_PS", "T_HOLD_SLACK_PS"})
CORE_CONTEXT_TEST_IDS = frozenset({"T_TEST_TIME_MS"})
CORE_ALL_TEST_IDS = CORE_PRIMARY_TEST_IDS | CORE_SECONDARY_TEST_IDS | CORE_CONTEXT_TEST_IDS

PARAMETRIC_PRIMARY_TEST_IDS = frozenset(
    {
        "T_VMIN",
        "T_VMAX",
        "T_IDDQ",
        "T_SUPPLY_CURRENT",
        "T_CONTACT_R",
        "T_INTERCONNECT_R",
        "T_ON_R",
    }
)
PARAMETRIC_CONTEXT_TEST_IDS = frozenset({"COND_VDD"})

CORE_EXPECTED_LIMITS: tuple[ExpectedLimit, ...] = (
    ExpectedLimit("T_IR_DROP_MV", "ir_drop", "UPPER", 25.0, "mV", "SOURCE_CONFIRMED"),
    ExpectedLimit("T_THERMAL_C", "thermal", "UPPER", 60.0, "°C", "SOURCE_CONFIRMED"),
)

PARAMETRIC_EXPECTED_LIMITS: tuple[ExpectedLimit, ...] = (
    ExpectedLimit("T_VMIN", "VMIN", "UPPER", 0.85, "V", "SYNTHETIC_ASSUMED"),
    ExpectedLimit("T_VMAX", "VMAX", "LOWER", 1.15, "V", "SYNTHETIC_ASSUMED"),
    ExpectedLimit("T_IDDQ", "IDDQ", "UPPER", 50.0, "uA", "SYNTHETIC_ASSUMED"),
    ExpectedLimit("T_SUPPLY_CURRENT", "SUPPLY_CURRENT", "UPPER", 120.0, "mA", "SYNTHETIC_ASSUMED"),
    ExpectedLimit("T_CONTACT_R", "CONTACT_RESISTANCE", "UPPER", 5.0, "ohm", "SYNTHETIC_ASSUMED"),
    ExpectedLimit(
        "T_INTERCONNECT_R", "INTERCONNECT_RESISTANCE", "UPPER", 15.0, "ohm", "SYNTHETIC_ASSUMED"
    ),
    ExpectedLimit("T_ON_R", "ON_RESISTANCE", "UPPER", 25.0, "ohm", "SYNTHETIC_ASSUMED"),
)

EXPECTED_CONDITION_IDS = frozenset(
    {"COND_RT_NOM", "COND_HOT_NOM", "COND_RT_LOWV", "COND_HOT_HIGHV"}
)

CORE_MEASUREMENT_PK = ("lot_id", "die_id", "pattern_id", "test_id")
PARAMETRIC_MEASUREMENT_PK = ("lot_id", "die_id", "condition_id", "test_id")

CORE_EXPECTED_LOT_COUNT = 31
CORE_EXPECTED_DIE_COUNT = 1550
CORE_EXPECTED_MEASUREMENT_ROWS = 1_550_000

PARAMETRIC_EXPECTED_LOT_COUNT = 43
PARAMETRIC_EXPECTED_DIE_COUNT = 2150
PARAMETRIC_EXPECTED_MEASUREMENT_ROWS = 60_200
PARAMETRIC_EXPECTED_CONDITION_COUNT = 4

LINKED_LOT_COUNT = 31
LINKED_DIE_COUNT = 1550
PARAMETRIC_ONLY_LOT_COUNT = 12
PARAMETRIC_ONLY_DIE_COUNT = 600
