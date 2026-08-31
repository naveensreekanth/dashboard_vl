import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Datasets used for training, temporal validation, and Month 12 inference.
# Month 12 inference uses ONLY the inference workbook below.
DATA_FILES = {
    "month_0": os.path.join(WORKSPACE_DIR, "ATE_Retest_50_Devices_Month_0_Historical.xlsx"),
    "month_6": os.path.join(WORKSPACE_DIR, "ATE_Retest_50_Devices_Month_6_Historical.xlsx"),
    "month_12": os.path.join(WORKSPACE_DIR, "ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx"),
    "ai_dataset": os.path.join(WORKSPACE_DIR, "ATE_Retest_50_Devices_AI_Dataset.xlsx"),
}

# Optional post-outcome file. NEVER used as prediction input.
MONTH_12_OUTCOMES_FILE = os.path.join(WORKSPACE_DIR, "Month_12_PRIVATE_VALIDATION_ONLY.xlsx")
if not os.path.exists(MONTH_12_OUTCOMES_FILE):
    MONTH_12_OUTCOMES_FILE = os.path.join(
        os.path.dirname(WORKSPACE_DIR), "Month_12_PRIVATE_VALIDATION_ONLY.xlsx"
    )

# Configurable ATE tester-time rate for cost KPIs. Not present in the workbooks
# and not a measured plant rate — override in the Decision Policy screen.
ATE_COST_PER_HOUR = 1800.0
ATE_COST_CURRENCY = "USD"

# Evaluation/reporting cutoff for model comparison tables only.
# This is NOT the operational decision policy (see decision/decision_policy.py).
EVAL_REPORTING_CUTOFF = 0.5

RANDOM_SEED = 42
MODEL_VERSION = "retest_option_b_v1"

# Optional Recursive Least Squares online calibration.
# Does not retrain or modify the primary classifier.
RLS_FORGETTING_FACTOR = 0.995
RLS_MIN_UPDATES_BEFORE_ACTIVE = 20
RLS_INITIAL_P_SCALE = 100.0

# Target Definition
TARGET_COL = "Ground_Truth"
TARGET_MAPPING = {
    "RETEST_BENEFICIAL": 1,
    "PERSISTENT_FAILURE": 0
}
TARGET_INV_MAPPING = {1: "RETEST_BENEFICIAL", 0: "PERSISTENT_FAILURE"}

# Confirmed Pre-Retest Feature Whitelist (Strictly observable before retest)
CATEGORICAL_FEATURES = [
    "Wafer_ID",
    "ATE_Site",
    "Fail_Test",
    "Fail_Bin",
    "First_Result"
]

NUMERICAL_FEATURES = [
    "Voltage_V",
    "Temperature_C",
    "First_Test_Time_sec",
    "Test_Month"
]

ALL_MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Excluded Fields (Metadata only - Never enter model feature matrix)
IDENTIFIER_COLS = [
    "Device_ID",
    "Failure_Event"
]

# Excluded Physical Measurement (Post-retest elapsed time - Never enters model feature matrix)
EXCLUDED_POST_RETEST_COLS = [
    "Retest_Time_sec"
]

# Leakage Blacklist (Strictly blocked from model feature matrix)
LEAKAGE_COLS = [
    "Ground_Truth",
    "Retest_Result",
    "Final_Result",
    "Retest_Count",
    "True_Retest_Pass_Probability",
    "AI_Retest_Probability",
    "AI_Recommendation"
]

# All Excluded Columns
ALL_EXCLUDED_COLS = IDENTIFIER_COLS + EXCLUDED_POST_RETEST_COLS + LEAKAGE_COLS

# Columns that must never appear in the prediction feature matrix X
FEATURE_MATRIX_FORBIDDEN_COLS = LEAKAGE_COLS + EXCLUDED_POST_RETEST_COLS

# Supplied Report Reference Baseline KPIs (RETEST~2.docx - Preserved verbatim & isolated)
DOCX_REFERENCE_KPIS = {
    "source": "RETEST~2.docx (Supplied Report Reference Baseline on 125 events)",
    "accuracy": 0.704,
    "precision": 0.699,
    "recall": 0.829,
    "specificity": 0.545,
    "unnecessary_retests_count": 25,
    "unnecessary_retests_pct": 20.0,
    "unnecessary_retest_time_sec": 853.0,
    "missed_opportunities_count": 12,
    "missed_opportunities_pct": 9.6,
    "confusion_matrix": {
        "TP": 58,
        "FP": 25,
        "FN": 12,
        "TN": 30
    },
    "test_type_breakdown": {
        "Scan": {"events": 50, "retest": 50, "skip": 0, "accuracy": 0.720, "policy": "100% RETEST", "unnecessary": 14, "missed": 0},
        "Func": {"events": 22, "retest": 15, "skip": 7, "accuracy": 0.591, "policy": "Mixed", "unnecessary": 6, "missed": 3},
        "MBIST": {"events": 22, "retest": 0, "skip": 22, "accuracy": 0.682, "policy": "100% DON'T RETEST", "unnecessary": 0, "missed": 7},
        "IDDQ": {"events": 16, "retest": 0, "skip": 16, "accuracy": 0.812, "policy": "100% DON'T RETEST", "unnecessary": 0, "missed": 2},
        "AtSpeed": {"events": 15, "retest": 11, "skip": 4, "accuracy": 0.733, "policy": "Mixed", "unnecessary": 3, "missed": 0}
    },
    "wafer_breakdown": {
        "W001": {"events": 21, "accuracy": 0.619, "missed": 3, "unnecessary": 5},
        "W002": {"events": 23, "accuracy": 0.739, "missed": 3, "unnecessary": 3},
        "W003": {"events": 24, "accuracy": 0.583, "missed": 2, "unnecessary": 8},
        "W004": {"events": 27, "accuracy": 0.889, "missed": 0, "unnecessary": 3},
        "W005": {"events": 30, "accuracy": 0.667, "missed": 4, "unnecessary": 6}
    }
}
