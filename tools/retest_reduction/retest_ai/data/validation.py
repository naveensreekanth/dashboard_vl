import pandas as pd
from typing import Dict, Any, List
from ..config.settings import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    LEAKAGE_COLS,
    ALL_EXCLUDED_COLS,
    FEATURE_MATRIX_FORBIDDEN_COLS,
    TARGET_COL,
    TARGET_MAPPING,
    IDENTIFIER_COLS,
    EXCLUDED_POST_RETEST_COLS
)

class ValidationError(Exception):
    """Custom exception raised when dataset validation fails."""
    pass


def log_feature_whitelist_and_exclusions() -> None:
    """Logs the final confirmed feature whitelist and excluded columns."""
    print("\n" + "=" * 40)
    print("FINAL MODEL FEATURES (WHITELIST)")
    print("=" * 40)
    for feat in ALL_MODEL_FEATURES:
        print(f"  • {feat}")

    print("\n" + "=" * 40)
    print("EXCLUDED FEATURES (LEAKAGE / TRACKING)")
    print("=" * 40)
    for excl in ALL_EXCLUDED_COLS:
        print(f"  ✕ {excl}")
    print("=" * 40 + "\n")


def validate_pre_retest_upload(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a pre-retest workbook against the existing model feature schema.

    Required columns are ALL_MODEL_FEATURES. Outcome/ground-truth columns are
    recorded and excluded from model input; they are not used as features.
    Missing required features are reported explicitly and are not filled.
    """
    required_columns = list(ALL_MODEL_FEATURES)
    excluded_outcome_columns = []
    if df is None or len(df) == 0:
        return {
            "is_valid": False,
            "required_columns": required_columns,
            "missing_columns": list(required_columns),
            "excluded_outcome_columns": excluded_outcome_columns,
            "errors": ["Pre-retest dataset is empty."],
            "warnings": [],
        }

    missing_columns = [c for c in required_columns if c not in df.columns]
    excluded_outcome_columns = [c for c in FEATURE_MATRIX_FORBIDDEN_COLS if c in df.columns]
    if missing_columns:
        return {
            "is_valid": False,
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "excluded_outcome_columns": excluded_outcome_columns,
            "errors": [
                "Missing required columns: "
                + ", ".join(missing_columns)
                + ". Required columns: "
                + ", ".join(required_columns)
                + "."
            ],
            "warnings": [],
        }

    df_for_model = df.drop(columns=excluded_outcome_columns) if excluded_outcome_columns else df
    report = validate_dataset(df_for_model, is_inference=True, strict_leakage_check=True)
    report["required_columns"] = required_columns
    report["missing_columns"] = []
    report["excluded_outcome_columns"] = excluded_outcome_columns
    if excluded_outcome_columns:
        report.setdefault("warnings", [])
        report["warnings"] = list(report["warnings"]) + [
            "Outcome/ground-truth columns were excluded from model input: "
            + ", ".join(excluded_outcome_columns)
        ]
    return report


def format_pre_retest_validation_error(report: Dict[str, Any]) -> str:
    """Human-readable validation error listing required and missing columns."""
    errors = list(report.get("errors") or [])
    if errors:
        return " ".join(str(e) for e in errors)
    missing = report.get("missing_columns") or []
    required = report.get("required_columns") or list(ALL_MODEL_FEATURES)
    if missing:
        return (
            "Missing required columns: "
            + ", ".join(missing)
            + ". Required columns: "
            + ", ".join(required)
            + "."
        )
    return "Pre-retest workbook is invalid."


def assert_no_leakage_in_feature_matrix(columns) -> None:
    """Fail clearly if a leakage or post-retest column is in the prediction matrix."""
    present = [c for c in columns if c in FEATURE_MATRIX_FORBIDDEN_COLS]
    if present:
        raise ValidationError(
            f"Leakage/post-retest columns passed into the prediction feature matrix: {present}"
        )
    overlap = [c for c in ALL_MODEL_FEATURES if c in FEATURE_MATRIX_FORBIDDEN_COLS]
    if overlap:
        raise ValidationError(
            f"Feature whitelist illegally includes forbidden columns: {overlap}"
        )


def validate_dataset(
    df: pd.DataFrame,
    is_inference: bool = False,
    strict_leakage_check: bool = True
) -> Dict[str, Any]:
    """
    Validates a dataset against schema, data types, missing values, duplicates, and leakage.
    Does not silently alter data.
    """
    errors: List[str] = []
    warnings: List[str] = []
    missing_features: List[str] = []
    leakage_detected: List[str] = []

    if df is None or len(df) == 0:
        errors.append("Dataset is empty.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "missing_features": missing_features,
            "leakage_detected": leakage_detected,
            "row_count": 0,
            "device_count": 0
        }

    for feat in ALL_MODEL_FEATURES:
        if feat not in df.columns:
            missing_features.append(feat)
    if missing_features:
        errors.append(f"Missing required model features: {', '.join(missing_features)}")

    found_leakage = [c for c in df.columns if c in LEAKAGE_COLS]
    if is_inference and found_leakage:
        leakage_detected = found_leakage
        msg = f"Inference leakage detected! Forbidden outcome columns present: {', '.join(found_leakage)}"
        if strict_leakage_check:
            errors.append(msg)
        else:
            warnings.append(msg)

    post_retest_present = [c for c in df.columns if c in EXCLUDED_POST_RETEST_COLS]
    if post_retest_present:
        warnings.append(
            f"Post-retest columns present in workbook but excluded from model features: {post_retest_present}"
        )

    if not is_inference:
        if TARGET_COL not in df.columns:
            errors.append(f"Training dataset missing required target column '{TARGET_COL}'.")
        else:
            valid_targets = set(TARGET_MAPPING.keys())
            present_targets = set(df[TARGET_COL].dropna().unique())
            invalid_targets = present_targets - valid_targets
            if invalid_targets:
                errors.append(f"Invalid target labels detected: {invalid_targets}. Expected {valid_targets}")

    present_feats = [c for c in ALL_MODEL_FEATURES if c in df.columns]
    null_counts = df[present_feats].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_str = ", ".join([f"{col}: {cnt}" for col, cnt in cols_with_nulls.items()])
        errors.append(f"Missing values found in model features: {null_str}")

    if all(col in df.columns for col in IDENTIFIER_COLS):
        dup_count = int(df.duplicated(subset=IDENTIFIER_COLS).sum())
        if dup_count > 0:
            errors.append(
                f"Duplicate device failure events found: {dup_count} duplicated (Device_ID, Failure_Event) pairs."
            )

    if "First_Result" in df.columns:
        invalid_first = set(df["First_Result"].dropna().astype(str).str.strip().unique()) - {"FAIL"}
        if invalid_first:
            warnings.append(f"Unexpected First_Result values: {sorted(invalid_first)}")

    if "Test_Month" in df.columns:
        months = pd.to_numeric(df["Test_Month"], errors="coerce")
        invalid_months = sorted({int(m) for m in months.dropna().unique() if int(m) not in {0, 6, 12}})
        if invalid_months:
            errors.append(f"Invalid Test_Month values: {invalid_months}. Expected 0, 6, or 12.")

    for col in NUMERICAL_FEATURES:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any() and not df[col].isna().any():
            errors.append(f"Non-numeric values in {col}.")
        if col == "Voltage_V" and (numeric.dropna() <= 0).any():
            errors.append("Voltage_V must be > 0.")
        if col == "First_Test_Time_sec" and (numeric.dropna() < 0).any():
            errors.append("First_Test_Time_sec must be >= 0.")

    device_count = df["Device_ID"].nunique() if "Device_ID" in df.columns else 0
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "missing_features": missing_features,
        "leakage_detected": leakage_detected,
        "row_count": len(df),
        "device_count": device_count
    }


def check_for_leakage(df: pd.DataFrame, is_inference: bool = True) -> List[str]:
    """Helper that returns any leakage column present in DataFrame."""
    return [col for col in df.columns if col in LEAKAGE_COLS]


def validate_feature_consistency(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Checks that required model features exist across Month 0 / 6 / 12."""
    errors = []
    for name, df in datasets.items():
        missing = [c for c in ALL_MODEL_FEATURES if c not in df.columns]
        if missing:
            errors.append(f"{name} missing features: {missing}")
    return {"is_valid": len(errors) == 0, "errors": errors}
