import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .decision_quality import calculate_decision_quality

__all__ = [
    "overview_recommendation_counts",
    "compute_group_breakdown",
    "filter_month12_batch_table",
]


def overview_recommendation_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Overview KPI counts from the active prediction dataframe via value_counts()."""
    if df is None or len(df) == 0 or "AI_Recommendation" not in df.columns:
        return {"total_events": 0, "retest": 0, "dont_retest": 0}
    counts = df["AI_Recommendation"].astype(str).str.strip().value_counts()
    return {
        "total_events": int(len(df)),
        "retest": int(counts.get("RETEST", 0)),
        "dont_retest": int(counts.get("DON'T RETEST", 0)),
    }


def compute_group_breakdown(
    df: pd.DataFrame,
    group_col: str,
    recommendations_col: str = "Recommendation",
    ground_truth_col: str = "Ground_Truth"
) -> pd.DataFrame:
    """
    Calculates detailed metrics grouped by a specified column (e.g. Fail_Test, Wafer_ID, ATE_Site).
    """
    if group_col not in df.columns:
        return pd.DataFrame()

    has_gt = ground_truth_col in df.columns
    groups = df[group_col].unique()
    rows = []

    for g in groups:
        sub_df = df[df[group_col] == g]
        cnt = len(sub_df)
        retests = (sub_df[recommendations_col] == "RETEST").sum()
        skips = (sub_df[recommendations_col] == "DON'T RETEST").sum()
        mean_prob = sub_df["AI_Probability"].mean() if "AI_Probability" in sub_df.columns else np.nan

        row_data = {
            group_col: g,
            "Events": cnt,
            "RETEST_Count": int(retests),
            "DONT_RETEST_Count": int(skips),
            "Mean_Probability": round(float(mean_prob), 3) if pd.notnull(mean_prob) else "-"
        }

        if has_gt:
            dq = calculate_decision_quality(sub_df[ground_truth_col], sub_df[recommendations_col])
            row_data["Accuracy"] = f"{dq['accuracy'] * 100:.1f}%"
            row_data["Precision"] = f"{dq['precision'] * 100:.1f}%"
            row_data["Recall"] = f"{dq['recall'] * 100:.1f}%"
            row_data["Unnecessary_Retests"] = dq["unnecessary_retests_count"]
            row_data["Missed_Opportunities"] = dq["missed_opportunities_count"]
            row_data["Observed_Beneficial_Rate"] = f"{(sub_df[ground_truth_col] == 'RETEST_BENEFICIAL').mean() * 100:.1f}%"

        rows.append(row_data)

    df_res = pd.DataFrame(rows)
    if "Events" in df_res.columns:
        df_res = df_res.sort_values(by="Events", ascending=False).reset_index(drop=True)
    return df_res

def get_test_type_family(fail_test_str: str) -> str:
    """
    Maps specific test names (e.g. Scan_145, MBIST_03) to high-level test family (Scan, MBIST, etc.)
    """
    s = str(fail_test_str).lower()
    if "scan" in s:
        return "Scan"
    elif "mbist" in s:
        return "MBIST"
    elif "iddq" in s:
        return "IDDQ"
    elif "func" in s:
        return "Func"
    elif "atspeed" in s or "at_speed" in s:
        return "AtSpeed"
    return str(fail_test_str)

def compute_test_family_breakdown(
    df: pd.DataFrame,
    recommendations_col: str = "Recommendation",
    ground_truth_col: str = "Ground_Truth"
) -> pd.DataFrame:
    """
    Aggregates metrics by Test Family (Scan, Func, MBIST, IDDQ, AtSpeed).
    """
    df_copy = df.copy()
    if "Fail_Test" in df_copy.columns:
        df_copy["Test_Family"] = df_copy["Fail_Test"].apply(get_test_type_family)
        return compute_group_breakdown(df_copy, "Test_Family", recommendations_col, ground_truth_col)
    return pd.DataFrame()


def filter_month12_batch_table(
    df: pd.DataFrame,
    rec_filter=None,
    test_filter=None,
    wafer_filter=None,
    site_filter=None,
    prob_range=(0.0, 1.0),
) -> pd.DataFrame:
    """
    Filter the Month 12 prediction table.

    Empty recommendation / fail-test / wafer / site selections mean All
    (do not drop rows). Probability range is always applied.
    Original row index is preserved.
    """
    out = df.copy()
    if "AI_Recommendation" in out.columns:
        out["AI_Recommendation"] = out["AI_Recommendation"].astype(str).str.strip()

    rec_filter = list(rec_filter or [])
    test_filter = list(test_filter or [])
    wafer_filter = list(wafer_filter or [])
    site_filter = list(site_filter or [])
    lo, hi = float(prob_range[0]), float(prob_range[1])

    if rec_filter:
        out = out[out["AI_Recommendation"].isin(rec_filter)]
    if test_filter and "Fail_Test" in out.columns:
        out = out[out["Fail_Test"].isin(test_filter)]
    if wafer_filter and "Wafer_ID" in out.columns:
        out = out[out["Wafer_ID"].isin(wafer_filter)]
    if site_filter and "ATE_Site" in out.columns:
        out = out[out["ATE_Site"].isin(site_filter)]
    if "P(RETEST_BENEFICIAL)" in out.columns:
        out = out[
            (out["P(RETEST_BENEFICIAL)"] >= lo)
            & (out["P(RETEST_BENEFICIAL)"] <= hi)
        ]
    return out
