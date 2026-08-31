import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from ..config.settings import ATE_COST_CURRENCY, ATE_COST_PER_HOUR
from ..decision.decision_policy import DONT_RETEST_LABEL, RETEST_LABEL

ESTIMATED_TIME_COL = "Estimated_Retest_Time_sec"
ACTUAL_TIME_COL = "Retest_Time_sec"


def ate_cost_per_second(cost_per_hour: float) -> float:
    return max(float(cost_per_hour), 0.0) / 3600.0


def seconds_to_cost(seconds: float, cost_per_hour: float) -> float:
    return float(seconds) * ate_cost_per_second(cost_per_hour)


def format_money(amount: float, currency: str = ATE_COST_CURRENCY) -> str:
    symbol = "$" if str(currency).upper() == "USD" else f"{currency} "
    sign = "-" if amount < 0 else ""
    return f"{sign}{symbol}{abs(amount):,.2f}"


def format_seconds(seconds: float) -> str:
    sec = float(seconds)
    return f"{sec:,.1f} s ({sec / 3600.0:.3f} h)"


def build_retest_time_lookup(
    historical_df: Optional[pd.DataFrame],
    time_col: str = ACTUAL_TIME_COL,
    group_col: str = "Fail_Test",
) -> Dict[str, Any]:
    """
    Mean actual retest duration by Fail_Test from labeled history.
    Used only to estimate duration before a physical retest runs.
    """
    empty = {
        "by_fail_test": {},
        "overall_mean_sec": None,
        "sample_count": 0,
        "time_col": time_col,
    }
    if historical_df is None or len(historical_df) == 0 or time_col not in historical_df.columns:
        return empty

    times = pd.to_numeric(historical_df[time_col], errors="coerce")
    valid_mask = times.notna()
    if not valid_mask.any():
        return empty

    overall = float(times.loc[valid_mask].mean())
    by_test: Dict[str, float] = {}
    if group_col in historical_df.columns:
        tmp = historical_df.loc[valid_mask, [group_col]].copy()
        tmp["_t"] = times.loc[valid_mask].to_numpy()
        grouped = tmp.groupby(group_col, dropna=False)["_t"].mean()
        by_test = {str(k): float(v) for k, v in grouped.items() if pd.notna(v)}

    return {
        "by_fail_test": by_test,
        "overall_mean_sec": overall,
        "sample_count": int(valid_mask.sum()),
        "time_col": time_col,
    }


def estimate_retest_time_seconds(
    df: pd.DataFrame,
    lookup: Optional[Dict[str, Any]] = None,
    first_test_time_col: str = "First_Test_Time_sec",
    fail_test_col: str = "Fail_Test",
) -> pd.Series:
    """
    Pre-retest duration estimate: Fail_Test historical mean, then overall mean,
    then First_Test_Time_sec. Never reads actual Retest_Time_sec from the scored frame.
    """
    lookup = lookup or {}
    by_test = lookup.get("by_fail_test") or {}
    overall = lookup.get("overall_mean_sec")
    out = pd.Series(np.nan, index=df.index, dtype=float)

    if fail_test_col in df.columns and by_test:
        mapped = df[fail_test_col].astype(str).map(by_test)
        out = pd.to_numeric(mapped, errors="coerce")

    if overall is not None:
        out = out.fillna(float(overall))

    if first_test_time_col in df.columns:
        first = pd.to_numeric(df[first_test_time_col], errors="coerce")
        out = out.fillna(first)

    return out.fillna(0.0).clip(lower=0.0).astype(float)


def attach_estimated_retest_times(
    df: pd.DataFrame,
    lookup: Optional[Dict[str, Any]] = None,
    col: str = ESTIMATED_TIME_COL,
) -> pd.DataFrame:
    out = df.copy()
    out[col] = estimate_retest_time_seconds(out, lookup)
    return out


def calculate_time_and_yield_impact(
    df: pd.DataFrame,
    recommendations_col: str = "Recommendation",
    ground_truth_col: str = "Ground_Truth",
    retest_time_col: str = "Retest_Time_sec"
) -> Dict[str, Any]:
    """
    Computes business and operational impact in ATE test seconds and device counts.
    Currency conversion is handled separately by calculate_time_and_cost_impact.
    """
    total_events = len(df)
    has_gt = ground_truth_col in df.columns
    has_time = retest_time_col in df.columns

    recs = df[recommendations_col].astype(str).str.strip() if recommendations_col in df.columns else pd.Series("", index=df.index)
    retest_recs = recs == RETEST_LABEL
    skip_recs = recs == DONT_RETEST_LABEL

    total_retests_recommended = int(retest_recs.sum())
    total_skips_recommended = int(skip_recs.sum())

    total_actual_retest_time = float(df[retest_time_col].sum()) if has_time else 0.0
    recommended_retest_time = float(df.loc[retest_recs, retest_time_col].sum()) if has_time else 0.0

    unnecessary_retests_count = 0
    unnecessary_retest_time_sec = 0.0
    missed_recoverable_devices = 0
    correctly_recovered_devices = 0

    if has_gt:
        gt_beneficial = (df[ground_truth_col] == "RETEST_BENEFICIAL")
        gt_persistent = (df[ground_truth_col] == "PERSISTENT_FAILURE")

        fp_mask = retest_recs & gt_persistent
        tp_mask = retest_recs & gt_beneficial
        fn_mask = skip_recs & gt_beneficial

        unnecessary_retests_count = int(fp_mask.sum())
        correctly_recovered_devices = int(tp_mask.sum())
        missed_recoverable_devices = int(fn_mask.sum())

        if has_time:
            unnecessary_retest_time_sec = float(df.loc[fp_mask, retest_time_col].sum())

    return {
        "total_events": total_events,
        "retest_recommendations_count": total_retests_recommended,
        "retest_recommendations_pct": (total_retests_recommended / total_events * 100.0) if total_events > 0 else 0.0,
        "skip_recommendations_count": total_skips_recommended,
        "skip_recommendations_pct": (total_skips_recommended / total_events * 100.0) if total_events > 0 else 0.0,
        "total_retest_time_sec": total_actual_retest_time,
        "ai_recommended_retest_time_sec": recommended_retest_time,
        "unnecessary_retests_count": unnecessary_retests_count,
        "unnecessary_retest_time_sec": unnecessary_retest_time_sec,
        "correctly_recovered_devices_count": correctly_recovered_devices,
        "missed_recoverable_devices_count": missed_recoverable_devices
    }


def calculate_time_and_cost_impact(
    df: pd.DataFrame,
    recommendations_col: str = "AI_Recommendation",
    time_col: str = ESTIMATED_TIME_COL,
    cost_per_hour: Optional[float] = None,
    currency: Optional[str] = None,
    ground_truth_col: str = "Ground_Truth",
) -> Dict[str, Any]:
    """
    Lot-level tester-time cost: all-device baseline vs AI-selected retests.
    Cost = time_seconds × (ATE $/hour / 3600). Does not invent a rate.
    """
    rate = max(float(ATE_COST_PER_HOUR if cost_per_hour is None else cost_per_hour), 0.0)
    curr = currency or ATE_COST_CURRENCY
    time_impact = calculate_time_and_yield_impact(
        df,
        recommendations_col=recommendations_col,
        ground_truth_col=ground_truth_col,
        retest_time_col=time_col,
    )

    all_time = float(time_impact["total_retest_time_sec"])
    ai_time = float(time_impact["ai_recommended_retest_time_sec"])
    skipped_time = all_time - ai_time
    unnec_time = float(time_impact["unnecessary_retest_time_sec"])

    all_cost = seconds_to_cost(all_time, rate)
    ai_cost = seconds_to_cost(ai_time, rate)
    return {
        **time_impact,
        "all_device_retest_time_sec": all_time,
        "ai_predicted_retest_time_sec": ai_time,
        "skipped_retest_time_sec": skipped_time,
        "cost_per_hour": rate,
        "cost_per_second": ate_cost_per_second(rate),
        "currency": curr,
        "all_device_retest_cost": all_cost,
        "ai_predicted_retest_cost": ai_cost,
        "estimated_savings": all_cost - ai_cost,
        "unnecessary_retest_cost": seconds_to_cost(unnec_time, rate),
        "time_column": time_col,
    }
