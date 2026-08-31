import unittest
import pandas as pd
from retest_ai.kpis.decision_quality import calculate_decision_quality
from retest_ai.validation.outcome_validator import validate_recommendations_against_outcomes
from retest_ai.kpis.business_impact import (
    build_retest_time_lookup,
    calculate_time_and_cost_impact,
    estimate_retest_time_seconds,
    format_money,
    seconds_to_cost,
)


class TestDecisionQualityKPIs(unittest.TestCase):
    def setUp(self):
        # TP, FP, FN, TN = 2, 1, 1, 2  (total 6)
        self.y_true = [
            "RETEST_BENEFICIAL",
            "RETEST_BENEFICIAL",
            "PERSISTENT_FAILURE",
            "RETEST_BENEFICIAL",
            "PERSISTENT_FAILURE",
            "PERSISTENT_FAILURE",
        ]
        self.recs = [
            "RETEST",
            "RETEST",
            "RETEST",
            "DON'T RETEST",
            "DON'T RETEST",
            "DON'T RETEST",
        ]

    def test_confusion_and_named_errors(self):
        q = calculate_decision_quality(self.y_true, self.recs)
        self.assertEqual(q["tp"], 2)
        self.assertEqual(q["fp"], 1)
        self.assertEqual(q["fn"], 1)
        self.assertEqual(q["tn"], 2)
        self.assertEqual(q["correct_retests"], 2)
        self.assertEqual(q["unnecessary_retests"], 1)
        self.assertEqual(q["missed_opportunities"], 1)
        self.assertEqual(q["correct_skips"], 2)

    def test_accuracy_precision_recall_specificity(self):
        q = calculate_decision_quality(self.y_true, self.recs)
        self.assertAlmostEqual(q["accuracy"], 4 / 6)
        self.assertAlmostEqual(q["precision"], 2 / 3)
        self.assertAlmostEqual(q["recall"], 2 / 3)
        self.assertAlmostEqual(q["specificity"], 2 / 3)
        self.assertAlmostEqual(q["unnecessary_retests_pct"], 1 / 6 * 100)
        self.assertAlmostEqual(q["missed_opportunities_pct"], 1 / 6 * 100)

    def test_rejects_raw_probabilities(self):
        with self.assertRaises(ValueError):
            calculate_decision_quality(self.y_true, [0.8, 0.2, 0.4, 0.1, 0.9, 0.05])

    def test_outcome_validator_event_split(self):
        events = pd.DataFrame({
            "Device_ID": [f"D{i}" for i in range(6)],
            "Failure_Event": list(range(6)),
        })
        result = validate_recommendations_against_outcomes(self.y_true, self.recs, events=events)
        self.assertEqual(len(result["correct_retest_events"]), 2)
        self.assertEqual(len(result["unnecessary_retest_events"]), 1)
        self.assertEqual(len(result["missed_opportunity_events"]), 1)
        self.assertEqual(len(result["correct_skip_events"]), 2)


class TestRetestCostImpact(unittest.TestCase):
    def test_seconds_to_cost_uses_hourly_rate(self):
        self.assertAlmostEqual(seconds_to_cost(3600.0, 1800.0), 1800.0)
        self.assertAlmostEqual(seconds_to_cost(15.0, 1800.0), 7.5)
        self.assertEqual(seconds_to_cost(100.0, 0.0), 0.0)
        self.assertEqual(format_money(1234.5), "$1,234.50")

    def test_lookup_prefers_fail_test_mean_then_overall_then_first_test_time(self):
        history = pd.DataFrame({
            "Fail_Test": ["Scan_1", "Scan_1", "MBIST_1"],
            "Retest_Time_sec": [10.0, 20.0, 40.0],
        })
        lookup = build_retest_time_lookup(history)
        self.assertEqual(lookup["sample_count"], 3)
        self.assertAlmostEqual(lookup["by_fail_test"]["Scan_1"], 15.0)
        self.assertAlmostEqual(lookup["overall_mean_sec"], 70.0 / 3.0)

        events = pd.DataFrame({
            "Fail_Test": ["Scan_1", "UNKNOWN", "UNKNOWN"],
            "First_Test_Time_sec": [99.0, 99.0, 12.0],
        })
        estimated = estimate_retest_time_seconds(events, lookup)
        self.assertAlmostEqual(estimated.iloc[0], 15.0)
        self.assertAlmostEqual(estimated.iloc[1], 70.0 / 3.0)
        empty_lookup = build_retest_time_lookup(pd.DataFrame({"Fail_Test": ["A"]}))
        fallback = estimate_retest_time_seconds(events, empty_lookup)
        self.assertAlmostEqual(fallback.iloc[2], 12.0)

    def test_all_device_cost_vs_ai_predicted_cost(self):
        df = pd.DataFrame({
            "AI_Recommendation": ["RETEST", "RETEST", "DON'T RETEST"],
            "Estimated_Retest_Time_sec": [10.0, 20.0, 30.0],
            "Ground_Truth": ["RETEST_BENEFICIAL", "PERSISTENT_FAILURE", "PERSISTENT_FAILURE"],
        })
        impact = calculate_time_and_cost_impact(df, cost_per_hour=1800.0)
        self.assertAlmostEqual(impact["all_device_retest_time_sec"], 60.0)
        self.assertAlmostEqual(impact["ai_predicted_retest_time_sec"], 30.0)
        self.assertAlmostEqual(impact["skipped_retest_time_sec"], 30.0)
        self.assertAlmostEqual(impact["all_device_retest_cost"], 30.0)
        self.assertAlmostEqual(impact["ai_predicted_retest_cost"], 15.0)
        self.assertAlmostEqual(impact["estimated_savings"], 15.0)
        self.assertAlmostEqual(impact["unnecessary_retest_time_sec"], 20.0)
        self.assertAlmostEqual(impact["unnecessary_retest_cost"], 10.0)

    def test_dont_retest_events_add_zero_ai_cost(self):
        df = pd.DataFrame({
            "AI_Recommendation": ["DON'T RETEST", "DON'T RETEST"],
            "Estimated_Retest_Time_sec": [50.0, 50.0],
        })
        impact = calculate_time_and_cost_impact(df, cost_per_hour=3600.0)
        self.assertAlmostEqual(impact["all_device_retest_cost"], 100.0)
        self.assertAlmostEqual(impact["ai_predicted_retest_cost"], 0.0)
        self.assertAlmostEqual(impact["estimated_savings"], 100.0)


if __name__ == "__main__":
    unittest.main()
