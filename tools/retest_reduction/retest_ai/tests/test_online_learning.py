import unittest

import numpy as np
import pandas as pd

from retest_ai.models.online_learning import (
    ADAPTED_PROB_COL,
    BASE_PROB_COL,
    RLSCalibrator,
    encode_ground_truth,
)
from retest_ai.models.service import MLService


def _validated_frame(rows):
    return pd.DataFrame(rows)


class TestRLSCalibrator(unittest.TestCase):
    def setUp(self):
        self.cal = RLSCalibrator(
            forgetting_factor=0.995,
            min_updates_before_active=20,
            initial_p_scale=100.0,
        )

    def test_new_calibrator_starts_inactive(self):
        status = self.cal.status()
        self.assertTrue(status["initialized"])
        self.assertFalse(status["active"])
        self.assertEqual(status["update_count"], 0)
        self.assertEqual(status["activation_threshold"], 20)
        self.assertEqual(status["learned_dataset_count"], 0)
        self.assertEqual(status["forgetting_factor"], 0.995)

    def test_base_probability_passes_through_before_activation(self):
        base = np.array([0.12, 0.47, 0.91])
        adapted = self.cal.adapt_probability(base)
        np.testing.assert_array_equal(adapted, base)
        self.assertEqual(self.cal.adapt_probability(0.73), 0.73)

    def test_rls_updates_after_valid_observations(self):
        df = _validated_frame(
            [
                {"Device_ID": "DEV001", "Failure_Event": 1, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.80},
                {"Device_ID": "DEV002", "Failure_Event": 2, "Ground_Truth": "PERSISTENT_FAILURE", BASE_PROB_COL: 0.20},
                {"Device_ID": "DEV003", "Failure_Event": 3, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.65},
            ]
        )
        result = self.cal.update_from_validated_frame(df)
        self.assertFalse(result["already_learned"])
        self.assertEqual(result["learned"], 3)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(self.cal.update_count, 3)
        self.assertFalse(self.cal.is_active)
        self.assertNotEqual(list(self.cal.weights.ravel()), [0.0, 1.0])

    def test_activation_occurs_after_configured_minimum_updates(self):
        cal = RLSCalibrator(min_updates_before_active=5)
        rows = []
        for i in range(5):
            rows.append(
                {
                    "Device_ID": f"DEV{i:03d}",
                    "Failure_Event": i,
                    "Ground_Truth": "RETEST_BENEFICIAL" if i % 2 == 0 else "PERSISTENT_FAILURE",
                    BASE_PROB_COL: 0.4 + 0.1 * (i % 3),
                }
            )
        result = cal.update_from_validated_frame(_validated_frame(rows))
        self.assertEqual(result["learned"], 5)
        self.assertTrue(result["active"])
        self.assertTrue(cal.is_active)
        adapted = cal.adapt_probability(0.55)
        self.assertIsInstance(adapted, float)
        self.assertGreaterEqual(adapted, 0.0)
        self.assertLessEqual(adapted, 1.0)

    def test_adapted_probability_remains_between_zero_and_one(self):
        cal = RLSCalibrator(min_updates_before_active=2)
        df = _validated_frame(
            [
                {"Device_ID": "A", "Failure_Event": 1, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.99},
                {"Device_ID": "B", "Failure_Event": 2, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.98},
            ]
        )
        cal.update_from_validated_frame(df)
        self.assertTrue(cal.is_active)
        cal.weights = np.array([[50.0], [50.0]], dtype=float)
        adapted = cal.adapt_probability(np.array([-0.2, 0.5, 1.4]))
        self.assertTrue(np.all(adapted >= 0.0))
        self.assertTrue(np.all(adapted <= 1.0))

    def test_invalid_ground_truth_rows_are_skipped(self):
        df = _validated_frame(
            [
                {"Device_ID": "DEV001", "Failure_Event": 1, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.70},
                {"Device_ID": "DEV002", "Failure_Event": 2, "Ground_Truth": None, BASE_PROB_COL: 0.40},
                {"Device_ID": "DEV003", "Failure_Event": 3, "Ground_Truth": "UNKNOWN_LABEL", BASE_PROB_COL: 0.50},
                {"Device_ID": "DEV004", "Failure_Event": 4, "Ground_Truth": "PERSISTENT_FAILURE", BASE_PROB_COL: np.nan},
                {"Device_ID": "DEV005", "Failure_Event": 5, "Ground_Truth": "  retest_beneficial  ", BASE_PROB_COL: 0.55},
            ]
        )
        result = self.cal.update_from_validated_frame(df)
        self.assertEqual(result["learned"], 2)
        self.assertEqual(result["skipped"], 3)
        self.assertEqual(self.cal.update_count, 2)
        self.assertEqual(encode_ground_truth("RETEST_NOT_BENEFICIAL"), 0)
        self.assertIsNone(encode_ground_truth("not a label"))

    def test_same_dataset_cannot_be_learned_twice(self):
        df = _validated_frame(
            [
                {"Device_ID": "DEV010", "Failure_Event": 1, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.61},
                {"Device_ID": "DEV011", "Failure_Event": 2, "Ground_Truth": "PERSISTENT_FAILURE", BASE_PROB_COL: 0.22},
            ]
        )
        first = self.cal.update_from_validated_frame(df)
        weights_after_first = self.cal.weights.copy()
        second = self.cal.update_from_validated_frame(df.copy())
        self.assertEqual(first["learned"], 2)
        self.assertTrue(second["already_learned"])
        self.assertEqual(second["learned"], 0)
        self.assertEqual(self.cal.update_count, 2)
        np.testing.assert_array_equal(self.cal.weights, weights_after_first)
        self.assertEqual(self.cal.status()["learned_dataset_count"], 1)

    def test_reset_clears_rls_learning_state(self):
        df = _validated_frame(
            [
                {"Device_ID": "DEV020", "Failure_Event": 1, "Ground_Truth": "RETEST_BENEFICIAL", BASE_PROB_COL: 0.77},
            ]
        )
        self.cal.update_from_validated_frame(df)
        self.cal.reset()
        status = self.cal.status()
        self.assertEqual(status["update_count"], 0)
        self.assertFalse(status["active"])
        self.assertEqual(status["learned_dataset_count"], 0)
        np.testing.assert_array_equal(self.cal.weights, np.array([[0.0], [1.0]]))
        np.testing.assert_array_equal(self.cal.P, np.eye(2) * 100.0)
        again = self.cal.update_from_validated_frame(df)
        self.assertFalse(again["already_learned"])
        self.assertEqual(again["learned"], 1)

    def test_learning_uses_base_probability_not_adapted_column(self):
        cal = RLSCalibrator(min_updates_before_active=1)
        df = _validated_frame(
            [
                {
                    "Device_ID": "DEV030",
                    "Failure_Event": 1,
                    "Ground_Truth": "RETEST_BENEFICIAL",
                    BASE_PROB_COL: 0.25,
                    "P(RETEST_BENEFICIAL)": 0.90,
                }
            ]
        )
        cal.update_from_validated_frame(df)
        reference = RLSCalibrator(min_updates_before_active=1)
        reference.update_from_validated_frame(
            _validated_frame(
                [
                    {
                        "Device_ID": "DEV030",
                        "Failure_Event": 1,
                        "Ground_Truth": "RETEST_BENEFICIAL",
                        BASE_PROB_COL: 0.25,
                    }
                ]
            )
        )
        np.testing.assert_allclose(cal.weights, reference.weights)


class TestMLServiceOnlineLearning(unittest.TestCase):
    def setUp(self):
        self.ml = MLService.get_instance()
        self.ml.reset_online_learning()

    def tearDown(self):
        self.ml.reset_online_learning()

    def test_existing_prediction_flow_unchanged_when_rls_inactive(self):
        status = self.ml.get_online_learning_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["update_count"], 0)
        self.assertIsNone(self.ml.month_12_outcomes)

        table = self.ml.get_month_12_batch_table()
        self.assertIn("P(RETEST_BENEFICIAL)", table.columns)
        self.assertIn(BASE_PROB_COL, table.columns)
        self.assertIn(ADAPTED_PROB_COL, table.columns)
        pd.testing.assert_series_equal(
            table["P(RETEST_BENEFICIAL)"],
            table[BASE_PROB_COL],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            table[ADAPTED_PROB_COL],
            table[BASE_PROB_COL],
            check_names=False,
        )
        self.assertTrue((table["P(RETEST_BENEFICIAL)"] >= 0).all())
        self.assertTrue((table["P(RETEST_BENEFICIAL)"] <= 1).all())

        event = table.iloc[0]
        pred = self.ml.predict_single_event(
            {
                "Device_ID": event["Device_ID"],
                "Failure_Event": int(event["Failure_Event"]),
                "Wafer_ID": event["Wafer_ID"],
                "ATE_Site": int(event["ATE_Site"]),
                "Fail_Test": event["Fail_Test"],
                "Fail_Bin": int(event["Fail_Bin"]),
                "First_Result": event["First_Result"],
                "Voltage_V": float(event["Voltage_V"]),
                "Temperature_C": float(event["Temperature_C"]),
                "First_Test_Time_sec": float(event["First_Test_Time_sec"]),
                "Test_Month": int(event["Test_Month"]),
            }
        )
        self.assertFalse(pred["online_adaptation_active"])
        self.assertEqual(pred["probability_retest_beneficial"], pred["probability_base"])
        self.assertEqual(pred["probability_adapted"], pred["probability_base"])

    def test_learning_does_not_set_month_12_outcomes_or_change_historical_validation(self):
        historical_before = self.ml.get_historical_validation_table()["P(RETEST_BENEFICIAL)"].copy()
        scored = self.ml.get_month_12_batch_table().head(8).copy()
        scored["Ground_Truth"] = ["RETEST_BENEFICIAL", "PERSISTENT_FAILURE"] * 4
        result = self.ml.update_from_validated_outcomes(scored)
        self.assertGreater(result["learned"], 0)
        self.assertIsNone(self.ml.month_12_outcomes)
        self.assertFalse(self.ml.month_12_outcomes_available())
        historical_after = self.ml.get_historical_validation_table()["P(RETEST_BENEFICIAL)"]
        pd.testing.assert_series_equal(historical_before, historical_after, check_names=False)


if __name__ == "__main__":
    unittest.main()
