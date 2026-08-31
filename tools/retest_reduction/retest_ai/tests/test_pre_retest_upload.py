import os
import tempfile
import unittest

import pandas as pd

from retest_ai.config.settings import ALL_MODEL_FEATURES, DATA_FILES, FEATURE_MATRIX_FORBIDDEN_COLS
from retest_ai.data.ingestion import load_dataset, load_pre_retest_workbook
from retest_ai.data.preprocessing import prepare_xy
from retest_ai.data.validation import format_pre_retest_validation_error, validate_pre_retest_upload
from retest_ai.kpis.breakdowns import overview_recommendation_counts
from retest_ai.models.service import MLService


class TestPreRetestUpload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml_service = MLService.get_instance()
        cls.month12 = load_dataset("month_12")

    def _write_xlsx(self, df, name="pre_retest.xlsx"):
        path = os.path.join(tempfile.gettempdir(), name)
        df.to_excel(path, index=False)
        return path

    def test_valid_pre_retest_file_can_be_loaded(self):
        path = self._write_xlsx(self.month12, "valid_pre_retest.xlsx")
        loaded = load_pre_retest_workbook(path)
        self.assertGreater(len(loaded), 0)
        for col in ALL_MODEL_FEATURES:
            self.assertIn(col, loaded.columns)
        report = validate_pre_retest_upload(loaded)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["missing_columns"], [])

        predicted = self.ml_service.load_and_predict_pre_retest_workbook(path)
        self.assertEqual(len(predicted), len(loaded))
        self.assertIn("P(RETEST_BENEFICIAL)", predicted.columns)
        self.assertIn("AI_Recommendation", predicted.columns)

    def test_file_like_xlsx_can_be_analyzed(self):
        import io
        path = self._write_xlsx(self.month12.head(5), "filelike_pre_retest.xlsx")
        with open(path, "rb") as f:
            uploaded = io.BytesIO(f.read())
        predicted = self.ml_service.load_and_predict_pre_retest_workbook(uploaded)
        self.assertEqual(len(predicted), 5)
        self.assertIn("P(RETEST_BENEFICIAL)", predicted.columns)
        self.assertIn("AI_Recommendation", predicted.columns)
        self.assertFalse(predicted["P(RETEST_BENEFICIAL)"].isna().any())
        self.assertFalse(predicted["AI_Recommendation"].astype(str).str.strip().eq("").any())
        self.assertTrue(
            set(predicted["AI_Recommendation"].astype(str).str.strip().unique()).issubset(
                {"RETEST", "DON'T RETEST"}
            )
        )

    def test_missing_required_feature_columns_lists_required_and_missing(self):
        df_missing = self.month12.drop(columns=["Temperature_C", "Voltage_V"])
        report = validate_pre_retest_upload(df_missing)
        self.assertFalse(report["is_valid"])
        self.assertIn("Temperature_C", report["missing_columns"])
        self.assertIn("Voltage_V", report["missing_columns"])
        self.assertEqual(report["required_columns"], list(ALL_MODEL_FEATURES))
        message = format_pre_retest_validation_error(report)
        self.assertIn("Temperature_C", message)
        self.assertIn("Voltage_V", message)
        self.assertIn("Required columns", message)
        for col in ALL_MODEL_FEATURES:
            self.assertIn(col, message)

        with self.assertRaises(ValueError) as ctx:
            self.ml_service.predict_pre_retest_table(df_missing)
        err = str(ctx.exception)
        self.assertIn("Temperature_C", err)
        self.assertIn("Required columns", err)

    def test_ground_truth_columns_are_not_used_as_model_features(self):
        df_with_outcomes = self.month12.copy()
        df_with_outcomes["Ground_Truth"] = "RETEST_BENEFICIAL"
        df_with_outcomes["Retest_Result"] = "PASS"
        df_with_outcomes["PERSISTENT_FAILURE"] = 0

        report = validate_pre_retest_upload(df_with_outcomes)
        self.assertTrue(report["is_valid"], report.get("errors"))
        self.assertIn("Ground_Truth", report["excluded_outcome_columns"])
        self.assertIn("Retest_Result", report["excluded_outcome_columns"])

        X, _ = prepare_xy(
            df_with_outcomes.drop(columns=[c for c in FEATURE_MATRIX_FORBIDDEN_COLS if c in df_with_outcomes.columns]),
            is_inference=True,
        )
        self.assertNotIn("Ground_Truth", X.columns)
        self.assertNotIn("Retest_Result", X.columns)
        self.assertEqual(list(X.columns), list(ALL_MODEL_FEATURES))

        predicted_with = self.ml_service.predict_pre_retest_table(df_with_outcomes)
        predicted_without = self.ml_service.predict_pre_retest_table(self.month12)
        self.assertNotIn("Ground_Truth", predicted_with.columns)
        self.assertNotIn("Retest_Result", predicted_with.columns)
        pd.testing.assert_series_equal(
            predicted_with["P(RETEST_BENEFICIAL)"].reset_index(drop=True),
            predicted_without["P(RETEST_BENEFICIAL)"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            predicted_with["AI_Recommendation"].reset_index(drop=True),
            predicted_without["AI_Recommendation"].reset_index(drop=True),
        )

    def test_prediction_output_contains_probability_and_recommendation(self):
        predicted = self.ml_service.predict_pre_retest_table(self.month12)
        self.assertIn("P(RETEST_BENEFICIAL)", predicted.columns)
        self.assertIn("AI_Recommendation", predicted.columns)
        self.assertTrue((predicted["P(RETEST_BENEFICIAL)"] >= 0).all())
        self.assertTrue((predicted["P(RETEST_BENEFICIAL)"] <= 1).all())
        self.assertTrue(
            set(predicted["AI_Recommendation"].astype(str).str.strip().unique()).issubset(
                {"RETEST", "DON'T RETEST"}
            )
        )

    def test_overview_counts_match_value_counts_of_active_prediction_dataframe(self):
        predicted = self.ml_service.predict_pre_retest_table(self.month12)
        counts = overview_recommendation_counts(predicted)
        value_counts = predicted["AI_Recommendation"].astype(str).str.strip().value_counts()
        self.assertEqual(counts["total_events"], len(predicted))
        self.assertEqual(counts["retest"], int(value_counts.get("RETEST", 0)))
        self.assertEqual(counts["dont_retest"], int(value_counts.get("DON'T RETEST", 0)))
        self.assertEqual(counts["retest"] + counts["dont_retest"], counts["total_events"])

        impact = self.ml_service.get_cost_impact(predicted, cost_per_hour=1800.0)
        self.assertGreaterEqual(impact["all_device_retest_cost"], impact["ai_predicted_retest_cost"])
        self.assertAlmostEqual(
            impact["estimated_savings"],
            impact["all_device_retest_cost"] - impact["ai_predicted_retest_cost"],
        )
        self.assertIn("Estimated_Retest_Time_sec", predicted.columns)
        self.assertNotIn("Retest_Time_sec", predicted.columns)

        subset = predicted.head(20).copy()
        subset_counts = overview_recommendation_counts(subset)
        subset_vc = subset["AI_Recommendation"].astype(str).str.strip().value_counts()
        self.assertEqual(subset_counts["total_events"], 20)
        self.assertEqual(subset_counts["retest"], int(subset_vc.get("RETEST", 0)))
        self.assertEqual(subset_counts["dont_retest"], int(subset_vc.get("DON'T RETEST", 0)))

    def test_uploaded_pre_retest_data_does_not_alter_outcome_validation_pipeline(self):
        before_flag = self.ml_service.month_12_outcomes_available()
        before_outcomes = self.ml_service.month_12_outcomes
        before_month12 = self.ml_service.datasets["month_12"].copy()
        before_preds = self.ml_service.get_month_12_batch_table().copy()
        before_kpis = self.ml_service.get_month_12_decision_kpis()
        before_val = self.ml_service.get_month_12_validation_table()

        uploaded = self.month12.head(10).copy()
        uploaded["Ground_Truth"] = "RETEST_BENEFICIAL"
        result = self.ml_service.predict_pre_retest_table(uploaded)

        self.assertEqual(len(result), 10)
        self.assertIs(self.ml_service.month_12_outcomes, before_outcomes)
        self.assertEqual(self.ml_service.month_12_outcomes_available(), before_flag)
        pd.testing.assert_frame_equal(self.ml_service.datasets["month_12"], before_month12)
        pd.testing.assert_frame_equal(self.ml_service.get_month_12_batch_table(), before_preds)
        self.assertEqual(self.ml_service.get_month_12_decision_kpis(), before_kpis)
        after_val = self.ml_service.get_month_12_validation_table()
        if before_val is None:
            self.assertIsNone(after_val)
        else:
            pd.testing.assert_frame_equal(after_val, before_val)


if __name__ == "__main__":
    unittest.main()
