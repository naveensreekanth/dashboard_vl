import unittest
import numpy as np
import pandas as pd
from retest_ai.data.ingestion import load_dataset
from retest_ai.data.preprocessing import prepare_xy
from retest_ai.models.logistic_model import build_logistic_model
from retest_ai.models.xgboost_model import build_xgboost_model
from retest_ai.models.gradient_boosting import build_gradient_boosting_model
from retest_ai.models.calibration import calibrate_model, evaluate_calibration
from retest_ai.models.trainer import train_and_compare_models, train_final_deployment_model
from retest_ai.models.service import MLService

class TestModels(unittest.TestCase):

    def setUp(self):
        self.df_m0 = load_dataset("month_0")
        self.df_m6 = load_dataset("month_6")
        self.df_m12 = load_dataset("month_12")
        self.X_train, self.y_train = prepare_xy(self.df_m0, is_inference=False)
        self.X_val, self.y_val = prepare_xy(self.df_m6, is_inference=False)

    def test_logistic_model_training_and_prob_range(self):
        model = build_logistic_model()
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_val)[:, 1]
        self.assertEqual(len(probs), len(self.df_m6))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_xgboost_model_training_and_prob_range(self):
        model = build_xgboost_model()
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_val)[:, 1]
        self.assertEqual(len(probs), len(self.df_m6))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_gradient_boosting_training_and_prob_range(self):
        model = build_gradient_boosting_model()
        model.fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_val)[:, 1]
        self.assertEqual(len(probs), len(self.df_m6))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_temporal_validation_comparison_table(self):
        comp = train_and_compare_models(self.df_m0, self.df_m6)
        self.assertIn("comparison_table", comp)
        df_table = comp["comparison_table"]
        self.assertEqual(len(df_table), 3)
        expected_cols = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC", "Brier Score", "Log Loss"]
        for col in expected_cols:
            self.assertIn(col, df_table.columns)

    def test_month_12_inference_generates_125_probabilities(self):
        ml_service = MLService.get_instance()
        df_m12_batch = ml_service.get_month_12_batch_table()
        self.assertEqual(len(df_m12_batch), 125)
        self.assertIn("P(RETEST_BENEFICIAL)", df_m12_batch.columns)
        self.assertIn("AI_Recommendation", df_m12_batch.columns)
        probs = df_m12_batch["P(RETEST_BENEFICIAL)"].values
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertTrue(set(df_m12_batch["AI_Recommendation"].unique()).issubset({"RETEST", "DON'T RETEST"}))
        self.assertNotIn("Ground_Truth", df_m12_batch.columns)
        self.assertNotIn("Retest_Time_sec", df_m12_batch.columns)
        self.assertIn("Estimated_Retest_Time_sec", df_m12_batch.columns)
        self.assertTrue((df_m12_batch["Estimated_Retest_Time_sec"] >= 0).all())

    def test_month_12_does_not_use_private_outcomes_for_prediction(self):
        ml_service = MLService.get_instance()
        self.assertFalse(ml_service.month_12_outcomes_available())
        self.assertIsNone(ml_service.get_month_12_decision_kpis())

    def test_selection_is_evidence_based(self):
        comp = train_and_compare_models(self.df_m0, self.df_m6)
        self.assertIn(comp["best_model_name"], ["XGBoost", "Logistic Regression", "Gradient Boosting"])
        self.assertTrue(len(comp["selection_reason"]) > 0)

if __name__ == "__main__":
    unittest.main()
