import unittest
import pandas as pd
import numpy as np
from retest_ai.data.validation import validate_dataset, check_for_leakage, ValidationError
from retest_ai.data.ingestion import load_dataset
from retest_ai.config.settings import ALL_MODEL_FEATURES, DATA_FILES, LEAKAGE_COLS

class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.valid_df = pd.DataFrame({
            "Device_ID": ["DEV001", "DEV002"],
            "Wafer_ID": ["W001", "W002"],
            "ATE_Site": [1, 2],
            "Failure_Event": [1, 1],
            "Fail_Test": ["Scan_145", "MBIST_03"],
            "Fail_Bin": [23, 45],
            "First_Result": ["FAIL", "FAIL"],
            "Voltage_V": [0.90, 0.85],
            "Temperature_C": [25, 55],
            "First_Test_Time_sec": [45.0, 60.0],
            "Test_Month": [0, 0],
            "Ground_Truth": ["RETEST_BENEFICIAL", "PERSISTENT_FAILURE"]
        })

    def test_valid_training_dataset(self):
        report = validate_dataset(self.valid_df, is_inference=False)
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)

    def test_missing_required_feature(self):
        df_missing = self.valid_df.drop(columns=["Temperature_C"])
        report = validate_dataset(df_missing, is_inference=False)
        self.assertFalse(report["is_valid"])
        self.assertIn("Temperature_C", report["missing_features"])

    def test_inference_leakage_detection(self):
        for leak_col in LEAKAGE_COLS:
            df_leak = self.valid_df.copy()
            df_leak[leak_col] = ["SAMPLE1", "SAMPLE2"]
            report = validate_dataset(df_leak, is_inference=True, strict_leakage_check=True)
            self.assertFalse(report["is_valid"], f"Leakage column {leak_col} was not detected!")
            self.assertIn(leak_col, report["leakage_detected"])

    def test_missing_values_are_reported_not_filled(self):
        from retest_ai.data.preprocessing import prepare_xy
        from retest_ai.data.validation import ValidationError
        df_missing = self.valid_df.copy()
        df_missing.loc[0, "Voltage_V"] = None
        with self.assertRaises((ValidationError, ValueError)):
            prepare_xy(df_missing, is_inference=True)

    def test_feature_matrix_rejects_leakage_columns(self):
        from retest_ai.data.validation import assert_no_leakage_in_feature_matrix, ValidationError
        with self.assertRaises(ValidationError):
            assert_no_leakage_in_feature_matrix(["Wafer_ID", "Ground_Truth"])
        with self.assertRaises(ValidationError):
            assert_no_leakage_in_feature_matrix(["Voltage_V", "Retest_Time_sec"])

    def test_real_workbooks_loading_and_validation(self):
        df_m0 = load_dataset("month_0")
        report_m0 = validate_dataset(df_m0, is_inference=False)
        self.assertTrue(report_m0["is_valid"])
        
        df_m6 = load_dataset("month_6")
        report_m6 = validate_dataset(df_m6, is_inference=False)
        self.assertTrue(report_m6["is_valid"])
        
        df_m12 = load_dataset("month_12")
        report_m12 = validate_dataset(df_m12, is_inference=True, strict_leakage_check=True)
        self.assertTrue(report_m12["is_valid"])
        self.assertEqual(len(report_m12["leakage_detected"]), 0)

if __name__ == "__main__":
    unittest.main()
