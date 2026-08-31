import unittest
from fastapi.testclient import TestClient
from retest_ai.api.main import app
from retest_ai.decision.decision_policy import DOCX_REFERENCE_THRESHOLD, POLICY_LABEL


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.valid_event = {
            "Device_ID": "DEV001",
            "Failure_Event": 1,
            "Wafer_ID": "W003",
            "ATE_Site": 4,
            "Fail_Test": "MBIST_03",
            "Fail_Bin": 45,
            "First_Result": "FAIL",
            "Voltage_V": 0.91,
            "Temperature_C": 60.0,
            "First_Test_Time_sec": 48.5,
            "Test_Month": 12
        }

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "ATE Retest-Benefit Prediction AI")

    def test_model_info_endpoint(self):
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model", data)
        self.assertEqual(data["target"], "RETEST_BENEFICIAL")
        self.assertIn("Wafer_ID", data["features"])
        self.assertIn("Voltage_V", data["features"])
        self.assertEqual(data["decision_policy_threshold"], DOCX_REFERENCE_THRESHOLD)
        self.assertEqual(data["decision_policy_label"], POLICY_LABEL)

    def test_single_predict_endpoint_valid(self):
        response = self.client.post("/predict", json=self.valid_event)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("probability_retest_beneficial", data)
        self.assertIn("probability_percent", data)
        self.assertIn("recommendation", data)
        self.assertIn(data["recommendation"], ["RETEST", "DON'T RETEST"])
        self.assertTrue(0.0 <= data["probability_retest_beneficial"] <= 1.0)
        self.assertEqual(data["Device_ID"], "DEV001")
        self.assertEqual(data["Failure_Event"], 1)
        self.assertEqual(data["policy_threshold"], 0.30)
        self.assertIn("estimated_retest_time_sec", data)
        self.assertGreaterEqual(data["estimated_retest_time_sec"], 0.0)
        self.assertIn("predicted_retest_time_sec", data)
        if data["recommendation"] == "DON'T RETEST":
            self.assertEqual(data["predicted_retest_time_sec"], 0.0)
        else:
            self.assertEqual(data["predicted_retest_time_sec"], data["estimated_retest_time_sec"])
        if data["probability_retest_beneficial"] >= 0.30:
            self.assertEqual(data["recommendation"], "RETEST")
        else:
            self.assertEqual(data["recommendation"], "DON'T RETEST")

    def test_single_predict_rejects_leakage_fields(self):
        leakage_payloads = [
            {**self.valid_event, "Ground_Truth": "RETEST_BENEFICIAL"},
            {**self.valid_event, "Retest_Result": "PASS"},
            {**self.valid_event, "Final_Result": "PASS"},
            {**self.valid_event, "Retest_Time_sec": 15.4},
            {**self.valid_event, "True_Retest_Pass_Probability": 0.85},
            {**self.valid_event, "AI_Retest_Probability": 0.78},
            {**self.valid_event, "AI_Recommendation": "RETEST"}
        ]
        for payload in leakage_payloads:
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 422, f"Leakage payload was not rejected: {payload.keys()}")

    def test_batch_predict_endpoint_valid(self):
        batch_payload = {
            "events": [
                {**self.valid_event, "Device_ID": "DEV001", "Failure_Event": 1},
                {**self.valid_event, "Device_ID": "DEV002", "Failure_Event": 2, "Fail_Test": "Scan_145"}
            ]
        }
        response = self.client.post("/predict/batch", json=batch_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 2)
        for pred in data["predictions"]:
            self.assertTrue(0.0 <= pred["probability_retest_beneficial"] <= 1.0)
            self.assertIn(pred["recommendation"], ["RETEST", "DON'T RETEST"])

    def test_batch_predict_rejects_leakage(self):
        batch_payload = {
            "events": [
                {**self.valid_event, "Device_ID": "DEV001", "Retest_Result": "PASS"}
            ]
        }
        response = self.client.post("/predict/batch", json=batch_payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
