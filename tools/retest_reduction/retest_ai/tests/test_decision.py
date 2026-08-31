import unittest
import numpy as np
from retest_ai.decision.decision_policy import (
    DOCX_REFERENCE_THRESHOLD,
    POLICY_LABEL,
    apply_decision_policy,
    apply_batch_decision_policy,
)


class TestDecisionPolicy(unittest.TestCase):
    def test_threshold_is_docx_reference_30_percent(self):
        self.assertEqual(DOCX_REFERENCE_THRESHOLD, 0.30)
        self.assertNotEqual(DOCX_REFERENCE_THRESHOLD, 0.50)
        self.assertIn("subject to validation", POLICY_LABEL.lower())

    def test_boundary_029_is_dont_retest(self):
        result = apply_decision_policy(0.29)
        self.assertEqual(result["recommendation"], "DON'T RETEST")
        self.assertEqual(result["probability"], 0.29)

    def test_boundary_030_is_retest(self):
        result = apply_decision_policy(0.30)
        self.assertEqual(result["recommendation"], "RETEST")
        self.assertEqual(result["probability"], 0.30)

    def test_high_probability_retest(self):
        result = apply_decision_policy(0.78)
        self.assertEqual(result["recommendation"], "RETEST")
        self.assertAlmostEqual(result["probability"], 0.78)
        self.assertAlmostEqual(result["probability_percent"], 78.0)

    def test_policy_does_not_alter_probability(self):
        raw = 0.2214
        result = apply_decision_policy(raw)
        self.assertAlmostEqual(result["probability"], raw)
        self.assertEqual(result["recommendation"], "DON'T RETEST")

    def test_batch_policy(self):
        df = apply_batch_decision_policy([0.29, 0.30, 0.81])
        self.assertEqual(df["AI_Recommendation"].tolist(), ["DON'T RETEST", "RETEST", "RETEST"])
        self.assertTrue(np.allclose(df["P(RETEST_BENEFICIAL)"].values, [0.29, 0.30, 0.81]))

    def test_no_50_percent_default(self):
        import inspect
        import retest_ai.decision.decision_policy as dp
        src = inspect.getsource(dp)
        self.assertNotIn("DEFAULT_DEMO_THRESHOLD", src)
        self.assertNotIn("0.50", src)
        self.assertEqual(DOCX_REFERENCE_THRESHOLD, 0.30)


if __name__ == "__main__":
    unittest.main()
