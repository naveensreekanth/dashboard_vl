import unittest
import pandas as pd

from retest_ai.kpis.breakdowns import filter_month12_batch_table
from retest_ai.models.service import MLService


class TestMonth12BatchFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = MLService.get_instance().get_month_12_batch_table().copy()
        cls.df["AI_Recommendation"] = cls.df["AI_Recommendation"].astype(str).str.strip()
        cls.n = len(cls.df)
        cls.n_retest = int((cls.df["AI_Recommendation"] == "RETEST").sum())
        cls.n_skip = int((cls.df["AI_Recommendation"] == "DON'T RETEST").sum())

    def test_unfiltered_counts_are_internally_consistent(self):
        self.assertEqual(self.n_retest + self.n_skip, self.n)
        self.assertGreater(self.n, 0)
        self.assertGreater(self.n_retest, 0)
        self.assertGreater(self.n_skip, 0)

    def test_empty_recommendation_filter_is_all_rows(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=[])
        self.assertEqual(len(filtered), self.n)
        filtered_none = filter_month12_batch_table(self.df, rec_filter=None)
        self.assertEqual(len(filtered_none), self.n)

    def test_retest_only(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=["RETEST"])
        self.assertEqual(len(filtered), self.n_retest)
        self.assertTrue((filtered["AI_Recommendation"] == "RETEST").all())

    def test_dont_retest_only(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=["DON'T RETEST"])
        self.assertEqual(len(filtered), self.n_skip)
        self.assertTrue((filtered["AI_Recommendation"] == "DON'T RETEST").all())

    def test_both_selected_is_all_matching_rows(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=["RETEST", "DON'T RETEST"])
        self.assertEqual(len(filtered), self.n)

    def test_clear_filter_returns_all(self):
        only_retest = filter_month12_batch_table(self.df, rec_filter=["RETEST"])
        self.assertEqual(len(only_retest), self.n_retest)
        cleared = filter_month12_batch_table(self.df, rec_filter=[])
        self.assertEqual(len(cleared), self.n)

    def test_combined_recommendation_and_fail_test(self):
        fail_tests = self.df.loc[self.df["AI_Recommendation"] == "RETEST", "Fail_Test"]
        sample_test = fail_tests.iloc[0]
        filtered = filter_month12_batch_table(
            self.df,
            rec_filter=["RETEST"],
            test_filter=[sample_test],
        )
        expected = self.df[
            (self.df["AI_Recommendation"] == "RETEST")
            & (self.df["Fail_Test"] == sample_test)
        ]
        self.assertEqual(len(filtered), len(expected))
        self.assertTrue((filtered["AI_Recommendation"] == "RETEST").all())
        self.assertTrue((filtered["Fail_Test"] == sample_test).all())

    def test_combined_all_filters_intersect(self):
        row = self.df.iloc[0]
        filtered = filter_month12_batch_table(
            self.df,
            rec_filter=[row["AI_Recommendation"]],
            test_filter=[row["Fail_Test"]],
            wafer_filter=[row["Wafer_ID"]],
            site_filter=[row["ATE_Site"]],
            prob_range=(0.0, 1.0),
        )
        self.assertGreaterEqual(len(filtered), 1)
        self.assertTrue((filtered["AI_Recommendation"] == row["AI_Recommendation"]).all())
        self.assertTrue((filtered["Fail_Test"] == row["Fail_Test"]).all())
        self.assertTrue((filtered["Wafer_ID"] == row["Wafer_ID"]).all())
        self.assertTrue((filtered["ATE_Site"] == row["ATE_Site"]).all())

    def test_probability_range_intersects(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=["RETEST"], prob_range=(0.0, 0.29))
        self.assertEqual(len(filtered), 0)
        filtered_hi = filter_month12_batch_table(self.df, rec_filter=["RETEST"], prob_range=(0.30, 1.0))
        self.assertEqual(len(filtered_hi), self.n_retest)

    def test_original_index_preserved(self):
        filtered = filter_month12_batch_table(self.df, rec_filter=["RETEST"])
        expected_index = self.df.index[self.df["AI_Recommendation"] == "RETEST"]
        self.assertTrue(filtered.index.equals(expected_index))

    def test_kpi_counts_match_same_dataframe(self):
        counts = self.df["AI_Recommendation"].value_counts()
        self.assertEqual(int(counts.get("RETEST", 0)), self.n_retest)
        self.assertEqual(int(counts.get("DON'T RETEST", 0)), self.n_skip)


if __name__ == "__main__":
    unittest.main()
